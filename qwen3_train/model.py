"""Offline text-prefix TTS using official Qwen3-TTS Talker and depth predictor.

The offline training protocol differs from the official generation protocol.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig, Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration, Qwen3TTSSpeakerEncoder
from transformers import AutoModel


class TTSModel(nn.Module):
    def __init__(self, config, speaker_config=None):
        super().__init__()
        self.config = config
        self.talker = Qwen3TTSTalkerForConditionalGeneration(config)
        self.speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_config) if speaker_config else None
        if self.speaker_encoder is not None:
            from .speaker import deterministic_speaker_padding
            deterministic_speaker_padding(self.speaker_encoder)
        if speaker_config is None:
            self.talker.text_projection = nn.Identity()
        self.groups = config.num_code_groups
        self.code_size = config.code_predictor_config.vocab_size
        self.bos = config.codec_bos_id
        self.eos = config.codec_eos_token_id

    @classmethod
    def from_assembled(cls, directory, load_weights=True):
        from .assembly import load_prefix, sha256
        directory = Path(directory)
        if not (directory / "ASSEMBLY_COMPLETE").exists():
            raise ValueError("Require a completed output of scripts/assemble_qwen3_tts.py")
        report = json.loads((directory / "assembly_report.json").read_text())
        for name, expected in report["artifact_sha256"].items():
            if name == "config.json" or (load_weights and name.startswith("model") and name.endswith(".safetensors")):
                if sha256(directory / name) != expected:
                    raise ValueError(f"Assembled artifact changed: {name}")
        config = Qwen3TTSConfig.from_dict(json.loads((directory / "config.json").read_text()))
        config.talker_config._attn_implementation = "sdpa"
        config.talker_config.code_predictor_config._attn_implementation = "sdpa"
        model = cls(config.talker_config, config.speaker_encoder_config)
        if load_weights:
            model.talker.load_state_dict(load_prefix(directory, "talker."), strict=True)
            model.speaker_encoder.load_state_dict(load_prefix(directory, "speaker_encoder."), strict=True)
        return model

    def initialize_backbone(self, path):
        base = AutoModel.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="sdpa")
        assert base.config.hidden_size == self.config.hidden_size
        self.talker.model.layers.load_state_dict(base.layers.state_dict(), strict=True)
        self.talker.model.norm.load_state_dict(base.norm.state_dict(), strict=True)
        self.talker.model.text_embedding.load_state_dict(base.embed_tokens.state_dict(), strict=True)
        return {"source": str(path), "loaded": ["layers", "norm", "text_embedding"],
                "new": ["codec_embedding", "codec_head", "code_predictor"],
                "parameters": sum(p.numel() for p in self.parameters())}

    def frame_embeddings(self, codes):
        value = self.talker.model.codec_embedding(codes[..., 0])
        for g in range(1, self.groups):
            value = value + self.talker.code_predictor.get_input_embeddings()[g - 1](codes[..., g])
        return value

    def hidden(self, batch):
        text = self.talker.text_projection(self.talker.model.text_embedding(batch["text_ids"]))
        if self.speaker_encoder is not None:
            speaker = self.speaker_encoder(batch["speaker_mels"]).unsqueeze(1)
            text = torch.cat([text, speaker.to(text.dtype)], dim=1)
        frames = self.frame_embeddings(batch["codes"])
        bos = self.talker.model.codec_embedding.weight[self.bos].expand(text.shape[0], 1, -1)
        # [text, optional speaker, audio BOS, historical frames]; BOS predicts frame_0.
        embeds = torch.cat([text, bos, frames], dim=1)
        mask = torch.cat([batch["text_mask"], torch.ones(text.shape[0], 2 if self.speaker_encoder is not None else 1, dtype=torch.bool, device=text.device), batch["frame_mask"]], dim=1)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        outputs = self.talker.model(inputs_embeds=embeds, attention_mask=mask,
                                    position_ids=positions, use_cache=False)
        return outputs.last_hidden_state[:, text.shape[1]:]

    def forward(self, batch, mode="loss"):
        hidden = self.hidden(batch)
        if mode == "next_frame":
            # Generation uses one unpadded sample on every rank.
            h = hidden[:, -1]
            logits = self.talker.codec_head(h).float()
            allowed = torch.cat([logits[:, :self.code_size], logits[:, self.eos:self.eos + 1]], dim=-1)
            first = allowed.argmax(-1)
            stop = first == self.code_size
            codes = [first.clamp_max(self.code_size - 1)]
            inputs = [h.unsqueeze(1), self.talker.model.codec_embedding(codes[0]).unsqueeze(1)]
            predictor = self.talker.code_predictor
            for g in range(1, self.groups):
                out = predictor.model(inputs_embeds=predictor.small_to_mtp_projection(torch.cat(inputs, dim=1)), use_cache=False)
                code = predictor.lm_head[g - 1](out.last_hidden_state[:, -1]).argmax(-1)
                codes.append(code)
                if g < self.groups - 1:
                    inputs.append(predictor.get_input_embeddings()[g - 1](code).unsqueeze(1))
            return torch.stack(codes, dim=-1), stop
        if mode != "loss":
            raise ValueError(mode)
        codes, valid = batch["codes"], batch["frame_mask"].bool()
        lengths = valid.sum(-1)
        first_labels = codes.new_full(hidden.shape[:2], -100)
        first_labels[:, :-1] = torch.where(valid, codes[..., 0], -100)
        first_labels.scatter_(1, lengths[:, None], self.eos)
        logits = self.talker.codec_head(hidden).float()
        first_loss = F.cross_entropy(logits.flatten(0, 1), first_labels.flatten(), ignore_index=-100, reduction="sum")
        # h_t precedes frame_t. No target-frame leakage into the Talker.
        h = hidden[:, :-1][valid]
        target = codes[valid]
        inputs = [h.unsqueeze(1), self.talker.model.codec_embedding(target[:, 0]).unsqueeze(1)]
        predictor = self.talker.code_predictor
        for g in range(1, self.groups - 1):
            inputs.append(predictor.get_input_embeddings()[g - 1](target[:, g]).unsqueeze(1))
        out = predictor.model(inputs_embeds=predictor.small_to_mtp_projection(torch.cat(inputs, dim=1)), use_cache=False)
        group_sums = torch.stack([
            F.cross_entropy(predictor.lm_head[g - 1](out.last_hidden_state[:, g]).float(), target[:, g], reduction="sum")
            for g in range(1, self.groups)
        ])
        return {"first_sum": first_loss, "residual_sum": group_sums.sum(),
                "group_sums": group_sums.detach(), "first_count": (first_labels != -100).sum(),
                "frame_count": valid.sum()}


def make_config(backbone_config=None, tiny=False):
    if tiny:
        dims = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                    text_hidden_size=64, text_vocab_size=256)
        depth = {k: v for k, v in dims.items() if not k.startswith("text_")}
        depth["num_hidden_layers"] = 2
        sections = [2, 3, 3]
    else:
        keys = ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "rms_norm_eps", "rope_theta", "hidden_act", "attention_bias"]
        dims = {k: getattr(backbone_config, k) for k in keys}
        dims.update(text_hidden_size=backbone_config.hidden_size, text_vocab_size=backbone_config.vocab_size)
        depth = {k: v for k, v in dims.items() if not k.startswith("text_")}
        depth["num_hidden_layers"] = 5
        sections = [24, 20, 20]
        if dims["head_dim"] != 128:
            raise ValueError("The production configuration expects Qwen3-0.6B head_dim=128")
    config = Qwen3TTSTalkerConfig(**dims, vocab_size=3072, num_code_groups=16,
        codec_bos_id=2149, codec_eos_token_id=2150, codec_pad_id=2148,
        rope_scaling={"rope_type": "default", "mrope_section": sections, "interleaved": True},
        code_predictor_config={**depth, "vocab_size": 2048, "num_code_groups": 16, "use_cache": False},
        use_cache=False)
    config._attn_implementation = "sdpa"
    config.code_predictor_config._attn_implementation = "sdpa"
    return config
