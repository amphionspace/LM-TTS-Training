"""Qwen3-TTS non-streaming training with a text Base initialization."""

import json
from pathlib import Path

import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig, Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSSpeakerEncoder,
    Qwen3TTSTalkerForConditionalGeneration,
)
from torch import nn
from transformers import AutoModel

from ..artifacts import file_hash
from .codec import CodecTTSModel


class TTSModel(CodecTTSModel):
    def __init__(self, config, speaker_config=None):
        super().__init__()
        backends = {c._attn_implementation for c in (config, config.code_predictor_config)}
        if not backends <= {"flash_attention_2", "sdpa"}:
            raise ValueError("Talker and Code Predictor require flash_attention_2 or sdpa")
        self.config = config
        self.talker = Qwen3TTSTalkerForConditionalGeneration(config)
        self.speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_config) if speaker_config else None
        if self.speaker_encoder is not None:
            from .speaker import deterministic_speaker_padding

            deterministic_speaker_padding(self.speaker_encoder)
        if (speaker_config is None and not hasattr(config, "lm_tts_text_projection")) or getattr(
            config, "lm_tts_text_projection", "mlp"
        ) == "identity":
            if config.text_hidden_size != config.hidden_size:
                raise ValueError("Direct text embeddings require matching backbone width")
            self.talker.text_projection = nn.Identity()
        if getattr(config, "lm_tts_freeze_text_frontend", False):
            self.talker.model.text_embedding.requires_grad_(False)
            self.talker.text_projection.requires_grad_(False)
        if self.speaker_encoder is not None and getattr(
            config, "lm_tts_freeze_speaker_encoder", False
        ):
            self.speaker_encoder.requires_grad_(False).eval()
        self.groups = config.num_code_groups
        self.code_size = config.code_predictor_config.vocab_size
        self.bos = config.codec_bos_id
        self.eos = config.codec_eos_token_id

    def train(self, mode=True):
        super().train(mode)
        if self.speaker_encoder is not None and getattr(
            self.config, "lm_tts_freeze_speaker_encoder", False
        ):
            self.speaker_encoder.eval()
        return self

    def pretrained_modules(self, include_text=False):
        modules = [self.talker.model.layers, self.talker.model.norm]
        if include_text:
            modules.append(self.talker.model.text_embedding)
        return modules

    @classmethod
    def from_assembled(
        cls,
        directory,
        load_weights=True,
        attn_implementation="flash_attention_2",
        use_speaker_embedding=None,
    ):
        if attn_implementation not in {"flash_attention_2", "sdpa"}:
            raise ValueError("TTSModel requires flash_attention_2 or sdpa")
        from .assembly.common import load_prefix

        directory = Path(directory)
        if not (directory / "ASSEMBLY_COMPLETE").exists():
            raise ValueError("Require a completed output of scripts/assemble_qwen3_tts.py")
        report = json.loads((directory / "assembly_report.json").read_text())
        for name, expected in report["artifact_sha256"].items():
            if name == "config.json" or (
                load_weights and name.startswith("model") and name.endswith(".safetensors")
            ):
                if file_hash(directory / name) != expected:
                    raise ValueError(f"Assembled artifact changed: {name}")
        raw = json.loads((directory / "config.json").read_text())
        model = cls.from_config(raw, attn_implementation, use_speaker_embedding)
        if load_weights:
            model.load_state_dict(load_prefix(directory, ""), strict=True)
        return model

    @classmethod
    def from_config(cls, raw, attn_implementation="sdpa", use_speaker_embedding=None):
        if attn_implementation not in {"flash_attention_2", "sdpa"}:
            raise ValueError("TTSModel requires flash_attention_2 or sdpa")
        config = Qwen3TTSConfig.from_dict(raw)
        if use_speaker_embedding is not None:
            if type(use_speaker_embedding) is not bool:
                raise ValueError("use_speaker_embedding must be a boolean")
            config.talker_config.lm_tts_use_speaker_embedding = use_speaker_embedding
        config.talker_config._attn_implementation = attn_implementation
        # Each predictor sequence has 16 positions. SDPA avoids FA2 deterministic
        # backward workspace failures on large frame batches, without any padding.
        config.talker_config.code_predictor_config._attn_implementation = "sdpa"
        return cls(config.talker_config, config.speaker_encoder_config)

    def initialize_backbone(self, path):
        base = AutoModel.from_pretrained(
            path, torch_dtype=torch.float32, attn_implementation="sdpa"
        )
        assert base.config.hidden_size == self.config.hidden_size
        self.talker.model.layers.load_state_dict(base.layers.state_dict(), strict=True)
        self.talker.model.norm.load_state_dict(base.norm.state_dict(), strict=True)
        self.talker.model.text_embedding.load_state_dict(
            base.embed_tokens.state_dict(), strict=True
        )
        return {
            "source": str(path),
            "loaded": ["layers", "norm", "text_embedding"],
            "new": ["codec_embedding", "codec_head", "code_predictor"],
            "parameters": sum(p.numel() for p in self.parameters()),
        }


def make_config(backbone_config=None, tiny=False):
    if tiny:
        dims = dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            text_hidden_size=64,
            text_vocab_size=256,
        )
        depth = {k: v for k, v in dims.items() if not k.startswith("text_")}
        depth["num_hidden_layers"] = 2
        sections = [2, 3, 3]
    else:
        keys = [
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "rope_theta",
            "hidden_act",
            "attention_bias",
        ]
        dims = {k: getattr(backbone_config, k) for k in keys}
        dims.update(
            text_hidden_size=backbone_config.hidden_size, text_vocab_size=backbone_config.vocab_size
        )
        depth = {k: v for k, v in dims.items() if not k.startswith("text_")}
        depth["num_hidden_layers"] = 5
        sections = [24, 20, 20]
        if dims["head_dim"] != 128:
            raise ValueError("The production configuration expects Qwen3-0.6B head_dim=128")
    config = Qwen3TTSTalkerConfig(
        **dims,
        vocab_size=3072,
        num_code_groups=16,
        codec_bos_id=2149,
        codec_eos_token_id=2150,
        codec_pad_id=2148,
        rope_scaling={"rope_type": "default", "mrope_section": sections, "interleaved": True},
        code_predictor_config={
            **depth,
            "vocab_size": 2048,
            "num_code_groups": 16,
            "use_cache": False,
        },
        use_cache=False,
    )
    config._attn_implementation = "sdpa" if tiny else "flash_attention_2"
    config.code_predictor_config._attn_implementation = config._attn_implementation
    return config
