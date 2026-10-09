"""LFM2 text backbone with a Qwen codec-depth predictor and isolated packed convolution."""

import json
from pathlib import Path

import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
)
from torch import nn
from torch.nn import functional as F
from transformers import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2DecoderLayer,
    Lfm2Model,
    Lfm2PreTrainedModel,
    Lfm2RMSNorm,
    Lfm2RotaryEmbedding,
    Lfm2ShortConv,
)

from ..artifacts import file_hash
from .assembly import load_prefix
from .codec import CodecTTSModel


class PackedShortConv(Lfm2ShortConv):
    def forward(self, hidden_states, position_ids):
        # Position IDs restart at each utterance. Mask every lag before summation,
        # including sequences shorter than the kernel; FA2 only isolates attention.
        b, c, x = self.in_proj(hidden_states).chunk(3, dim=-1)
        bx = b * x
        result = torch.zeros_like(bx, dtype=torch.float32)
        weights = self.conv.weight[:, 0].float()
        for lag in range(self.L_cache):
            if lag >= bx.shape[1]:
                break
            shifted = bx if lag == 0 else F.pad(bx[:, :-lag], (0, 0, lag, 0))
            valid = (position_ids >= lag).unsqueeze(-1)
            result = result + shifted.float() * weights[:, self.L_cache - 1 - lag] * valid
        if self.conv.bias is not None:
            result = result + self.conv.bias.float()
        return self.out_proj(c * result.to(c.dtype))


class PackedLfmLayer(Lfm2DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        if not self.is_attention_layer:
            self.conv = PackedShortConv(config, layer_idx)

    def forward(self, hidden_states, position_embeddings, position_ids=None, **kwargs):
        if self.is_attention_layer:
            return super().forward(
                hidden_states, position_embeddings, position_ids=position_ids, **kwargs
            )
        hidden_states = hidden_states + self.conv(self.operator_norm(hidden_states), position_ids)
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class LfmCodecBackbone(Lfm2Model):
    def __init__(self, config, codec_vocab_size):
        Lfm2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.codec_embedding = nn.Embedding(codec_vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [PackedLfmLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Lfm2RotaryEmbedding(config=config)
        self.pos_emb = Lfm2RotaryEmbedding(config)
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    @property
    def text_embedding(self):
        # Expose the shared input protocol without registering the same weight twice.
        return self.embed_tokens

    def forward(self, *args, use_cache=False, past_key_values=None, **kwargs):
        if use_cache or past_key_values is not None:
            raise ValueError(
                "LFM codec training uses complete prefixes, without a generation cache"
            )
        return super().forward(*args, use_cache=False, **kwargs)


class LfmTalker(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone = Lfm2Config.from_dict(config.lm_tts_lfm_config)
        backbone._attn_implementation = config._attn_implementation
        self.model = LfmCodecBackbone(backbone, config.vocab_size)
        self.text_projection = nn.Identity()
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        nn.init.normal_(self.codec_head.weight, std=config.initializer_range)
        self.code_predictor = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
            config.code_predictor_config, config
        )


class LfmTTSModel(CodecTTSModel):
    def __init__(self, config):
        super().__init__()
        if getattr(config, "lm_tts_use_speaker_embedding", False):
            raise ValueError("LFM assembly supports pure codec conditioning only")
        if config.hidden_size != config.text_hidden_size:
            raise ValueError("LFM text embeddings must match the Talker width")
        self.config = config
        self.talker = LfmTalker(config)
        self.speaker_encoder = None
        self.groups = config.num_code_groups
        self.code_size = config.code_predictor_config.vocab_size
        self.bos = config.codec_bos_id
        self.eos = config.codec_eos_token_id

    @classmethod
    def from_assembled(
        cls,
        directory,
        load_weights=True,
        attn_implementation="flash_attention_2",
        use_speaker_embedding=None,
    ):
        directory = Path(directory)
        if not (directory / "ASSEMBLY_COMPLETE").is_file():
            raise ValueError("Require a completed LFM assembly")
        if use_speaker_embedding not in (None, False):
            raise ValueError("LFM assembly requires model.use_speaker_embedding=false")
        if attn_implementation not in {"sdpa", "flash_attention_2"}:
            raise ValueError("LFM requires sdpa or flash_attention_2")
        report = json.loads((directory / "assembly_report.json").read_text())
        for name, expected in report["artifact_sha256"].items():
            if name == "config.json" or (load_weights and name.endswith(".safetensors")):
                if file_hash(directory / name) != expected:
                    raise ValueError(f"Assembled artifact changed: {name}")
        raw = json.loads((directory / "config.json").read_text())
        config = Qwen3TTSTalkerConfig.from_dict(raw["talker_config"])
        config._attn_implementation = attn_implementation
        config.code_predictor_config._attn_implementation = "sdpa"
        model = cls(config)
        if load_weights:
            model.load_state_dict(load_prefix(directory, ""), strict=True)
        return model
