"""LFM text backbone adapted to the shared codec prediction protocol."""

import json
from pathlib import Path

import torch
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    Qwen3TTSTalkerResizeMLP,
)
from torch import nn
from transformers import Lfm2Config

from ...artifacts import file_hash
from ..assembly.common import load_prefix
from ..codec import CodecTTSModel
from .backbone import LfmCodecBackbone
from .configuration import LfmTTSConfig
from .frontend import initialize_paired_frontend


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
        if config.lm_tts_text_projection == "mlp":
            native = self.model.embed_tokens
            # Keep audio-module and new-token RNG streams identical to the native
            # recipe, so frontend ablations start from the same remaining weights.
            with torch.random.fork_rng(devices=[]):
                self.model.embed_tokens = nn.Embedding(
                    config.text_vocab_size, config.text_hidden_size
                )
                self.text_projection = Qwen3TTSTalkerResizeMLP(
                    config.text_hidden_size,
                    config.text_hidden_size,
                    config.hidden_size,
                    "silu",
                    bias=True,
                )
            initialize_paired_frontend(self.model.embed_tokens, self.text_projection, native.weight)


class LfmTTSModel(CodecTTSModel):
    def pretrained_modules(self, include_text=False):
        modules = [self.talker.model.layers, self.talker.model.embedding_norm]
        if include_text:
            modules.append(self.talker.model.embed_tokens)
        return modules

    def __init__(self, config):
        super().__init__()
        if not isinstance(config, LfmTTSConfig):
            backend = config._attn_implementation
            config = LfmTTSConfig.from_dict(config.to_dict())
            config._attn_implementation = backend
            config.code_predictor_config._attn_implementation = "sdpa"
        if getattr(config, "lm_tts_use_speaker_embedding", False):
            raise ValueError("LFM assembly supports pure codec conditioning only")
        if config.lm_tts_freeze_text_frontend:
            raise ValueError("LFM requires a trainable text frontend")
        if config.lm_tts_text_projection not in {"identity", "mlp"}:
            raise ValueError("LFM text projection must be identity or mlp")
        if (
            config.hidden_size != config.lm_tts_lfm_config["hidden_size"]
            or config.text_vocab_size != config.lm_tts_lfm_config["vocab_size"]
        ):
            raise ValueError("LFM text dimensions disagree with the native backbone config")
        multiplier = 2 if config.lm_tts_text_projection == "mlp" else 1
        if config.text_hidden_size != multiplier * config.hidden_size:
            raise ValueError(
                "LFM text width must be native for identity or doubled for the SiLU MLP"
            )
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
        model = cls.from_config(raw, attn_implementation, use_speaker_embedding)
        if load_weights:
            model.load_state_dict(load_prefix(directory, ""), strict=True)
        return model

    @classmethod
    def from_config(cls, raw, attn_implementation="sdpa", use_speaker_embedding=None):
        if use_speaker_embedding not in (None, False):
            raise ValueError("LFM requires model.use_speaker_embedding=false")
        if attn_implementation not in {"sdpa", "flash_attention_2"}:
            raise ValueError("LFM requires sdpa or flash_attention_2")
        config = LfmTTSConfig.from_dict(raw["talker_config"])
        config._attn_implementation = attn_implementation
        config.code_predictor_config._attn_implementation = "sdpa"
        return cls(config)
