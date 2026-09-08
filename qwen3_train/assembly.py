"""Audited initialization of an official-format Qwen3-TTS checkpoint."""
import copy
import gc
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoTokenizer
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

BACKBONE_FIELDS = (
    "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "rms_norm_eps", "rope_theta", "hidden_act",
    "attention_bias", "attention_dropout", "sliding_window",
)
TEXT_SPECIALS = {"tts_pad_token_id": "<tts_pad>", "tts_bos_token_id": "<tts_text_bos>",
                 "tts_eos_token_id": "<tts_text_eod>", "im_start_token_id": "<|im_start|>",
                 "im_end_token_id": "<|im_end|>"}


def vocabulary_plan(source_vocab, target_vocab, embedding_rows):
    conflicts = {token: [index, target_vocab.get(token)] for token, index in source_vocab.items()
                 if target_vocab.get(token) != index}
    if conflicts:
        raise ValueError(f"Shared tokenizer IDs differ: {list(conflicts.items())[:5]}")
    added = {token: index for token, index in target_vocab.items() if token not in source_vocab}
    rows = max(embedding_rows, max(target_vocab.values()) + 1)
    return {"source_token_count": len(source_vocab), "target_token_count": len(target_vocab),
            "source_embedding_rows": embedding_rows, "target_embedding_rows": rows,
            "physical_rows_added": rows - embedding_rows, "added_tokens": added,
            "shared_token_ids_verified": len(source_vocab)}


def audit_sources(backbone, template):
    base = AutoConfig.from_pretrained(backbone)
    raw = json.loads((Path(template) / "config.json").read_text())
    tts = Qwen3TTSConfig.from_dict(raw)
    if base.model_type != "qwen3" or tts.tts_model_type != "base":
        raise ValueError("Require a Qwen3 text model and a Qwen3-TTS Base template")
    if tts.tokenizer_type != "qwen3_tts_tokenizer_12hz":
        raise ValueError("Only the 12Hz codec is supported")
    differences = {key: {"backbone": getattr(base, key), "talker": getattr(tts.talker_config, key)}
                   for key in BACKBONE_FIELDS if getattr(base, key) != getattr(tts.talker_config, key)}
    if differences:
        raise ValueError(f"Backbone weights are not structurally compatible: {differences}")
    source_tokenizer = AutoTokenizer.from_pretrained(backbone)
    target_tokenizer = AutoTokenizer.from_pretrained(template)
    vocab = vocabulary_plan(source_tokenizer.get_vocab(), target_tokenizer.get_vocab(), base.vocab_size)
    for field, token in TEXT_SPECIALS.items():
        if target_tokenizer.convert_tokens_to_ids(token) != getattr(tts, field):
            raise ValueError(f"TTS special token mismatch: {field}")
    if tts.speaker_encoder_config.enc_dim != tts.talker_config.hidden_size:
        raise ValueError("Speaker vector width must match Talker width")
    if tts.talker_config.num_code_groups != 16 or tts.talker_config.code_predictor_config.vocab_size != 2048:
        raise ValueError("Unexpected 12Hz codec dimensions")
    modified = copy.deepcopy(raw)
    modified["talker_config"]["text_hidden_size"] = base.hidden_size
    modified["talker_config"]["text_vocab_size"] = vocab["target_embedding_rows"]
    config = Qwen3TTSConfig.from_dict(modified)
    config._attn_implementation = "sdpa"
    config.talker_config._attn_implementation = "sdpa"
    config.talker_config.code_predictor_config._attn_implementation = "sdpa"
    report = {
        "backbone_compatibility": {k: getattr(base, k) for k in BACKBONE_FIELDS},
        "text_vocabulary": vocab,
        "adaptations": {
            "text_hidden_size": {"official": raw["talker_config"]["text_hidden_size"], "assembled": base.hidden_size},
            "text_projection": "Official ResizeMLP, resized to text width and newly initialized; not Identity",
            "position_encoding": "Preserve Talker MRoPE; identical axis positions reduce to ordinary 1D positions",
        },
        "audio_vocabulary": {"first_head_and_embedding": config.talker_config.vocab_size,
                             "codebook_size": 2048, "codebooks": 16},
        "speaker_encoder": {"architecture": "ECAPA-TDNN", "dimension": config.speaker_encoder_config.enc_dim,
                            "sample_rate": config.speaker_encoder_config.sample_rate,
                            "source": "speaker_encoder.* from the TTS checkpoint; external pretraining provenance not assumed"},
        "loaded_modules": ["talker.model.layers", "talker.model.norm", "talker.model.text_embedding (shared rows)",
                           "speaker_encoder", "speech_tokenizer encoder/decoder"],
        "new_modules": ["talker.text_projection", "talker.model.codec_embedding", "talker.codec_head",
                        "talker.code_predictor", "new TTS text-token rows"],
        "not_loaded": ["text LM head", "pretrained TTS Talker and code predictor"],
    }
    return config, target_tokenizer, report


def load_prefix(directory, prefix):
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
        filenames = sorted({name for key, name in index.items() if key.startswith(prefix)})
    else:
        filenames = [p.name for p in sorted(directory.glob("model*.safetensors"))]
    values = {}
    for name in filenames:
        with safe_open(directory / name, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    short = key[len(prefix):]
                    if short in values:
                        raise ValueError(f"Duplicate checkpoint tensor: {key}")
                    values[short] = f.get_tensor(key)
    if not values:
        raise ValueError(f"No {prefix} weights found in {directory}")
    return values


def initialize_model(config, backbone, speaker_source, added_tokens, dtype=torch.bfloat16, seed=42):
    torch.manual_seed(seed)
    # Build in FP32 so newly initialized weights use ordinary initialization,
    # then store the assembled model in the requested dtype.
    model = Qwen3TTSForConditionalGeneration(config).to(dtype=dtype)
    base = AutoModel.from_pretrained(backbone, dtype=dtype, attn_implementation="sdpa")
    model.talker.model.layers.load_state_dict(base.layers.state_dict(), strict=True)
    model.talker.model.norm.load_state_dict(base.norm.state_dict(), strict=True)
    with torch.no_grad():
        target = model.talker.model.text_embedding.weight
        source = base.embed_tokens.weight
        target[:source.shape[0]].copy_(source)
        if added_tokens:
            ids = torch.tensor(sorted(added_tokens.values()), dtype=torch.long)
            fresh = torch.empty(len(ids), target.shape[1], dtype=torch.float32)
            fresh.normal_(mean=0, std=config.talker_config.initializer_range)
            target[ids] = fresh.to(dtype)
    del base
    gc.collect()
    speaker = load_prefix(speaker_source, "speaker_encoder.")
    model.speaker_encoder.load_state_dict(speaker, strict=True)
    return model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_codec(source, destination):
    source, destination = Path(source), Path(destination)
    if not (source / "config.json").exists() or not list(source.glob("*.safetensors")):
        raise ValueError("Codec directory needs config.json and safetensors weights")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".cache", ".git", ".ipynb_checkpoints"))


def save_model(model, directory):
    model.save_pretrained(directory, safe_serialization=True, max_shard_size="2GB")
    path = Path(directory) / "config.json"
    config = json.loads(path.read_text())
    # qwen-tts 0.1.1 serializes this field but its speaker config constructor
    # rejects it. Remove only the unsupported metadata before public reload.
    config["speaker_encoder_config"].pop("model_type", None)
    path.write_text(json.dumps(config, indent=2) + "\n")


def validate_saved(directory, dtype=torch.bfloat16):
    from qwen_tts import Qwen3TTSModel
    # Use the public wrapper, including its processor and codec loading.
    loaded = Qwen3TTSModel.from_pretrained(str(directory), device_map="cpu", dtype=dtype,
                                         attn_implementation="sdpa")
    if loaded.model.talker.text_projection.__class__.__name__ != "Qwen3TTSTalkerResizeMLP":
        raise ValueError("Projection architecture changed on reload")
    for field, token in TEXT_SPECIALS.items():
        if loaded.processor.tokenizer.convert_tokens_to_ids(token) != getattr(loaded.model.config, field):
            raise ValueError(f"Special token changed on reload: {token}")
    result = {"public_wrapper_reload": True,
              "speaker_encoder": loaded.model.speaker_encoder.__class__.__name__,
              "codec_sample_rate": loaded.model.speech_tokenizer.get_output_sample_rate(),
              "text_width": loaded.model.talker.model.text_embedding.embedding_dim}
    del loaded
    gc.collect()
    return result
