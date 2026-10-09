"""Offline, local-only initialization of a pure-codec LFM TTS artifact."""

import json
import shutil
import uuid
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast

from ...artifacts import file_hash
from ..lfm import LfmTTSModel
from ..lfm.configuration import LfmTTSConfig
from .common import TEXT_SPECIALS, copy_codec, load_prefix


def prepare_config(backbone, template, text_frontend="native"):
    if text_frontend not in {"native", "qwen-mlp"}:
        raise ValueError("text_frontend must be native or qwen-mlp")
    backbone, template = Path(backbone), Path(template)
    base = AutoConfig.from_pretrained(backbone, local_files_only=True)
    raw = json.loads((template / "config.json").read_text())
    if base.model_type != "lfm2" or base.hidden_size != raw["talker_config"]["hidden_size"]:
        raise ValueError("Require an LFM2 backbone matching the codec predictor input width")
    if raw["tokenizer_type"] != "qwen3_tts_tokenizer_12hz":
        raise ValueError("Require the Qwen 12Hz codec template")
    predictor = raw["talker_config"]["code_predictor_config"]
    if (predictor["num_hidden_layers"], predictor["vocab_size"], predictor["num_code_groups"]) != (
        5,
        2048,
        16,
    ):
        raise ValueError("Require the 5-layer, 16-codebook predictor template")
    # Transformers 4.57 supports the model but not the newer TokenizersBackend
    # class name. Preserve the actual tokenizer pipeline and export a portable class.
    tc = json.loads((backbone / "tokenizer_config.json").read_text())
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(backbone / "tokenizer.json"),
        **{k: tc[k] for k in ("bos_token", "eos_token", "pad_token")},
        clean_up_tokenization_spaces=tc.get("clean_up_tokenization_spaces", False),
    )
    source_vocab = tokenizer.get_vocab()
    tokenizer.add_special_tokens({"additional_special_tokens": list(TEXT_SPECIALS.values())})
    added = {s: i for s, i in tokenizer.get_vocab().items() if s not in source_vocab}
    rows = max(base.vocab_size, len(tokenizer))
    base.vocab_size = rows
    # The Qwen config supplies codec protocol and predictor fields only. The nested
    # LFM config is authoritative for the hybrid backbone's heads, FFN and norms.
    talker = raw["talker_config"]
    talker.update(
        text_hidden_size=base.hidden_size * (2 if text_frontend == "qwen-mlp" else 1),
        text_vocab_size=rows,
        lm_tts_lfm_config=base.to_dict(),
        lm_tts_text_projection="mlp" if text_frontend == "qwen-mlp" else "identity",
        lm_tts_input_protocol="qwen3_non_streaming",
        lm_tts_use_speaker_embedding=False,
        lm_tts_freeze_text_frontend=False,
        lm_tts_role_ids=tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False),
        lm_tts_pad_token_id=tokenizer.convert_tokens_to_ids("<tts_pad>"),
    )
    config = LfmTTSConfig.from_dict(talker)
    config._attn_implementation = "sdpa"
    config.code_predictor_config._attn_implementation = "sdpa"
    artifact = {
        "model_type": "lfm2_tts",
        "architectures": ["LfmTTSModel"],
        "tokenizer_type": raw["tokenizer_type"],
        "talker_config": config.to_dict(),
        **{key: tokenizer.convert_tokens_to_ids(token) for key, token in TEXT_SPECIALS.items()},
    }
    return config, artifact, tokenizer, added, len(source_vocab)


def assemble(
    backbone, template, codec, output, seed=42, dtype=torch.float32, *, text_frontend="native"
):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    config, artifact, tokenizer, added, source_tokens = prepare_config(
        backbone, template, text_frontend
    )
    torch.manual_seed(seed)
    model = LfmTTSModel(config)
    source = load_prefix(backbone, "model.")
    original_embedding = source.pop("embed_tokens.weight")
    incompatible = model.talker.model.load_state_dict(source, strict=False)
    if set(incompatible.missing_keys) != {"embed_tokens.weight", "codec_embedding.weight"}:
        raise ValueError(f"Incomplete backbone initialization: {incompatible}")
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected backbone weights: {incompatible.unexpected_keys}")
    with torch.no_grad():
        target = model.talker.model.embed_tokens.weight
        native = target[:, : config.hidden_size]
        native[: len(original_embedding)].copy_(original_embedding)
        fresh_ids = torch.tensor(sorted(added.values()), dtype=torch.long)
        fresh = torch.empty(len(fresh_ids), config.hidden_size).normal_(
            std=config.initializer_range
        )
        native[fresh_ids] = fresh
        if text_frontend == "qwen-mlp":
            target[:, config.hidden_size :].copy_(-native)
    model.to(dtype=dtype)
    # Check every reused tensor, not just a few representative layers.
    reused = model.talker.model.state_dict()
    for key, value in source.items():
        if not torch.equal(reused[key], value.to(dtype)):
            raise ValueError(f"Pretrained tensor changed: {key}")
    shared = torch.ones(len(original_embedding), dtype=torch.bool)
    shared[fresh_ids[fresh_ids < len(shared)]] = False
    target = model.talker.model.embed_tokens.weight
    if not torch.equal(
        target[: len(shared), : config.hidden_size][shared], original_embedding[shared].to(dtype)
    ):
        raise ValueError("Pretrained text embedding rows changed")
    if text_frontend == "qwen-mlp":
        if not torch.equal(target[:, config.hidden_size :], -target[:, : config.hidden_size]):
            raise ValueError("Paired text embedding initialization changed")
    # Probe the actual computation in bounded batches; all reused rows are checked
    # exactly above, including the negative half for the wider frontend.
    probe_ids = torch.linspace(0, len(target) - 1, min(len(target), 256)).long()
    with torch.no_grad():
        projected = model.talker.text_projection(target[probe_ids])
        expected = target[probe_ids, : config.hidden_size]
        projection_error = (projected.float() - expected.float()).abs().max().item()
        torch.testing.assert_close(
            projected,
            expected,
            atol=1e-6 if dtype == torch.float32 else 0.01,
            rtol=1e-5 if dtype == torch.float32 else 0.01,
        )
    del source, original_embedding, reused
    report = {
        "format_version": 1,
        "model_type": "lfm2_tts",
        "seed": seed,
        "dtype": str(dtype).removeprefix("torch."),
        "text_initialization": "lfm-base",
        "text_frontend": text_frontend,
        "text_projection_init": "paired-silu" if text_frontend == "qwen-mlp" else "identity",
        "freeze_text_frontend": False,
        "use_speaker_embedding": False,
        "sources": {
            name: {
                "resolved_directory": str(Path(path).resolve()),
                "sha256": {
                    f.name: file_hash(f)
                    for f in sorted(Path(path).glob("*"))
                    if f.is_file() and f.suffix in {".json", ".safetensors"}
                },
            }
            for name, path in (("backbone", backbone), ("tts_template", template), ("codec", codec))
        },
        "text_vocabulary": {
            "source_token_count": source_tokens,
            "target_embedding_rows": config.text_vocab_size,
            "added_tokens": added,
            "shared_rows_verified": int(shared.sum()),
        },
        "loaded_modules": ["LFM layers", "LFM final norm", "LFM text embedding (shared rows)"],
        "new_modules": ["TTS token rows", "codec embedding/head", "5-layer Code Predictor"],
        "text_frontend_verification": {
            "probe_rows": len(probe_ids),
            "max_projection_error": projection_error,
            "text_hidden_size": config.text_hidden_size,
        },
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "code_predictor": sum(p.numel() for p in model.talker.code_predictor.parameters()),
        },
    }
    if text_frontend == "qwen-mlp":
        report["new_modules"].append(
            f"{config.text_hidden_size}-wide text frontend (paired embedding and SiLU MLP)"
        )
    temporary = output.with_name(output.name + ".incomplete-" + uuid.uuid4().hex[:8])
    temporary.mkdir(parents=True)
    try:
        (temporary / "config.json").write_text(json.dumps(artifact, indent=2) + "\n")
        tokenizer.save_pretrained(temporary)
        save_file(
            model.state_dict(), str(temporary / "model.safetensors"), metadata={"format": "pt"}
        )
        copy_codec(codec, temporary / "speech_tokenizer")
        report["artifact_sha256"] = {
            str(p.relative_to(temporary)): file_hash(p)
            for p in sorted(temporary.rglob("*"))
            if p.is_file()
        }
        (temporary / "assembly_report.json").write_text(json.dumps(report, indent=2) + "\n")
        (temporary / "ASSEMBLY_COMPLETE").write_text("lfm2_tts\n")
        reloaded = LfmTTSModel.from_assembled(temporary, attn_implementation="sdpa").to(dtype)
        for name, value in model.state_dict().items():
            if not torch.equal(value, reloaded.state_dict()[name]):
                raise ValueError(f"Reload changed tensor: {name}")
        loaded_tokenizer = AutoTokenizer.from_pretrained(temporary, local_files_only=True)
        if loaded_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("Reload changed tokenizer vocabulary")
        report["verification"] = {"all_reused_weights_equal": True, "exact_reload": True}
        (temporary / "assembly_report.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return report
