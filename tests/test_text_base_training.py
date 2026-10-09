import json
import shutil
from pathlib import Path

import pytest
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig

from lm_tts.artifacts import file_hash
from lm_tts.data.batch import collate
from lm_tts.data.build import tokenizer_identity, verify_tokenizer_compatibility
from lm_tts.models.qwen import TTSModel, make_config
from lm_tts.objectives.tts import tts_loss
from lm_tts.training.config import load_config
from lm_tts.training.optimizer import parameter_groups
from scripts.prepare_models import assembly_recipe, verify_assembly


def test_text_frontend_gradients_and_opt_in_optimizer_groups():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = make_config(tiny=True)
    config.lm_tts_text_projection = "mlp"
    config.lm_tts_freeze_text_frontend = False
    config.lm_tts_freeze_speaker_encoder = True
    config.lm_tts_input_protocol = "qwen3_non_streaming"
    config.lm_tts_pad_token_id = 20
    config.lm_tts_role_ids = [21, 22, 23]
    config.codec_nothink_id = 2151
    config.codec_think_bos_id = 2152
    config.codec_think_eos_id = 2153
    speaker = Qwen3TTSSpeakerEncoderConfig(enc_dim=64, mel_dim=8, enc_channels=[16, 16, 16, 16, 48])
    model = TTSModel(config, speaker).train()
    settings = {"lr": 3e-4, "backbone_lr": 1e-4}
    legacy = [[], []]
    for name, param in model.named_parameters():
        if param.requires_grad:
            index = 0 if name.startswith(("talker.model.layers.", "talker.model.norm.")) else 1
            legacy[index].append(id(param))
    assert [[id(p) for p in g["params"]] for g in parameter_groups(model, settings)] == legacy
    groups = parameter_groups(model, {**settings, "text_embedding_lr_group": "backbone"})
    low, high = [{id(p) for p in group["params"]} for group in groups]
    embedding = model.talker.model.text_embedding.weight
    projector = model.talker.text_projection
    assert id(embedding) in low
    assert all(id(p) in high for p in projector.parameters())
    assert not low & high
    assert low | high == {id(p) for p in model.parameters() if p.requires_grad}
    batch = collate(
        [
            {
                "text_ids": [18, 3, 7, 19],
                "codes": torch.randint(0, 2048, (3, 16)),
                "speaker_embedding": torch.randn(64),
            }
        ]
    )
    result = tts_loss(model(batch), batch, model.eos)
    loss = result["first_sum"] / 4 + 0.3 * result["residual_sum"] / 45
    loss.backward()
    for index in [3, 7, 18, 19, 20, 21, 22, 23]:
        assert torch.isfinite(embedding.grad[index]).all()
        assert embedding.grad[index].abs().sum() > 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in projector.parameters()
    )
    assert all(p.grad is None and not p.requires_grad for p in model.speaker_encoder.parameters())
    assert not model.speaker_encoder.training
    before = embedding.detach().clone()
    torch.optim.AdamW(groups).step()
    assert not torch.equal(before, embedding)


def test_assembly_recipe_and_existing_artifact_identity(tmp_path):
    recipe = assembly_recipe(
        {
            "assembly": {
                "text_initialization": "text-base",
                "text_projection_init": "random",
                "train_text_frontend": True,
            }
        }
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights").write_bytes(b"weights")
    report = {
        **{
            key: recipe[key]
            for key in ("seed", "dtype", "text_initialization", "text_projection_init")
        },
        "freeze_text_frontend": False,
        "freeze_speaker_encoder": True,
        "adaptations": {"input_protocol": "qwen3_non_streaming"},
        "sources": {
            "backbone": {
                "resolved_directory": str(source),
                "config_sha256": file_hash(source / "config.json"),
            }
        },
        "artifact_sha256": {"weights": file_hash(artifact / "weights")},
    }
    (artifact / "assembly_report.json").write_text(json.dumps(report))
    verify_assembly(artifact, recipe, {"backbone": source})
    with pytest.raises(ValueError, match="recipe mismatch"):
        verify_assembly(artifact, assembly_recipe({}), {"backbone": source})
    (artifact / "weights").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_assembly(artifact, recipe, {"backbone": source})
    with pytest.raises(ValueError, match="Unknown assembly"):
        assembly_recipe({"assembly": {"typo": True}})


def test_keep_all_is_accepted_without_changing_legacy_defaults(tmp_path):
    base = Path(__file__).resolve().parents[1] / "configs/train-bf16.yaml"
    assert load_config(base)["train"]["keep_checkpoints"] == 2
    config = tmp_path / "experiment.yaml"
    config.write_text(f"extends: {base}\ntrain:\n  keep_checkpoints: null\n")
    assert load_config(config)["train"]["keep_checkpoints"] is None
    config.write_text(f"extends: {base}\ntrain:\n  text_embedding_lr_group: typo\n")
    with pytest.raises(ValueError, match="text_embedding_lr_group"):
        load_config(config)


def test_tokenizer_compatibility_preserves_build_identity_and_protocol(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    (source / "tokenizer.json").write_text('{"tokens": ["a", "b"]}')
    config = {
        "model_type": "qwen3_tts",
        "tts_bos_token_id": 18,
        "talker_config": {
            "text_hidden_size": 2048,
            "text_vocab_size": 256,
            "lm_tts_freeze_text_frontend": True,
            "lm_tts_role_ids": [1, 2, 3],
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    identity = tokenizer_identity(source)
    target = tmp_path / "new"
    shutil.copytree(source, target)
    config["talker_config"].update(text_hidden_size=1024, lm_tts_freeze_text_frontend=False)
    (target / "config.json").write_text(json.dumps(config))
    verify_tokenizer_compatibility(target, identity)
    assert tokenizer_identity(source) == identity
    config["tts_bos_token_id"] = 19
    (target / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="protocol differs"):
        verify_tokenizer_compatibility(target, identity)
    (target / "tokenizer.json").write_text('{"tokens": ["b", "a"]}')
    with pytest.raises(ValueError, match="files differ"):
        verify_tokenizer_compatibility(target, identity)
    (source / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Build tokenizer artifact changed"):
        verify_tokenizer_compatibility(target, identity)


def test_prepare_rejects_family_mismatch_before_copying(tmp_path, monkeypatch):
    import sys

    import yaml

    from scripts import prepare_models

    model = tmp_path / "assembled"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "lfm2_tts"}')
    config = tmp_path / "wrong.yaml"
    config.write_text(
        yaml.safe_dump(
            {"paths": {"project": str(tmp_path)}, "model": {"assembled_model": str(model)}}
        )
    )
    monkeypatch.setattr(sys, "argv", ["prepare_models", "--config", str(config)])
    with pytest.raises(ValueError, match="assembly.family disagrees"):
        prepare_models.main()
    assert not (tmp_path / "assets").exists()
    with pytest.raises(ValueError, match="only accepts"):
        assembly_recipe({"assembly": {"family": "lfm2", "text_initialization": "qwen-tts"}})
