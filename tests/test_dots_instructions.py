import json

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch

from lm_tts.artifacts import digest, file_hash
from lm_tts.data.dots import (
    REFERENCE_MIX,
    DotsCollator,
    DotsFeatureDataset,
    pairable,
)
from lm_tts.data.flexivoice import (
    DEFAULT_INSTRUCTION,
    STYLE_ID,
    conflict_matrix,
    instruction_prefix,
    speaker_key,
    style_class,
)
from lm_tts.data.split import split_build


class FakeLayout:
    ids = {"span": 9, "end": 8, "start": 7, "text_end": 6}
    prompts = {"text": [1], "audio": [2], "interleave": [3]}
    eos = 0
    overhead = 4

    def sequence(self, text, patches, interleave, prefix=()):
        from lm_tts.data.dots import Layout

        return Layout.sequence(self, text, patches, interleave, prefix)


def test_prefix_precedes_body_in_both_layouts():
    layout = FakeLayout()
    assert layout.sequence([20, 21], 2, False, [30, 31]) == [1, 30, 31, 20, 21, 2, 7, 9, 9, 8, 0]
    assert layout.sequence([20, 21], 2, False) == [1, 20, 21, 2, 7, 9, 9, 8, 0]
    assert layout.sequence([20], 2, True, [30]) == [3, 30, 20, 9, 6, 9, 8, 0]
    assert layout.sequence([20, 21, 22], 1, True, [30]) == [3, 30, 20, 9, 21, 8, 22, 6, 0]


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("text", [[20, 21], [20, 21, 22, 23, 24]])
def test_collator_counts_spans_and_predictors_with_a_prefix(interleave, text):
    row = {
        "text_ids": text,
        "prefix": [30, 31, 32],
        "interleave": interleave,
        "vae_mean": torch.zeros(8, 128),
        "vae_logstd": torch.zeros(8, 128),
        "reference": (0, 0),
        "xvector": torch.zeros(512),
        "duration": 1.0,
    }
    batch = DotsCollator(FakeLayout())([row])
    assert len(batch["span_positions"]) == len(batch["predictor_positions"]) == 2
    expected = [3, 30, 31, 32, 20, 9] if interleave else [1, 30, 31, 32, 20, 21]
    assert batch["input_ids"][:6].tolist() == expected
    assert batch["reference_intervals"].tolist() == [[0, 0]]


def test_instruction_prefix_defaults_when_missing():
    assert instruction_prefix(None) == f"Instruction: {DEFAULT_INSTRUCTION}\nText:\n"
    assert instruction_prefix("  ") == instruction_prefix(None)
    assert instruction_prefix("用开心的语气") == "Instruction: 用开心的语气\nText:\n"


@pytest.mark.parametrize(
    ("config", "path", "key"),
    [
        ("genshin", "audio/instruction_data/genshin/ZH/派蒙/vo_x.wav", "genshin:ZH:派蒙"),
        ("ears", "audio/instruction_data/ears/p043/sentences_11_highpitch.wav", "ears:p043"),
        (
            "expresso",
            "audio/instruction_data/expresso/audio_48khz/conversational_vad_segmented/ex03-ex01/"
            "calm/ex03-ex01_calm_002_channel2_segment_1_2.wav",
            "expresso:ex01",
        ),
        (
            "expresso",
            "audio/instruction_data/expresso/audio_48khz/read/ex02/sad/base/a.wav",
            "expresso:ex02",
        ),
        (
            "emotion",
            "audio/instruction_data/IEMOCAP/Session3/sentences/wav/Ses03M_impro05b/Ses03M_impro05b_F029.wav",
            "IEMOCAP:Session3-F",
        ),
        ("emotion", "audio/instruction_data/MELD/train_wav/dia6_utt7.wav", None),
        ("emilia_psc", "audio/emilia/EN/EN_B00019_S00750_W000023.mp3", None),
    ],
)
def test_speaker_keys_come_from_the_path(config, path, key):
    assert speaker_key(config, path) == key


def test_style_classes_come_from_paths_and_labels():
    ears = "audio/instruction_data/ears/p043/"
    assert style_class("ears", ears + "rainbow_03_whisper.wav", None) == STYLE_ID["whisper"]
    assert style_class("ears", ears + "sentences_regular.wav", None) == STYLE_ID["neutral"]
    assert style_class("ears", ears + "emo_anger_sentences.wav", None) == STYLE_ID["angry"]
    assert style_class("ears", ears + "emo_confusion_sentences.wav", None) == STYLE_ID["other"]
    assert style_class("emotion", "x/ESD/0013/Angry/0013_1.wav", ["生气"]) == STYLE_ID["angry"]
    assert style_class("genshin", "audio/instruction_data/genshin/ZH/派蒙/a.wav", None) == -1
    conflicts = np.asarray(conflict_matrix())
    assert conflicts[STYLE_ID["happy"], STYLE_ID["sad"]]
    assert not conflicts[STYLE_ID["happy"], STYLE_ID["happy"]]


def test_pairable_needs_another_long_utterance_with_other_text():
    groups = np.array([0, 0, 1, 1, 2, 3, 3])
    texts = np.array([1, 2, 5, 5, 7, 8, 9], dtype=np.uint64)
    long = np.array([True, True, True, True, True, False, True])
    # group 1 repeats one text; group 2 is alone; in group 3 only row 6 is long enough
    assert pairable(groups, texts, long).tolist() == [True, True, False, False, False, True, False]


def fake_dataset(styles, texts, long, groups):
    ds = DotsFeatureDataset.__new__(DotsFeatureDataset)
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    ds.references = {
        0: {
            "group": groups,
            "order": order,
            "starts": np.searchsorted(groups[order], np.arange(groups.max() + 2)),
            "text": np.asarray(texts, dtype=np.uint64),
            "long": np.asarray(long),
            "style": np.asarray(styles, dtype=np.int8),
        }
    }
    ds.neutral = STYLE_ID["neutral"]
    ds.conflicts = np.asarray(conflict_matrix())
    ds.rng = np.random.default_rng(0)
    return ds


def test_reference_draw_mixes_neutral_conflicting_and_any_and_never_returns_itself():
    happy, sad, neutral, fast = (STYLE_ID[s] for s in ("happy", "sad", "neutral", "fast"))
    # row 0 is the happy target; 1 neutral, 2 sad (conflict), 3 fast, 4 same text as the target
    ds = fake_dataset([happy, neutral, sad, fast, sad], [1, 2, 3, 4, 1], [True] * 5, [0] * 5)
    draws = np.array([ds.draw_reference(0, 0) for _ in range(20000)])
    assert set(draws) == {1, 2, 3}
    shares = np.bincount(draws, minlength=4)[1:] / len(draws)
    any_share = REFERENCE_MIX[2] / 3
    np.testing.assert_allclose(
        shares, [REFERENCE_MIX[0] + any_share, REFERENCE_MIX[1] + any_share, any_share], atol=0.02
    )


def test_unlabelled_targets_draw_any_reference_of_their_speaker():
    ds = fake_dataset([-1, -1, -1, -1], [1, 2, 3, 4], [True, True, False, True], [0, 0, 0, 1])
    draws = {ds.draw_reference(0, 0) for _ in range(2000)}
    assert draws == {1}  # row 2 is too short, row 3 is another speaker


def test_metadata_comes_only_from_the_pinned_selection_branch(tmp_path):
    from lm_tts.data.dots import flexivoice_rows

    release = tmp_path / "datasets/flexivoice/v0.1"
    release.mkdir(parents=True)
    table = lance.write_dataset(
        pa.table(
            {
                "sample_id": ["kept", "excluded"],
                "source_config": ["ears", "ears"],
                "sample_rate": [48000, 48000],
                "metadata_json": [
                    json.dumps(
                        {
                            "instruction": "happy",
                            "upstream": {
                                "wav_path": f"audio/instruction_data/ears/p00{i}/sentences_regular.wav"
                            },
                        }
                    )
                    for i in (1, 2)
                ],
            }
        ),
        release / "samples.lance",
    )
    (release / "manifest.json").write_text("{}")
    branch = table.create_branch("selected", table.version)
    branch.add_columns(pa.table({"selection_reason": pa.array([0, 7001], pa.uint16())}))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "outputs": [
                    {
                        "dataset_id": "flexivoice",
                        "branch": "selected",
                        "lance_version": branch.version,
                        "selected_rows": 1,
                    }
                ],
            }
        )
    )
    manifest = {
        "rows": 1,
        "inputs": [
            {
                "artifact_kind": "base",
                "dataset_id": "flexivoice",
                "release_id": "v0.1",
                "table_path": "samples.lance",
                "base_version": table.version,
                "branch": "selected",
                "lance_version": branch.version,
                "manifest_sha256": file_hash(release / "manifest.json"),
            }
        ],
        "selection": {
            "mode": "selection_branch",
            "selection_id": "selected",
            "manifest_path": "selection.json",
            "manifest_sha256": file_hash(selection_path),
        },
    }
    result = flexivoice_rows(tmp_path, manifest)
    assert set(result) == {"kept"}
    assert result["kept"][1] == "ears:p001"
    manifest["inputs"][0]["lance_version"] += 1
    with pytest.raises(ValueError, match="differs"):
        flexivoice_rows(tmp_path, manifest)


def test_build_and_split_use_selected_text_and_isolate_speakers(tmp_path, monkeypatch):
    from lm_tts.data import dots

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, texts, **kwargs):
            def encode(text):
                return [ord(c) for c in text]

            return {
                "input_ids": encode(texts) if isinstance(texts, str) else [encode(t) for t in texts]
            }

        def get_added_vocab(self):
            return {"[Laughter]": 1000, "[Breathing]": 1001}

        def convert_tokens_to_ids(self, token):
            return list(dots.SPECIAL_TOKENS.values()).index(token) + 1

    # Selected instruction variants share a closed pool of different-text long recordings.
    records, metadata = [], {}
    rule = "dots-reference-v2:min-3s-on-patches"
    for speaker in range(4):
        for variant, patches in enumerate((25, 25, 30)):
            target = f"{speaker}-{variant}"
            start, end = dots.reference_interval(target, patches, rule)
            records.append(
                {
                    "target_id": target,
                    "status": "ok",
                    "text": "short[Laughter][Breathing]" if variant < 2 else "long",
                    "text_kind": "raw",
                    "language": "en",
                    "num_latent_frames": patches * 4,
                    "processed_frames": patches * 7680,
                    "reference_start_patch": start,
                    "reference_end_patch": end,
                    "vae_mean": np.zeros((patches * 4, 128), dtype=np.float32).tolist(),
                    "vae_logstd": np.zeros((patches * 4, 128), dtype=np.float32).tolist(),
                    "xvector": [float(speaker + 1)] * 512,
                }
            )
            metadata[target] = ("ears", f"ears:{speaker}", -1, f"instruction {variant}", 48000)
    table = pa.Table.from_pylist(records)
    for field in ("vae_mean", "vae_logstd"):
        table = table.set_column(
            table.schema.get_field_index(field),
            field,
            pa.array([r[field] for r in records], type=pa.list_(pa.list_(pa.float32(), 128))),
        )
    table = table.set_column(
        table.schema.get_field_index("xvector"),
        "xvector",
        pa.array([r["xvector"] for r in records], type=pa.list_(pa.float32(), 512)),
    )
    source = lance.write_dataset(table, tmp_path / "features.lance")
    profile = {"reference_rule": rule}
    manifest = {
        "dataset_id": "flexivoice",
        "release_id": "v0.1",
        "rows": len(records),
        "profile": profile,
        "profile_id": digest(profile),
    }
    source_path = tmp_path / "feature-manifest.json"
    source_path.write_text(json.dumps(manifest))
    reference = {
        "table_path": str(tmp_path / "features.lance"),
        "lance_version": source.version,
        "manifest_path": str(source_path),
        "manifest_sha256": file_hash(source_path),
    }
    monkeypatch.setattr(dots, "feature_reference", lambda *args: (manifest, reference, source))
    monkeypatch.setattr(dots, "flexivoice_rows", lambda *args: metadata)
    recipe = {
        "root": str(tmp_path),
        "evaluation": "no_holdout",
        "instructions": "flexivoice-v1",
        "bindings": [
            {
                "dataset_id": "flexivoice",
                "dots_manifest": "unused",
                "metadata": "flexivoice",
                "reference": "cross_speaker",
            }
        ],
    }
    tokenizer = Tokenizer()
    output = tmp_path / "build"
    dots.bind_dots(recipe, output, {}, tokenizer)
    ds = DotsFeatureDataset(output / "manifest.json", dots.Layout(tokenizer))
    assert len(ds) == 12
    for i in range(len(ds)):
        row = ds[i]
        assert row["reference"] == (0, 0)
        assert row["text_ids"] == [
            ord(c) for c in ("short[Laughter][Breathing]" if i % 3 < 2 else "long")
        ]
        assert row["xvector"][0] == i // 3 + 1
    split_build(ds.path, fraction=0.25, seed=42)
    speakers, pools = {}, {}
    for role in ("train", "validation"):
        split = DotsFeatureDataset(output / role / "manifest.json", dots.Layout(tokenizer))
        speakers[role] = {int(split[i]["id"].split("-")[0]) for i in range(len(split))}
        pools[role] = set(split.references[0]["rows"].tolist())
        assert len(split) == len(speakers[role]) * 3
        for i in range(len(split)):
            row = split[i]
            assert row["xvector"][0] == int(row["id"].split("-")[0]) + 1
    assert speakers["train"].isdisjoint(speakers["validation"])
    assert pools["train"].isdisjoint(pools["validation"])
    assert pools["train"] | pools["validation"] == set(range(12))
    from transformers import AutoTokenizer

    from scripts.acp.after_features import validate_build

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    report = validate_build(
        {
            "model": {"assembled_model": str(tmp_path)},
            "data": {
                "build": str(output / "train/manifest.json"),
                "val_build": str(output / "validation/manifest.json"),
            },
            "train": {"max_batch_frames": 600, "max_batch_tokens": 1000},
        }
    )
    assert report["train"][0]["rows"] == 9
    assert report["validation"][0]["rows"] == 3
