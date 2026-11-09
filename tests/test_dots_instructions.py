import numpy as np
import pytest
import torch

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


class FakeLayout:
    ids = {"span": 9, "end": 8, "start": 7, "text_end": 6}
    prompts = {"text": [1], "audio": [2], "interleave": [3]}
    eos = 0
    overhead = 4

    def sequence(self, text, patches, interleave, prefix=()):
        from lm_tts.data.dots import Layout

        return Layout.sequence(self, text, patches, interleave, prefix)


def test_prefix_sits_between_the_text_prompt_and_the_text_and_never_interleaves():
    layout = FakeLayout()
    assert layout.sequence([20, 21], 2, False, [30, 31]) == [1, 30, 31, 20, 21, 2, 7, 9, 9, 8, 0]
    assert layout.sequence([20, 21], 2, False) == [1, 20, 21, 2, 7, 9, 9, 8, 0]
    with pytest.raises(ValueError):
        layout.sequence([20], 2, True, [30])


def test_collator_counts_spans_and_predictors_with_a_prefix():
    row = {
        "text_ids": [20, 21],
        "prefix": [30, 31, 32],
        "interleave": False,
        "vae_mean": torch.zeros(8, 128),
        "vae_logstd": torch.zeros(8, 128),
        "reference": (0, 0),
        "xvector": torch.zeros(512),
        "duration": 1.0,
    }
    batch = DotsCollator(FakeLayout())([row])
    assert len(batch["span_positions"]) == len(batch["predictor_positions"]) == 2
    assert batch["input_ids"][:6].tolist() == [1, 30, 31, 32, 20, 21]
    assert batch["reference_intervals"].tolist() == [[0, 0]]


def test_instruction_prefix_defaults_when_missing():
    assert instruction_prefix(None) == f"Instruction: {DEFAULT_INSTRUCTION}\nText: "
    assert instruction_prefix("  ") == instruction_prefix(None)
    assert instruction_prefix("用开心的语气") == "Instruction: 用开心的语气\nText: "


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
