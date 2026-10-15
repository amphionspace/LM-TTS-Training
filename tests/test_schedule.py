import pytest

from lm_tts.training.schedule import finished, learning_rate_factor


def test_wsd_boundaries_and_epoch_stop():
    settings = {
        "epochs": 3,
        "max_steps": None,
        "warmup_steps": 1000,
        "scheduler": {"name": "wsd", "decay_ratio": 0.1, "min_lr_ratio": 0.1},
    }
    assert learning_rate_factor(settings, 0, 0) == pytest.approx(0.001)
    assert learning_rate_factor(settings, 999, 0.01) == 1
    assert learning_rate_factor(settings, 1000, 1.5) == 1
    assert learning_rate_factor(settings, 2000, 2.7) == 1
    assert learning_rate_factor(settings, 3000, 2.85) == pytest.approx(0.55)
    assert learning_rate_factor(settings, 4000, 3) == pytest.approx(0.1)
    assert not finished(settings, {"step": 999999, "epoch": 2})
    assert finished(settings, {"step": 999999, "epoch": 3})
    assert finished({**settings, "max_steps": 10}, {"step": 10, "epoch": 0})


def test_wsd_fraction_warmup_is_independent_of_update_count():
    settings = {
        "epochs": 3,
        "warmup_steps": None,
        "scheduler": {"name": "wsd", "warmup_ratio": 0.01, "decay_ratio": 0.1, "min_lr_ratio": 0.1},
    }
    assert learning_rate_factor(settings, 0, 0) == 0
    assert learning_rate_factor(settings, 500, 0.015) == pytest.approx(0.5)
    assert learning_rate_factor(settings, 1000, 0.015) == pytest.approx(0.5)
    assert learning_rate_factor(settings, 2000, 0.03) == 1
    assert learning_rate_factor(settings, 3000, 2.85) == pytest.approx(0.55)


def test_anneal_matches_the_wsd_decay_phase_and_stops_after_it():
    wsd = {
        "epochs": 3,
        "warmup_steps": None,
        "scheduler": {"name": "wsd", "warmup_ratio": 0.01, "decay_ratio": 0.1, "min_lr_ratio": 0.1},
    }
    anneal = {
        "epochs": 1,
        "max_steps": None,
        "warmup_steps": None,
        "scheduler": {"name": "anneal", "decay_epochs": 0.3, "min_lr_ratio": 0.1},
    }
    for offset in (0, 0.05, 0.15, 0.299, 0.3):
        assert learning_rate_factor(anneal, 0, offset) == pytest.approx(
            learning_rate_factor(wsd, 0, 2.7 + offset)
        )
    assert learning_rate_factor(anneal, 0, 0) == 1
    assert learning_rate_factor(anneal, 0, 0.5) == pytest.approx(0.1)
    assert not finished(anneal, {"step": 10**6, "epoch": 0}, 0.2999)
    assert finished(anneal, {"step": 1, "epoch": 0}, 0.3)
    assert finished({**anneal, "max_steps": 5}, {"step": 5, "epoch": 0}, 0.1)
