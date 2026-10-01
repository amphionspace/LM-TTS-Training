import pytest

from qwen3_train.training.schedule import finished, learning_rate_factor


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
