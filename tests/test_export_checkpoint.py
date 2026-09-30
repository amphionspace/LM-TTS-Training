import json
import sys

import pytest

from scripts.export_checkpoint import main


def test_export_rejects_a_different_assembly_before_loading_weights(tmp_path, monkeypatch):
    checkpoint, assembled = tmp_path / "checkpoint", tmp_path / "assembled"
    checkpoint.mkdir()
    assembled.mkdir()
    (checkpoint / "COMPLETE").touch()
    (checkpoint / "metadata.json").write_text(
        json.dumps({"signature": {"assembly_sha256": "another-assembly"}})
    )
    (assembled / "assembly_report.json").write_text("{}")
    output = tmp_path / "export"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--assembled-model",
            str(assembled),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="different assembly identities"):
        main()
    assert not output.exists()
