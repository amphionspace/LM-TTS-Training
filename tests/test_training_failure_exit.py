import os
import subprocess
import sys
from pathlib import Path


def test_rank_failure_exits_with_traceback_without_collective_teardown(tmp_path):
    assembly = tmp_path / "assembly"
    assembly.mkdir()
    (assembly / "config.json").write_text('{"model_type":"dots_tts"}')
    (assembly / "ASSEMBLY_COMPLETE").touch()
    marker = tmp_path / "teardown-called"
    code = f"""
from pathlib import Path
import torch
from lm_tts.training import engine
engine.training_precision = lambda *a: (None, "sdpa")
engine.initialize = lambda seed: "cpu"
engine.dist.get_rank = lambda: 0
engine.dist.get_world_size = lambda: 2
engine.dist.destroy_process_group = lambda: Path({str(marker)!r}).touch()
def fail(*a):
    raise torch.OutOfMemoryError("injected rank-local OOM")
engine.task_for = lambda family: fail
engine.run({{"seed":42,"train":{{}},"model":{{"assembled_model":{str(assembly)!r}}}}})
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 1
    assert "injected rank-local OOM" in result.stderr
    assert "Traceback" in result.stderr
    assert not marker.exists()
