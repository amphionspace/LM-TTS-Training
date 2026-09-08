"""Synthetic codes for reproducible trainer checks; not a speech-quality test."""
import json
from pathlib import Path
import numpy as np

root = Path("data/synthetic-smoke")
(root / "codes").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(123)
rows = []
for i in range(11):
    frames = 2 + i % 4
    path = f"codes/{i}.npz"
    np.savez(root / path, codes=rng.integers(0, 2048, (frames, 16), dtype=np.uint16))
    rows.append(dict(id=f"synthetic-{i}", text="synthetic test", text_ids=[3, 7, 11][:1 + i % 3],
                     codes=path, num_frames=frames, duration=frames / 12.5))
for name, data in [("train", rows[:8]), ("val", rows[8:])]:
    (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
print(root)
