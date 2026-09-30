"""DNSMOS P.835, non-personalized calibration from Microsoft's DNS Challenge."""

from pathlib import Path

import numpy as np

from ..artifacts import file_hash


class DNSMOS:
    def __init__(self, path, threads=4):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.identity = {
            "path": str(Path(path).resolve()),
            "sha256": file_hash(path),
            "method": "P835_non_personalized_9.01s_1s_hop",
            "sample_rate": 16000,
            "short_audio": "repeat_by_doubling",
        }

    def score(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("DNSMOS requires nonempty finite mono audio at 16kHz")
        window, hop = 144160, 16000
        while len(audio) < window:
            audio = np.concatenate([audio, audio])
        raw = []
        for start in range(0, len(audio) - window + 1, hop):
            raw.append(
                np.asarray(
                    self.session.run(None, {"input_1": audio[None, start : start + window]})[0]
                ).reshape(-1)
            )
        values = np.asarray(raw)
        if values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("Invalid DNSMOS model output")
        coefficients = [
            [-0.08397278, 1.22083953, 0.0052439],
            [-0.13166888, 1.60915514, -0.39604546],
            [-0.06766283, 1.11546468, 0.04602535],
        ]
        return {
            name: float(np.polyval(poly, values[:, i]).mean())
            for i, (name, poly) in enumerate(
                zip(("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"), coefficients, strict=True)
            )
        }
