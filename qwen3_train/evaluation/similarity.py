"""Speaker verification using a fixed WavLM x-vector checkpoint."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.build import file_hash


def model_identity(directory):
    directory = Path(directory).resolve()
    files = {
        p.name: file_hash(p)
        for p in directory.iterdir()
        if p.is_file() and (p.suffix in (".json", ".bin", ".safetensors") or p.name == "REVISION")
    }
    if not any(name.endswith((".bin", ".safetensors")) for name in files):
        raise ValueError(f"Missing local evaluation model weights: {directory}")
    return {"path": str(directory), "files": files}


class WavLMSimilarity:
    def __init__(self, directory, device="cpu", chunk_seconds=10):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self.identity = {
            **model_identity(directory),
            "sample_rate": 16000,
            "chunk_seconds": chunk_seconds,
            "aggregation": "mean_normalized_xvectors",
            "dtype": "float32",
        }
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(directory, local_files_only=True)
        self.model = (
            WavLMForXVector.from_pretrained(directory, local_files_only=True)
            .float()
            .to(device)
            .eval()
        )
        self.device = device
        self.chunk_frames = int(chunk_seconds * 16000)
        if self.chunk_frames < 16000:
            raise ValueError("WavLM chunks must be at least one second")

    @torch.inference_mode()
    def embedding(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or len(audio) < 16000 or not np.isfinite(audio).all():
            raise ValueError("WavLM requires at least 1s finite mono audio at 16kHz")
        # Even chunks avoid a tiny trailing segment; every input frame contributes.
        chunks = np.array_split(audio, int(np.ceil(len(audio) / self.chunk_frames)))
        embeddings = []
        for chunk in chunks:
            inputs = self.extractor(chunk, sampling_rate=16000, return_tensors="pt")
            value = self.model(
                **{k: v.to(self.device) for k, v in inputs.items()}
            ).embeddings.float()
            embeddings.append(F.normalize(value, dim=-1)[0])
        vector = torch.stack(embeddings).mean(0)
        if not torch.isfinite(vector).all() or vector.norm() == 0:
            raise ValueError("Invalid WavLM embedding")
        return F.normalize(vector, dim=0).cpu()

    def score(self, generated, reference):
        return float(torch.dot(self.embedding(generated), self.embedding(reference)).clamp(-1, 1))
