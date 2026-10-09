"""A common waveform convention for all acoustic metrics."""

import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path, sample_rate=16000, max_seconds=180):
    info = sf.info(path)
    if info.frames == 0 or info.duration > max_seconds:
        raise ValueError(f"Empty or oversized evaluation audio: {path}")
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite evaluation audio: {path}")
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, rate // divisor)
    return np.asarray(audio, dtype=np.float32)
