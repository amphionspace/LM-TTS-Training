"""Reference features for the official ECAPA speaker encoder."""
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


def audio_mel(row):
    locator = row.get('audio_source')
    if locator and locator['kind'] == 'tar_member':
        from .sources import decode_emilia_audio
        waveform = torch.from_numpy(decode_emilia_audio({**row, 'audio': locator}))
    else:
        audio, sr = sf.read(row['audio'], dtype='float32', always_2d=True)
        waveform = torch.from_numpy(np.ascontiguousarray(audio.mean(axis=1)))
        if sr != 24000:
            waveform = AF.resample(waveform, sr, 24000)
    if not len(waveform) or not torch.isfinite(waveform).all():
        raise ValueError(f"Empty or non-finite speaker audio: {row['id']}")
    return mel_spectrogram(waveform.unsqueeze(0), n_fft=1024, num_mels=128,
                           sampling_rate=24000, hop_size=256, win_size=1024,
                           fmin=0, fmax=12000)[0].T.contiguous()


def _reflect_input(module, args):
    value = args[0]
    left, right = module._reversed_padding_repeated_twice
    if left or right:
        size = value.shape[-1]
        indices = torch.arange(-left, size + right, device=value.device)
        indices = torch.where(indices < 0, -indices, indices)
        indices = torch.where(indices >= size, 2 * size - 2 - indices, indices)
        value = value.index_select(-1, indices)
    return (value,)


def deterministic_speaker_padding(encoder):
    # CUDA reflection_pad1d backward uses atomic additions. Index selection has
    # a deterministic backward and preserves the official reflection samples.
    for module in encoder.modules():
        if isinstance(module, torch.nn.Conv1d) and module.padding_mode == "reflect":
            module.register_forward_pre_hook(_reflect_input)
            module.padding_mode = "zeros"
            module.padding = (0,)
