"""Select the model implementation from the immutable assembled artifact."""

import json
from pathlib import Path


def load_model(directory, **kwargs):
    config = json.loads((Path(directory) / "config.json").read_text())
    if config.get("model_type") == "lfm2_tts":
        from .lfm import LfmTTSModel

        return LfmTTSModel.from_assembled(directory, **kwargs)
    from .qwen import TTSModel

    return TTSModel.from_assembled(directory, **kwargs)
