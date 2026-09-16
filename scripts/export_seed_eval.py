"""Export the three completed training checkpoints for native Qwen inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

from qwen3_train.model import TTSModel
from qwen3_train.assembly import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    assembled = root / 'pretrained/assembled-qwen3-tts-frozen-conditioning'
    for name in ('token', 'sqrt', 'sample'):
        suffix = '' if name == 'token' else '-' + name
        checkpoint = root / f'runs/emilia-en-zh-dynamic-10000h{suffix}/checkpoints/step-00019000'
        assert (checkpoint / 'COMPLETE').is_file(), checkpoint
        target = args.output / name
        if (target / 'export.json').exists():
            print(f'Already exported: {target}', flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        model = TTSModel.from_assembled(assembled, load_weights=False)
        state = model.state_dict()
        dcp.load({'model': state}, checkpoint_id=checkpoint / 'distributed')
        state = {key: value.to(torch.bfloat16).contiguous() for key, value in state.items()}
        save_file(state, str(target / 'model.safetensors'), metadata={'format': 'pt'})
        for source in assembled.iterdir():
            if source.name == 'speech_tokenizer':
                (target / source.name).symlink_to(source.resolve(), target_is_directory=True)
            elif source.is_file() and source.suffix in ('.json', '.txt') and source.name != 'assembly_report.json':
                shutil.copy2(source, target / source.name)
        info = {'checkpoint': str(checkpoint), 'step': 19000, 'tensors': len(state),
                'parameters': sum(value.numel() for value in state.values()),
                'checkpoint_metadata_sha256': hashlib.sha256((checkpoint / 'metadata.json').read_bytes()).hexdigest(),
                'weights_sha256': sha256(target / 'model.safetensors')}
        (target / 'export.json').write_text(json.dumps(info, indent=2))
        print(json.dumps({'model': name, **info}), flush=True)
        del model, state


if __name__ == '__main__':
    main()
