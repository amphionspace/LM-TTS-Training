"""Check trained DCP text frontend against the exact assembled initialization."""
import argparse
import json
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from qwen3_train.assembly import load_prefix


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assembled', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--include-speaker', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    expected = {}
    prefixes = ['talker.model.text_embedding.', 'talker.text_projection.']
    if args.include_speaker:
        prefixes.append('speaker_encoder.')
    for prefix in prefixes:
        expected.update({prefix + key: tensor.float() for key, tensor in load_prefix(args.assembled, prefix).items()})
    state = {'model': {key: tensor.clone() for key, tensor in expected.items()}}
    dcp.load(state, checkpoint_id=str(Path(args.checkpoint) / 'distributed'))
    for key, tensor in expected.items():
        torch.testing.assert_close(state['model'][key], tensor, atol=0, rtol=0, msg=key)
    print(json.dumps({'status': 'passed', 'exact_frozen_tensors': len(expected), 'include_speaker': args.include_speaker,
                      'parameters': sum(t.numel() for t in expected.values())}))


if __name__ == '__main__':
    main()
