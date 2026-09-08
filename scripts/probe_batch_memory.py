"""Run real FSDP optimizer steps on the longest manifest rows."""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import torch
import torch.distributed as dist
from qwen3_train.data import CodeDataset, collate
from qwen3_train.model import TTSModel
from qwen3_train.train import configure_fsdp, move


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assembled', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--batch-size', type=int, required=True)
    args = p.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dist.init_process_group('nccl', device_id=device)
    try:
        data = CodeDataset(args.manifest)
        cfg = json.loads((Path(args.assembled) / 'config.json').read_text())
        for row in data.rows:
            row['text_ids'] = [cfg['tts_bos_token_id'], *row['text_ids'], cfg['tts_eos_token_id']]
        data.set_speaker_references(data, 3)
        # Bound padding cost using both the maximum text and maximum audio length.
        longest_audio = max(range(len(data)), key=lambda i: data.rows[i]['num_frames'])
        longest_text = max(range(len(data)), key=lambda i: len(data.rows[i]['text_ids']))
        audio_row, text_row = data[longest_audio], data[longest_text]
        rows = [{**audio_row, 'text_ids': text_row['text_ids']}] * args.batch_size
        batch = move(collate(rows), device)
        model = TTSModel.from_assembled(args.assembled)
        configure_fsdp(model, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            result = model(batch)
            loss = result['first_sum'] / result['first_count'] + .3 * result['residual_sum'] / (15 * result['frame_count'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            torch.cuda.synchronize()
            print(json.dumps({'rank': rank, 'step': step + 1, 'batch_size': args.batch_size,
                'text_tokens': len(text_row['text_ids']), 'audio_frames': len(audio_row['codes']),
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                'reserved_gib': torch.cuda.memory_reserved() / 2**30}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
