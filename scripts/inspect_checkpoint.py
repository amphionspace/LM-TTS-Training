"""Inspect teacher-forced codec accuracy and sensitivity to text order."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
import yaml
from qwen3_train.data import CodeDataset, collate
from qwen3_train.model import TTSModel
from qwen3_train.speaker import reference_mel

@torch.no_grad()
def generate(model, batch, max_frames):
    prefix = {**batch, 'codes': batch['codes'][:, :0], 'frame_mask': batch['frame_mask'][:, :0]}
    stopped = False
    for _ in range(max_frames):
        frame, stop = model(prefix, mode='next_frame')
        if stop.item():
            stopped = True
            break
        prefix['codes'] = torch.cat([prefix['codes'], frame[:, None]], dim=1)
        prefix['frame_mask'] = torch.ones(prefix['codes'].shape[:2], dtype=torch.bool, device=frame.device)
    return prefix['codes'][0].cpu().numpy(), stopped


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    for key in ('config', 'checkpoint', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--generate', action='store_true')
    p.add_argument('--max-frames', type=int, default=96)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = yaml.safe_load(Path(args.config).read_text())
    assembled = Path(config['model']['assembled_model'])
    full = json.loads((assembled / 'config.json').read_text())
    checkpoint = Path(args.checkpoint)
    if not (checkpoint / 'COMPLETE').exists():
        raise ValueError('Require a completed checkpoint')
    model = TTSModel.from_assembled(assembled, load_weights=False)
    dcp.load({'model': model.state_dict()}, checkpoint_id=checkpoint / 'distributed')
    model.to(device='cuda').eval()
    # FSDP casts parameters, not RoPE buffers. Preserve FP32 inverse frequencies.
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    datasets = {s: CodeDataset(config['data'][s]) for s in ('train', 'val')}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for split, dataset in datasets.items():
        dataset.set_speaker_references(datasets['train'], config['data']['speaker_reference_seconds'])
        row = dataset[0]
        row['text_ids'] = [full['tts_bos_token_id'], *row['text_ids'], full['tts_eos_token_id']]
        batch = {k: v.cuda() for k, v in collate([row]).items()}
        batch['speaker_mels'] = batch['speaker_mels'].bfloat16()
        h = model.hidden(batch)[:, :-1].flatten(0, 1)
        target = batch['codes'][0]
        depth, _ = model.talker.forward_sub_talker_finetune(target, h)
        predicted = torch.cat([model.talker.codec_head(h)[:, :model.code_size].argmax(-1, keepdim=True), depth.argmax(-1)], dim=1)
        residual_only = predicted.clone()
        residual_only[:, 0] = target[:, 0]
        normal = model(batch)
        shuffled = {**batch, 'text_ids': batch['text_ids'].clone()}
        shuffled['text_ids'][:, 1:-1] = shuffled['text_ids'][:, 1:-1].flip(-1)
        altered = model(shuffled)
        other_audio = next(r['audio'] for r in datasets['train'].rows
                           if r['audio'] not in (row['audio'], row['reference_audio']))
        other_mel = reference_mel(other_audio, config['data']['speaker_reference_seconds'])
        swapped = {**batch, 'speaker_mels': other_mel.unsqueeze(0).to(device='cuda', dtype=torch.bfloat16)}
        swapped_loss = model(swapped)
        result = {'split': split, 'id': row['id'], 'text': row['text'], 'audio': row['audio'], 'speaker_reference': row['reference_audio'], 'swapped_reference': other_audio,
                  'code_accuracy': (predicted == target).float().mean(0).cpu().tolist(),
                  'first_ce': (normal['first_sum'] / normal['first_count']).item(),
                  'reversed_text_first_ce': (altered['first_sum'] / altered['first_count']).item(),
                  'swapped_reference_first_ce': (swapped_loss['first_sum'] / swapped_loss['first_count']).item()}
        code_arrays = dict(target=target.cpu().numpy(), teacher_forced=predicted.cpu().numpy(), residual_only=residual_only.cpu().numpy())
        result['first_frame_target'] = target[0, 0].item()
        result['first_frame_predictions'] = {}
        other_row = next(r for r in datasets['train'].rows if r['audio'] == other_audio)
        other_ids = torch.tensor([[full['tts_bos_token_id'], *other_row['text_ids'], full['tts_eos_token_id']]], device='cuda')
        other_text = {**batch, 'text_ids': other_ids, 'text_mask': torch.ones_like(other_ids, dtype=torch.bool)}
        result['generation_texts'] = {'generated_other_text': other_row['text']}
        conditions = [('original', batch), ('reversed_text', shuffled), ('swapped_reference', swapped), ('other_text', other_text)]
        for name, condition in conditions:
            state = model.hidden(condition)[:, 0]
            result['first_frame_predictions'][name] = model.talker.codec_head(state)[0, :model.code_size].argmax().item()
            if args.generate:
                code_arrays['generated_' + name], stopped = generate(model, condition, args.max_frames)
                result[name + '_generation'] = {'eos_reached': stopped, 'frames': len(code_arrays['generated_' + name])}
        if args.generate:
            with np.load(datasets['train'].path.parent / other_row['codes']) as expected:
                result['other_text_exact_codes'] = np.array_equal(code_arrays['generated_other_text'], expected['codes'])
        np.savez(output / f'{split}-codes.npz', **code_arrays)
        results.append(result)
        print(json.dumps(result), flush=True)
    (output / 'summary.json').write_text(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
