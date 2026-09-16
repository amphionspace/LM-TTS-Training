"""Validate Seed-TTS coverage and report the six model/conditioning comparisons."""
import argparse
import json
import math
from pathlib import Path
import statistics

import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifests = Path('/119010446/UltraEval-Audio/raw_data/voice_clone_manifests')
    canonical = {}
    for language in ('en', 'zh'):
        for line in (manifests / f'seed_tts_eval_{language}.jsonl').read_text().splitlines():
            row = json.loads(line)
            canonical[(row['dataset'], row['index'])] = row
    table = []
    outliers = []
    for model in ('token', 'sqrt', 'sample'):
        for mode in ('xvec_only', 'icl_xvec'):
            name = f'{model}-{mode}'
            directory = args.run_dir / 'results' / name
            generated, scores = {}, {}
            for path in sorted(directory.glob('inference-*.jsonl')):
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row['status'] != 'ok':
                        continue
                    key = (row['dataset'], row['index'])
                    assert all(row[field] == value for field, value in canonical[key].items()), (name, key)
                    assert row['model'] == name and row['speaker_embedding'] and row['icl'] == (mode == 'icl_xvec')
                    audio = sf.info(row['audio'])
                    assert audio.frames > 0 and audio.samplerate == row['sample_rate'] == 24000
                    assert abs(audio.duration - row['duration']) < 1 / 24000
                    generated[key] = row
            for path in sorted(directory.glob('scores-*.jsonl')):
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row['status'] == 'ok':
                        assert row['model'] == name
                        scores[(row['dataset'], row['index'])] = row['score']
            for language in ('en', 'zh'):
                dataset = f'seed_tts_eval_{language}'
                expected = {key for key in canonical if key[0] == dataset}
                inf = {key: value for key, value in generated.items() if key[0] == dataset}
                scored = {key: value for key, value in scores.items() if key[0] == dataset}
                assert set(scored) <= set(inf) <= expected
                metric = 'wer%' if language == 'en' else 'cer%'
                for score in scored.values():
                    assert all(math.isfinite(float(score[m])) for m in (metric, 'simo'))
                row = {'model': model, 'mode': mode, 'dataset': dataset, 'expected': len(expected),
                       'generated': len(inf), 'scored': len(scored), 'complete': set(inf) == set(scored) == expected,
                       'error_percent': statistics.mean(v[metric] for v in scored.values()) if scored else None,
                       'similarity_percent': 100 * statistics.mean(v['simo'] for v in scored.values()) if scored else None,
                       'no_eos': sum(not v['eos_reached'] for v in inf.values()),
                       'over_30_seconds': sum(v['duration'] > 30 for v in inf.values())}
                table.append(row)
                for key in sorted(scored, key=lambda k: scored[k][metric], reverse=True)[:10]:
                    outliers.append({'model': name, **inf[key], 'score': scored[key]})
    (args.run_dir / 'summary.json').write_text(json.dumps(table, indent=2))
    (args.run_dir / 'outliers.json').write_text(json.dumps(outliers, ensure_ascii=False, indent=2))
    lines = ['# 19000-step 模型 Seed-TTS 评测', '',
             '使用 UltraEval-Audio 的 Seed-TTS 英文 1,088 条、中文 2,020 条；token、sqrt、sample 三个模型均取 step 19000。', '',
             '`xvec_only` 只使用参考音频的说话人向量；`icl_xvec` 同时使用说话人向量、参考文本和参考音频 token。', '',
             '推理采用 Qwen3-TTS 0.1.1 原生 API、BF16、Flash Attention 2、non-streaming、Auto 语言协议。'
             '所有组共享采样配置：temperature=0.9、top_k=50、top_p=1、repetition_penalty=1.05；'
             'Code Predictor 同样以 temperature=0.9、top_k=50、top_p=1 采样；最多 2048 个新 token，最少 2 个。'
             '固定分片和批次种子，批次信息保存在 worker JSON 和逐条记录中。', '',
             '评分直接复用 UltraEval 注册的 Seed 评测器：英文 Whisper-large-v3 WER、中文 Paraformer CER，'
             '说话人相似度使用 WavLM。错误率按逐句百分比取算术平均，SIM 将原始相似度均值乘 100。'
             '这与训练中使用的小样本 ASR 检查不属于同一评测。', '',
             '| 模型 | 条件 | 语言 | 生成/总数 | 评分/总数 | WER/CER % ↓ | SIM % ↑ | 未到 EOS |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for row in table:
        error = '—' if row['error_percent'] is None else f'{row["error_percent"]:.3f}'
        similarity = '—' if row['similarity_percent'] is None else f'{row["similarity_percent"]:.3f}'
        lines.append(f'| {row["model"]} | {row["mode"]} | {row["dataset"][-2:]} | '
                     f'{row["generated"]}/{row["expected"]} | {row["scored"]}/{row["expected"]} | '
                     f'{error} | {similarity} | {row["no_eos"]} |')
    complete = all(row['complete'] for row in table)
    lines.extend(['', '**状态：' + ('全量完成。' if complete else '尚未完成；部分结果不能当作全量结论。') + '**', '',
                  f'运行目录：`{args.run_dir.resolve()}`。其中 `pipeline_status.json` 记录进程和阶段，'
                  '`logs/` 保存日志，`results/` 保存音频和逐条评分，`exports/` 保存 checkpoint 导出及校验信息。', '',
                  '详细设置和复现方式见 [评测说明](seed-tts-eval-19000.md)。'])
    report = root / 'docs/training/seed-tts-eval-19000-results.md'
    report.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if not complete and not args.allow_incomplete:
        raise SystemExit('Evaluation coverage is incomplete')


if __name__ == '__main__':
    main()
