"""Keep clean, non-overlapping dialogue utterances using their actual audio."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly

from qwen3_train.sources import decode_emilia_audio
from scripts.prepare_emilia_streaming import publish_json


MODEL = Path('/119010446/tts-assets/DNSMOS/DNSMOS/sig_bak_ovr.onnx')
THRESHOLDS = {'metadata_dnsmos': 3.4, 'OVRL': 3.4, 'SIG': 3.5, 'BAK': 4.0,
              'maximum_clipped_fraction': 0.001}


class AudioQuality:
    def __init__(self, model):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])

    def score(self, audio):
        # DNSMOS P.835 calibration, with integer sample windows so floating-point
        # rounding cannot shorten a window and silently discard the clip's tail.
        while len(audio) < 144160:
            audio = np.concatenate((audio, audio))
        predictions = []
        for index in range((len(audio) - 144160) // 16000 + 1):
            segment = audio[index * 16000:index * 16000 + 144160].astype(np.float32)
            sig, bak, overall = self.session.run(None, {'input_1': segment[None]})[0][0]
            predictions.append((np.polyval([-0.08397278, 1.22083953, 0.0052439], sig),
                                np.polyval([-0.13166888, 1.60915514, -0.39604546], bak),
                                np.polyval([-0.06766283, 1.11546468, 0.04602535], overall)))
        return dict(zip(('SIG', 'BAK', 'OVRL'), map(float, np.mean(predictions, axis=0))))


def overlaps_other_speaker(row, shorts):
    start, end = row['audio']['start_frame'], row['audio']['end_frame']
    speaker = row['speaker'].removeprefix('emilia2:')
    return any(item.get('speaker') != speaker and
               max(start, item['rel_start_samples']) < min(end, item['rel_end_samples'])
               for item in shorts)


def accepted(scores, clipped_fraction):
    return (all(math.isfinite(scores[key]) and scores[key] >= THRESHOLDS[key] for key in ('OVRL', 'SIG', 'BAK'))
            and clipped_fraction <= THRESHOLDS['maximum_clipped_fraction'])


def initialize(model):
    global quality
    quality = AudioQuality(model)


def process_shard(task):
    source, root = map(Path, task)
    result = root / 'stats' / (source.stem + '.json')
    if result.exists():
        return json.loads(result.read_text())
    counts, hours = Counter(), Counter()
    target = root / 'kept' / source.name
    score_path = root / 'scores' / source.name
    index = None
    carrier_id, shorts = None, None
    with source.open() as stream, target.with_suffix('.incomplete').open('w') as output, \
            score_path.with_suffix('.incomplete').open('w') as scores_output:
        for line in stream:
            row = json.loads(line)
            language = row['language']
            counts[f'{language}/input'] += 1
            metadata_score = row['source'].get('dnsmos')
            if not isinstance(metadata_score, (int, float)) or not math.isfinite(metadata_score) or metadata_score < THRESHOLDS['metadata_dnsmos']:
                counts[f'{language}/metadata_rejected'] += 1
                continue
            locator = row['audio']
            if index is None:
                index = {}
                for entry in Path(locator['archive'] + '.idx').read_text().splitlines():
                    name, offset, size = entry.split('\t')
                    if name.endswith('.json'):
                        index[name] = (int(offset), int(size))
            if carrier_id != locator['member']:
                offset, size = index[locator['member'][:-4] + '.json']
                with Path(locator['archive']).open('rb') as archive:
                    archive.seek(offset)
                    meta = json.loads(archive.read(size))
                assert meta['type'] == 'dialogue' and meta['id'] == row['source']['carrier_id']
                carrier_id, shorts = locator['member'], meta['short']
            if overlaps_other_speaker(row, shorts):
                counts[f'{language}/overlapping_speaker_rejected'] += 1
                continue
            audio = decode_emilia_audio(row)
            if not len(audio) or not np.isfinite(audio).all() or not np.any(audio):
                counts[f'{language}/invalid_audio_rejected'] += 1
                continue
            score = quality.score(resample_poly(audio, 2, 3))
            clipped = float(np.mean(np.abs(audio) >= 0.999))
            keep = accepted(score, clipped)
            record = {'id': row['id'], 'language': language, 'audio_source': locator,
                      'metadata_dnsmos': metadata_score, **score, 'clipped_fraction': clipped, 'accepted': keep}
            scores_output.write(json.dumps(record) + '\n')
            counts[f'{language}/scored'] += 1
            if not keep:
                counts[f'{language}/audio_quality_rejected'] += 1
                continue
            row['source']['quality'] = {'method': 'dnsmos_p835_actual_slice', **score,
                                        'clipped_fraction': clipped, 'overlapping_speaker': False}
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[f'{language}/kept'] += 1
            hours[language] += row['duration'] / 3600
    target.with_suffix('.incomplete').replace(target)
    score_path.with_suffix('.incomplete').replace(score_path)
    report = {'source': str(source), 'counts': dict(counts), 'hours': dict(hours)}
    publish_json(result, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    data = args.output.resolve()
    root = data / 'dialogue-quality'
    for name in ('kept', 'scores', 'stats'):
        (root / name).mkdir(parents=True, exist_ok=True)
    inventory = data / 'INVENTORY_COMPLETE.json'
    recipe = {'thresholds': THRESHOLDS, 'model': str(MODEL),
              'model_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(),
              'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'overlap': 'reject_any_other_annotated_speaker', 'audio': 'native_sample_slice_then_24khz',
              'scorer': 'DNSMOS P.835 non-personalized, repeated short input, all integer 144160-sample windows / 16000-sample hop',
              'onnxruntime': ort.__version__}
    recipe_path = root / 'recipe.json'
    if recipe_path.exists():
        assert json.loads(recipe_path.read_text()) == recipe, 'Dialogue quality recipe changed'
    publish_json(recipe_path, recipe)
    if (data / 'DIALOGUE_QUALITY_COMPLETE.json').exists():
        return
    paths = sorted((data / 'candidates/dialogue').glob('emilia*.jsonl'))
    counts, hours = Counter(), Counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(MODEL,),
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(process_shard, (str(path), str(root))) for path in paths]
        for index, future in enumerate(as_completed(futures), 1):
            report = future.result()
            counts.update(report['counts'])
            hours.update(report['hours'])
            if index % 25 == 0 or index == len(paths):
                status = {'completed_shards': index, 'total_shards': len(paths),
                          'counts': dict(counts), 'hours': dict(hours)}
                publish_json(root / 'status.json', status)
                print(json.dumps(status), flush=True)
    assert all(counts[f'{language}/kept'] > 0 for language in ('en', 'zh'))
    publish_json(data / 'DIALOGUE_QUALITY_COMPLETE.json', {**status, 'recipe': recipe})


if __name__ == '__main__':
    main()
