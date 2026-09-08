"""Dataset adapters emit raw records; training only reads prepared manifests."""
import json
from pathlib import Path
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf


def short_record(meta, archive, member, offset, size):
    # The carrier type is authoritative. Nested short views are never selected.
    if meta.get('type') != 'short':
        return None
    entries = meta['short']
    if len(entries) != 1:
        raise ValueError(f"Expected one full-file short: {meta['id']}")
    item = entries[0]
    if item['rel_start_samples'] != 0 or item['rel_end_samples'] != meta['frames']:
        raise ValueError(f"Short does not cover its carrier: {meta['id']}")
    if not item.get('speaker') or not item.get('text', '').strip():
        raise ValueError(f"Missing speaker or text: {meta['id']}")
    return {'schema_version': 1, 'id': 'emilia2:' + meta['id'],
            'text': item['text'].strip(), 'speaker': 'emilia2:' + item['speaker'],
            'language': item['language'], 'duration': meta['frames'] / meta['sample_rate'],
            'audio': {'kind': 'tar_member', 'archive': str(archive), 'member': member,
                      'offset': offset, 'size': size, 'frames': meta['frames'],
                      'sample_rate': meta['sample_rate']},
            'source': {'dataset': 'emilia2', 'type': 'short', 'recording_id': meta['recording_id'],
                       'dnsmos': item.get('dnsmos')}}


def _emilia_shard(index):
    records = []
    members = {}
    for line in index.read_text().splitlines():
        name, offset, size = line.split('\t')
        if name in members:
            raise ValueError(f'Duplicate tar member: {name}')
        members[name] = (int(offset), int(size))
    archive = index.with_suffix('')
    with archive.open('rb') as audio_file:
        for name, (offset, size) in members.items():
            if not name.endswith('.json'):
                continue
            audio_file.seek(offset)
            meta = json.loads(audio_file.read(size))
            if meta.get('type') != 'short':
                continue
            audio_name = name[:-5] + '.m4a'
            audio_offset, audio_size = members[audio_name]
            records.append(short_record(meta, archive, audio_name, audio_offset, audio_size))
    return records


def emilia_short(root, max_shards=0, workers=8):
    indices = sorted((Path(root).resolve() / 'data').glob('*.tar.idx'))
    if not indices:
        raise ValueError('No Emilia tar indices found')
    indices = indices[:max_shards or None]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Bounded prefetch; deterministic shard order regardless of I/O completion.
        for start in range(0, len(indices), workers):
            for records in executor.map(_emilia_shard, indices[start:start + workers]):
                yield from records


def ljspeech(root):
    root = Path(root).resolve()
    for line in (root / 'metadata.csv').read_text().splitlines():
        uid, raw, normalized = line.split('|', 2)
        path = root / 'wavs' / f'{uid}.wav'
        info = sf.info(path)
        yield {'schema_version': 1, 'id': uid, 'text': normalized.strip() or raw.strip(),
               'speaker': 'ljspeech', 'language': 'en', 'duration': info.duration,
               'audio': {'kind': 'file', 'path': str(path)},
               'source': {'dataset': 'ljspeech', 'type': 'utterance'}}


def materialize_audio(record, destination):
    locator = record['audio']
    destination = Path(destination)
    if locator['kind'] == 'file':
        return Path(locator['path']).resolve()
    if locator['kind'] != 'tar_member':
        raise ValueError(f"Unknown audio locator: {locator['kind']}")
    # PyAV uses FFmpeg's MP4 demuxer with edit-list handling. BytesIO is seekable.
    import av
    with Path(locator['archive']).open('rb') as archive:
        archive.seek(locator['offset'])
        payload = archive.read(locator['size'])
        if len(payload) != locator['size']:
            raise ValueError('Truncated tar member')
    chunks = []
    with av.open(io.BytesIO(payload)) as container:
        resampler = av.AudioResampler(format='flt', layout='mono', rate=locator['sample_rate'])
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    samples = np.concatenate(chunks)
    if not locator['frames'] <= len(samples) <= locator['frames'] + 1023:
        raise ValueError(f"Decoded length differs from annotation: {len(samples)} / {locator['frames']}")
    samples = samples[:locator['frames']]
    from scipy.signal import resample_poly
    from math import gcd
    divisor = gcd(locator['sample_rate'], 24000)
    samples = resample_poly(samples, 24000 // divisor, locator['sample_rate'] // divisor)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, samples, 24000, subtype='FLOAT')
    return destination.resolve()


def evaluation_audio(row):
    path = Path(row['audio'])
    if not path.exists():
        materialize_audio({**row, 'audio': row['audio_source']}, path)
    return path
