"""Freeze versioned geometry; never write to reconstruction or manual-edit sources."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def freeze_sources(source_root: Path, identity_csv: Path, output: Path) -> dict:
    source_root, output = Path(source_root), Path(output)
    entries = read_json(source_root / 'final_delivery_manifest.json')['entries']
    with Path(identity_csv).open(encoding='utf-8-sig', newline='') as stream:
        identities = list(csv.DictReader(stream))
    mapping = {str(Path(row['source_dir']).resolve()).casefold(): row for row in identities}
    hashes, samples, by_size = {}, [], defaultdict(list)
    video_paths = sorted({str(Path(v).resolve()) for entry in entries for v in entry['source_videos']})
    for path in video_paths:
        by_size[Path(path).stat().st_size].append(path)
    video_hashes = {}
    for paths in by_size.values():
        if len(paths) > 1:
            for path in paths:
                video_hashes[path] = digest(path)
    hashes[str(Path(identity_csv))] = digest(identity_csv)
    hashes[str(source_root / 'final_delivery_manifest.json')] = digest(source_root / 'final_delivery_manifest.json')
    for entry in entries:
        key = entry['key']
        state_path = source_root / 'manual_edits' / key / 'state.json'
        selection_path = source_root / key / 'selection.json'
        selection = read_json(selection_path) if selection_path.exists() else {}
        model_path = Path(selection.get('model_path', entry['selected_model']))
        source = state_path if state_path.exists() else model_path
        source_hash = digest(source)
        data = read_json(source)
        if digest(source) != source_hash:
            raise RuntimeError(f'Input changed during snapshot: {source}')
        hashes[str(source)] = source_hash
        models = []
        for item in data.get('models', data.get('berries', [])):
            models.append({k: item[k] for k in ('id', 'center', 'radius', 'evidence', 'manual',
                          'all_berries_verified', 'individually_verified_around_full_orbit') if k in item})
            models[-1]['shape'] = 'sphere'
            if item.get('shape') == 'ellipsoid' or 'axes' in item:
                models[-1]['evidence'] = 'sphere_conversion_approximate'
        videos = [str(Path(path).resolve()) for path in entry['source_videos']]
        matches = [mapping.get(str(Path(v).parent).casefold()) for v in videos]
        if not matches or any(row is None for row in matches):
            raise ValueError(f'Missing identity mapping for {key}')
        if len({row['bunch_id'] for row in matches}) != 1:
            raise ValueError(f'Conflicting source identities for {key}')
        row = matches[0]
        source_keys = [f'path:{p.casefold()}' for p in videos]
        source_keys += [f'sha256:{video_hashes[p]}' for p in videos if p in video_hashes]
        geometry_hash = hashlib.sha256(json.dumps(
            [{'center': m['center'], 'radius': m['radius']} for m in models],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        samples.append(dict(sample_id=key, bunch_id=row['bunch_id'], group=row['identity_group'],
                            date=row['source_relpath'].split('/')[0], stage=row['stage'],
                            period=entry['period'], revision=data.get('revision', 0), models=models,
                            model_source=str(source), model_sha256=source_hash, geometry_sha256=geometry_hash,
                            source_videos=videos, source_video_keys=source_keys,
                            measured_cloud=entry['measured_cloud'], excluded=False,
                            all_berries_verified=False, coordinate_units='arbitrary COLMAP units'))
    by_video = defaultdict(list)
    for sample in samples:
        for key in sample['source_video_keys']:
            by_video[key].append(sample)
    conflicts = set()
    for key, rows in by_video.items():
        if len({row['bunch_id'] for row in rows}) > 1:
            conflicts.update(row['sample_id'] for row in rows)
    for sample in samples:
        if sample['sample_id'] in conflicts:
            sample.update(excluded=True, exclusion_reason='identical video with conflicting identity labels')
    result = dict(schema_version=1, frozen_at=datetime.now(timezone.utc).isoformat(),
                  samples=samples, exclusions=sorted(conflicts), source_hashes=hashes,
                  video_hashes=video_hashes, limits=['model-estimated geometry, not per-berry ground truth',
                  'independent COLMAP scales; no physical growth measurements'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'samples.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--identity-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = freeze_sources(args.source_root, args.identity_csv, args.output)
    print(json.dumps({'samples': len(result['samples']), 'excluded': result['exclusions'],
                      'identities': len({s['bunch_id'] for s in result['samples'] if not s['excluded']})}))
