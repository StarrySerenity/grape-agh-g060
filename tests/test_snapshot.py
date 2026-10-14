import hashlib
import json


def test_manual_snapshot_preserves_input_and_marks_conflicting_videos(tmp_path):
    from snapshot import freeze_sources

    root = tmp_path / 'source'
    root.mkdir()
    video_a = root / 'a' / 'one.mp4'
    video_b = root / 'b' / 'two.mp4'
    for video in (video_a, video_b):
        video.parent.mkdir()
        video.write_bytes(b'identical source video')
    entries = []
    for key, video in [('A', video_a), ('B', video_b)]:
        model = root / f'{key}.json'
        model.write_text(json.dumps({'berries': [{'id': '1', 'center': [0, 0, 0], 'radius': 1}]}))
        entries.append({'key': key, 'period': 'May', 'selected_model': str(model),
                        'source_videos': [str(video)], 'measured_cloud': str(model)})
    (root / 'final_delivery_manifest.json').write_text(json.dumps({'entries': entries}))
    state = root / 'manual_edits' / 'A' / 'state.json'
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'revision': 7, 'models': [{'id': '1', 'center': [2, 3, 4], 'radius': 2, 'axes': [1, 2, 3], 'manual': True}]}))
    before = hashlib.sha256(state.read_bytes()).hexdigest()
    identity = tmp_path / 'identity.csv'
    identity.write_text('bunch_id,identity_group,source_dir,source_relpath,stage\n'
                        f'I1,G,{video_a.parent},20260528/a,young\n'
                        f'I2,G,{video_b.parent},20260528/b,young\n')
    result = freeze_sources(root, identity, tmp_path / 'out')
    a = next(item for item in result['samples'] if item['sample_id'] == 'A')
    assert a['revision'] == 7
    assert a['models'][0]['center'] == [2, 3, 4]
    assert a['models'][0]['radius'] == 2
    assert a['models'][0]['shape'] == 'sphere'
    assert 'axes' not in a['models'][0]
    assert {item['sample_id'] for item in result['samples'] if item['excluded']} == {'A', 'B'}
    assert hashlib.sha256(state.read_bytes()).hexdigest() == before
    assert json.loads((tmp_path / 'out' / 'samples.json').read_text())['source_hashes'][str(state)] == before
