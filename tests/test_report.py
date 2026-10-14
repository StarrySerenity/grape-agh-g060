from pathlib import Path


def test_portable_cloud_preview_falls_back_to_frozen_preview(tmp_path):
    import json
    from report import cloud_preview
    sample = {'sample_id': 'portable', 'measured_cloud': str(tmp_path / 'missing.ply')}
    archive = tmp_path / 'clouds.json'
    archive.write_text(json.dumps({'portable': [[1., 2., 3.]]}), encoding='utf-8')
    assert cloud_preview(sample, archive) == [[1., 2., 3.]]
