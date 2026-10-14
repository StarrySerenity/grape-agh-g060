import copy
import numpy as np
import pytest


def models():
    xyz = np.random.default_rng(42).normal(size=(18, 3))
    return [{'id': str(i), 'center': p.tolist(), 'radius': 0.2 + i * 0.005} for i, p in enumerate(xyz)]


def test_descriptor_is_invariant_to_rotation_translation_and_scale():
    from geometry import describe
    original = models()
    q, _ = np.linalg.qr(np.random.default_rng(13).normal(size=(3, 3)))
    changed = copy.deepcopy(original)
    for model in changed:
        model['center'] = (3.7 * q @ np.array(model['center']) + [8, -4, 2]).tolist()
        model['radius'] *= 3.7
    a, b = describe(original), describe(changed)
    assert len(a['vector']) == 72
    np.testing.assert_allclose(a['vector'], b['vector'], atol=1e-10)
    assert a['metrics']['berry_count'] == 18


@pytest.mark.parametrize('bad', ['nan', 'radius', 'few', 'duplicate'])
def test_invalid_or_degenerate_models_are_rejected(bad):
    from geometry import describe
    data = models()
    if bad == 'nan': data[0]['center'][0] = float('nan')
    if bad == 'radius': data[0]['radius'] = 0
    if bad == 'few': data = data[:3]
    if bad == 'duplicate':
        for model in data: model['center'] = [1, 1, 1]
    with pytest.raises(ValueError):
        describe(data)
