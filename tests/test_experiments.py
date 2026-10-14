import copy
import pytest


def berries(radius=0.1):
    return [{'id': str(i), 'center': p, 'radius': radius}
            for i, p in enumerate([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])]


def record(key, identity, radius=0.1, period='May'):
    return dict(sample_id=key, bunch_id=identity, models=berries(radius), group='GROW',
                period=period, date='20260528' if period == 'May' else '20260707',
                excluded=False, source_video_keys=['path:' + key], geometry_sha256=key)


def test_rank_removes_self_and_shared_source_before_scoring():
    from experiments import rank_samples
    query = record('query', 'A')
    duplicate = record('duplicate', 'B')
    duplicate['source_video_keys'] = query['source_video_keys']
    ranked = rank_samples([query, duplicate, record('match', 'A', .11), record('other', 'B', .5)], 'query')
    assert [r['sample_id'] for r in ranked] == ['match', 'other']
    assert ranked[0]['bunch_id'] == 'A'


def test_retrieval_denominator_ignores_identities_without_an_independent_match():
    from experiments import evaluate_all
    result = evaluate_all([record('a1', 'A'), record('a2', 'A', .11), record('b1', 'B', .5)])
    assert result['structure']['eligible_queries'] == 2
    assert result['structure']['top1'] == 1.0


def test_forecast_keeps_test_identity_out_of_training_and_mean_baseline():
    from experiments import evaluate_forecast
    samples = []
    for i in range(1, 6):
        samples += [record(f'{i}-early', f'GROW_{i}', .1 * i),
                    record(f'{i}-late', f'GROW_{i}', .2 * i, 'July')]
    result = evaluate_forecast(samples)
    assert result['pairs'] == 5
    first = result['folds'][0]
    assert first['bunch_id'] == 'GROW_1'
    assert first['bunch_id'] not in first['train_ids']
    assert len(first['train_ids']) == 4
    assert first['reference'][0] == pytest.approx(.2)
    assert first['baseline'][0] == pytest.approx(.7)


def test_forecast_refuses_missing_longitudinal_pairs():
    from experiments import evaluate_forecast
    with pytest.raises(ValueError, match='pairs'):
        evaluate_forecast([record('a1', 'A')])
