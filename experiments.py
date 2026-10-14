"""Identity retrieval and identity-held-out forecasting; fixed settings, no test tuning."""
import copy
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from geometry import describe


def independent(a, b):
    return (a['sample_id'] != b['sample_id'] and
            a.get('geometry_sha256') != b.get('geometry_sha256') and
            not set(a.get('source_video_keys', [])).intersection(b.get('source_video_keys', [])))


def rank_samples(samples: list[dict], query_id: str, kind: str = 'structure') -> list[dict]:
    if kind not in ('structure', 'baseline'):
        raise ValueError('Unknown descriptor kind')
    query = next((s for s in samples if s['sample_id'] == query_id), None)
    if query is None or query.get('excluded'):
        raise ValueError('Unknown or identity-conflicted query sample')
    field = 'vector' if kind == 'structure' else 'baseline'
    q = np.asarray(describe(query['models'])[field])
    ranked = []
    for sample in samples:
        if sample.get('excluded') or not independent(query, sample):
            continue
        try:
            vector = np.asarray(describe(sample['models'])[field])
        except ValueError:
            continue
        ranked.append(dict(sample_id=sample['sample_id'], bunch_id=sample['bunch_id'],
                           date=sample.get('date'), distance=float(np.linalg.norm(vector - q))))
    return sorted(ranked, key=lambda s: (s['distance'], s['sample_id']))


def evaluate_all(samples: list[dict]) -> dict:
    result = {}
    for kind in ('baseline', 'structure'):
        queries = []
        for query in samples:
            if query.get('excluded'):
                continue
            ranked = rank_samples(samples, query['sample_id'], kind)
            matches = [i + 1 for i, row in enumerate(ranked) if row['bunch_id'] == query['bunch_id']]
            queries.append(dict(sample_id=query['sample_id'], identity=query['bunch_id'],
                                eligible=bool(matches), correct_rank=min(matches) if matches else None,
                                candidates=ranked[:5]))
        eligible = [q for q in queries if q['eligible']]
        result[kind] = dict(eligible_queries=len(eligible), queries=queries,
                            top1=float(np.mean([q['correct_rank'] == 1 for q in eligible])) if eligible else None,
                            top3=float(np.mean([q['correct_rank'] <= 3 for q in eligible])) if eligible else None)
    dated = {}
    for scope in ('same_date', 'cross_date'):
        queries = []
        for query in samples:
            if query.get('excluded'):
                continue
            ranked = rank_samples(samples, query['sample_id'])
            ranked = [r for r in ranked if (r['date'] == query['date']) == (scope == 'same_date')]
            matches = [i + 1 for i, r in enumerate(ranked) if r['bunch_id'] == query['bunch_id']]
            if matches:
                queries.append(dict(sample_id=query['sample_id'], identity=query['bunch_id'],
                                    correct_rank=min(matches), candidates=ranked[:5]))
        dated[scope] = dict(eligible_queries=len(queries), queries=queries,
                           top1=float(np.mean([q['correct_rank'] == 1 for q in queries])) if queries else None,
                           top3=float(np.mean([q['correct_rank'] <= 3 for q in queries])) if queries else None)
    result['scopes'] = dated
    return result


def evaluate_forecast(samples: list[dict]) -> dict:
    groups = {}
    for sample in samples:
        if sample.get('group') == 'GROW' and not sample.get('excluded'):
            groups.setdefault(sample['bunch_id'], {}).setdefault(sample['period'], []).append(sample)
    pairs = []
    for identity, periods in sorted(groups.items()):
        if len(periods.get('May', [])) == 1 and len(periods.get('July', [])) == 1:
            early, late = periods['May'][0], periods['July'][0]
            a, b = describe(early['models']), describe(late['models'])
            fields = ['median_radius_over_center_scale', 'covariance_minor_major_ratio']
            pairs.append(dict(identity=identity, early=early['sample_id'], late=late['sample_id'],
                              x=a['baseline'], y=[b['metrics'][f] for f in fields],
                              early_metrics=a['metrics'], late_metrics=b['metrics']))
    if len(pairs) < 4:
        raise ValueError('At least four independent longitudinal pairs are required')
    x = np.array([p['x'] for p in pairs])
    y = np.array([p['y'] for p in pairs])
    folds = []
    for index, pair in enumerate(pairs):
        mask = np.arange(len(pairs)) != index
        model = make_pipeline(StandardScaler(), Ridge(alpha=1))
        model.fit(x[mask], y[mask])
        prediction = model.predict(x[index:index + 1])[0]
        folds.append(dict(bunch_id=pair['identity'], early=pair['early'], late=pair['late'],
                          train_ids=[p['identity'] for i, p in enumerate(pairs) if i != index],
                          reference=pair['y'], prediction=prediction.tolist(), baseline=y[mask].mean(axis=0).tolist(),
                          early_metrics=pair['early_metrics'], late_metrics=pair['late_metrics']))
    predictions = np.array([f['prediction'] for f in folds])
    baseline = np.array([f['baseline'] for f in folds])
    return dict(pairs=len(pairs), targets=fields, folds=folds,
                model_mae=np.mean(np.abs(predictions - y), axis=0).tolist(),
                baseline_mae=np.mean(np.abs(baseline - y), axis=0).tolist(),
                model_better_than_baseline=(np.mean(np.abs(predictions-y), axis=0) < np.mean(np.abs(baseline-y), axis=0)).tolist(),
                limits=['ten or fewer biological identities in current data',
                        'targets are dimensionless fitted-model estimates, not ground truth growth',
                        'negative or out-of-range forecasts are retained as model failures'],
                out_of_range_predictions=int(np.sum((predictions < 0).any(axis=1) | (predictions[:, 1] > 1))))


def robustness(sample, seed=20261013):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    original = np.array(describe(sample['models'])['vector'])
    moved = copy.deepcopy(sample['models'])
    for berry in moved:
        berry['center'] = (5 * q @ np.array(berry['center']) + [2, 3, 4]).tolist()
        berry['radius'] *= 5
    transform_error = float(np.linalg.norm(np.array(describe(moved)['vector']) - original))
    keep = rng.choice(len(moved), max(4, int(len(moved) * .7)), replace=False)
    subset_error = float(np.linalg.norm(np.array(describe([moved[i] for i in keep])['vector']) - original))
    return dict(sample_id=sample['sample_id'], type='synthetic robustness only',
                transform_distance=transform_error, remove_30_percent_distance=subset_error)
