"""Compute reproducible results and portable visualization data."""
import csv
import json
from pathlib import Path
import numpy as np
from geometry import describe
from experiments import evaluate_all, evaluate_forecast, rank_samples, robustness


def read_cloud_preview(path, limit=1200):
    path = Path(path)
    if not path.exists():
        return []
    types = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
             'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int': '<i4', 'uint': '<u4'}
    with path.open('rb') as stream:
        properties, count, vertex, encoding = [], 0, False, None
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError('PLY header incomplete')
            line = raw.decode('ascii').strip()
            if line.startswith('format '): encoding = line.split()[1]
            if line.startswith('element '):
                vertex = line.startswith('element vertex ')
                if vertex: count = int(line.split()[-1])
            if vertex and line.startswith('property '):
                fields = line.split()
                if len(fields) != 3 or fields[1] not in types:
                    raise ValueError('Unsupported PLY vertex property')
                properties.append((fields[2], types[fields[1]]))
            if line == 'end_header': break
        if encoding == 'binary_little_endian':
            data = np.fromfile(stream, dtype=np.dtype(properties), count=count)
            xyz = np.column_stack([data[k] for k in ('x', 'y', 'z')])
        elif encoding == 'ascii':
            xyz = np.loadtxt(stream, max_rows=count, usecols=tuple([name for name, _ in properties].index(k) for k in ('x', 'y', 'z')))
        else:
            raise ValueError('Unsupported PLY encoding')
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError('Cloud contains invalid points')
    return xyz[np.linspace(0, len(xyz)-1, min(limit, len(xyz))).astype(int)].round(9).tolist()


def cloud_preview(sample, archive=None):
    points = read_cloud_preview(sample['measured_cloud'])
    if points:
        return points
    archive = Path(archive or Path(__file__).resolve().parent / 'data' / 'cloud-preview.json')
    if archive.exists():
        return json.loads(archive.read_text(encoding='utf-8')).get(sample['sample_id'], [])
    return []


def generate_report(samples, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    retrieval, forecast = evaluate_all(samples), evaluate_forecast(samples)
    checks = [robustness(s) for s in samples if not s['excluded']]
    rows = []
    for sample in samples:
        description = describe(sample['models'])
        preview = dict(sample_id=sample['sample_id'], bunch_id=sample['bunch_id'],
                       date=sample['date'], group=sample['group'], stage=sample['stage'],
                       excluded=sample['excluded'], revision=sample['revision'],
                       metrics=description['metrics'], models=sample['models'],
                       candidates=[] if sample['excluded'] else rank_samples(samples, sample['sample_id'])[:5])
        try:
            preview['points'] = cloud_preview(sample)
        except (ValueError, OSError) as exc:
            preview.update(points=[], cloud_preview_error=str(exc))
        rows.append(preview)
    payload = dict(samples=rows, retrieval=retrieval, forecast=forecast, robustness=checks,
                   disclaimer='球模型是重建与人工修正的估计；检索与预测为探索性验证，不是农业真值。')
    for name, value in [('retrieval', retrieval), ('forecast', forecast), ('robustness', checks), ('dashboard', payload)]:
        (destination / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (destination / 'dashboard.js').write_text('window.GRAPE_DATA=' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + ';', encoding='utf-8')
    with (destination / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['sample_id', 'bunch_id', 'date', 'excluded', *rows[0]['metrics'].keys()]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows: writer.writerow({**{k: row[k] for k in fields if k in row}, **row['metrics']})
    figures(destination, retrieval, forecast)
    summary = dict(sample_count=len(samples), evaluation_samples=sum(not s['excluded'] for s in samples),
                   structure_top1=retrieval['structure']['top1'], baseline_top1=retrieval['baseline']['top1'],
                   forecast_better_than_baseline=forecast['model_better_than_baseline'],
                   max_transform_distance=max(c['transform_distance'] for c in checks),
                   report='results/dashboard.js', metrics='results/metrics.csv')
    (destination / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def figures(destination, retrieval, forecast):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 12})
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    labels = ['基础摘要\n47次查询', '结构描述\n47次查询', '同日期结构\n14次查询', '跨日期结构\n47次查询']
    values = [retrieval['baseline']['top1'], retrieval['structure']['top1'],
              retrieval['scopes']['same_date']['top1'], retrieval['scopes']['cross_date']['top1']]
    bars = ax.bar(labels, np.array(values) * 100, color=['#789589', '#235e47', '#9abd8e', '#d39c63'], width=.55)
    ax.bar_label(bars, fmt='%.1f%%', padding=5)
    ax.set_ylim(0, max(30, max(values)*125))
    ax.set_ylabel('Top-1 (%)'); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('不同拍摄记录的身份检索：结果仍需改进', loc='left', pad=20)
    fig.savefig(destination/'retrieval.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    for i, label in enumerate(['中位半径 / 果串距离尺度', '最小 / 最大协方差特征值']):
        bars = axes[i].bar(['训练集均值', '留一法 Ridge'], [forecast['baseline_mae'][i], forecast['model_mae'][i]], color=['#789589', '#235e47'], width=.55)
        axes[i].bar_label(bars, fmt='%.4f', padding=4)
        axes[i].set_ylabel('MAE（无量纲）'); axes[i].set_title(label, fontsize=12)
        axes[i].spines[['top', 'right']].set_visible(False)
        axes[i].set_ylim(0, max(forecast['baseline_mae'][i], forecast['model_mae'][i])*1.25)
    fig.suptitle('10个身份留一验证：预测没有超过均值基线', fontsize=15)
    fig.savefig(destination/'forecast.png', dpi=180); plt.close(fig)
