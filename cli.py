"""Structured numerical tools for the local workbench and the AGH plugin."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def dataset():
    return json.loads((ROOT / 'data' / 'samples.json').read_text(encoding='utf-8'))


def execute(command, sample_id=None, output=None, case_index=0, model_key='real_only'):
    if command in ('latest', 'structure_evaluate', 'growth_evaluate', 'structure_case'):
        import updates
        if command == 'latest': return updates.overview()
        if command == 'structure_evaluate': return updates.structure_evaluate()
        if command == 'growth_evaluate': return updates.growth_evaluate()
        return updates.structure_case(case_index, model_key)
    frozen = dataset()
    samples = frozen['samples']
    if command == 'inventory':
        valid = [s for s in samples if not s['excluded']]
        return dict(ok=True, samples=len(samples), evaluation_samples=len(valid),
                    evaluation_identities=len({s['bunch_id'] for s in valid}), exclusions=frozen['exclusions'],
                    candidate_queries=[s['sample_id'] for s in valid if s['group'] in ('GROW', 'YG')],
                    limitations=frozen['limits'])
    if command in ('describe', 'retrieve'):
        from geometry import describe
        from experiments import rank_samples
        if not sample_id or not re.fullmatch(r'[A-Za-z0-9_]{1,80}', sample_id):
            raise ValueError('Invalid sample ID')
        sample = next((s for s in samples if s['sample_id'] == sample_id), None)
        if sample is None:
            raise ValueError('Unknown sample ID')
        if command == 'describe':
            result = describe(sample['models'])
            return dict(ok=True, sample_id=sample_id, bunch_id=sample['bunch_id'],
                        metrics=result['metrics'], warnings=result['warnings'],
                        identity_excluded=sample['excluded'], descriptor_dimensions=len(result['vector']))
        ranked = rank_samples(samples, sample_id)
        matches = [i + 1 for i, row in enumerate(ranked) if row['bunch_id'] == sample['bunch_id']]
        return dict(ok=True, sample_id=sample_id, reference_identity=sample['bunch_id'],
                    candidates=ranked[:5], reference_rank=min(matches) if matches else None,
                    warning='ranking is an experimental suggestion; no verified identity decision')
    if command == 'evaluate':
        from experiments import evaluate_all
        result = evaluate_all(samples)
        return dict(ok=True, baseline={k: v for k, v in result['baseline'].items() if k != 'queries'},
                    structure={k: v for k, v in result['structure'].items() if k != 'queries'},
                    scopes={scope: {k: v for k, v in value.items() if k != 'queries'} for scope, value in result['scopes'].items()},
                    limitations=['queries exclude self, exact geometry and shared source video',
                                 'no per-berry ground truth; no tuned weights'])
    if command == 'forecast':
        from experiments import evaluate_forecast
        result = evaluate_forecast(samples)
        return dict(ok=True, **{k: v for k, v in result.items() if k != 'folds'},
                    validation='leave-one-biological-identity-out, train-only normalization, fixed alpha=1')
    if command == 'report':
        from report import generate_report
        destination = (ROOT / (output or 'results')).resolve()
        if destination != ROOT / 'results':
            raise ValueError('Report output is limited to this project results directory')
        return dict(ok=True, **generate_report(samples, destination))
    raise ValueError('Unsupported command')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['inventory', 'describe', 'retrieve', 'evaluate', 'forecast', 'report', 'serve', 'latest', 'structure_evaluate', 'growth_evaluate', 'structure_case'])
    parser.add_argument('--sample-id')
    parser.add_argument('--output')
    parser.add_argument('--port', type=int, default=8870)
    parser.add_argument('--case-index', type=int, default=0)
    parser.add_argument('--model-key', choices=['real_only', 'mixed'], default='real_only')
    args = parser.parse_args()
    if args.command == 'serve':
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        if not 1024 <= args.port <= 65535:
            raise ValueError('Invalid local port')
        print(f'http://127.0.0.1:{args.port}/web/', flush=True)
        ThreadingHTTPServer(('127.0.0.1', args.port), partial(SimpleHTTPRequestHandler, directory=str(ROOT))).serve_forever()
        return
    try:
        result = execute(args.command, args.sample_id, args.output, args.case_index, args.model_key)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        print(json.dumps(dict(ok=False, error=str(exc)), ensure_ascii=True))
        sys.exit(2)


if __name__ == '__main__':
    main()
