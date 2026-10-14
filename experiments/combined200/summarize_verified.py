"""Postprocess frozen decisions while explicitly reporting a non-input report edit.

The original snapshot and evaluation lock remain intact. Any changed training,
model, image, mask, geometry, matcher or protocol input still stops the summary.
"""
import experiment as e
from pathlib import Path

def check_sources(expected,report_path):
    allowed=Path(report_path).resolve();changes=[]
    for name,expected_hash in expected.items():
        actual=e.sha(name)
        if actual!=expected_hash:
            detail=dict(path=name,expected_sha256=expected_hash,actual_sha256=actual)
            if Path(name).resolve()!=allowed:raise ValueError('Scientific input/source changed: '+name)
            detail.update(role='upstream reporting-only script; not imported or read by this run',
                baseline_rewritten=False,last_modified_epoch=Path(name).stat().st_mtime)
            changes.append(detail)
    return len(expected)-len(changes),changes

def main():
    e.verify_lock();e.splits_check();engine=e.runtime()
    from evaluate import summary as synthetic_summary,legacy_review,detection_metrics
    # This source is neither imported nor read by train.py, data.py, engine200.py,
    # matching_fast.py or experiment.py. Its change cannot alter saved decisions.
    for file in [e.ROOT/'experiment.py',*[e.TRAIN/name for name in ['train.py','data.py','engine200.py','matching_fast.py','evaluate.py']]]:
        if 'build_report' in file.read_text(encoding='utf-8'):raise ValueError('Reporting file became a dependency')
    snapshot=e.read(e.ROOT/'source-snapshot.json')
    unchanged,changes=check_sources(snapshot['protected'],e.SOURCE/'real_consumer_test_20261014/build_report.py')
    e.save(e.ROOT/'source-variance.json',dict(unchanged=unchanged,changes=changes,
        original_snapshot_sha256=e.sha(e.ROOT/'source-snapshot.json'),
        frozen_model_and_decision_inputs_unchanged=True,snapshot_rewritten=False,
        explanation='Reporting script changed after snapshot creation; explicitly distinguished from frozen scientific inputs. No original file restored or overwritten.'))
    synthetic=[e.read(e.RESULTS/'synthetic'/(r['key']+'.json')) for r in e.source_queries() if r['split']=='evaluation']
    assert len(synthetic)==40 and all(r['fresh_image_inference'] and not r['truth_used_for_ranking'] for r in synthetic)
    syn=synthetic_summary(synthetic,e.read(e.OUT/'decision_threshold.json'))
    physical=legacy_review.audit_correspondences(synthetic)
    detections=e.read(e.RESULTS/'real_detections.json');queries=[r for r in detections if r['role']=='query']
    records=[e.read(e.RESULTS/'real'/(r['key']+'.json')) for r in queries]
    assert len(records)==61
    closed=[r['closed'] for r in records];opened=[r['open_set'] for r in records]
    e.save(e.RESULTS/'real_rankings.json',dict(closed=closed,open_set=opened))
    e.save(e.RESULTS/'synthetic_rankings.json',synthetic)
    history=e.read(e.OUT/'detector_history.json');protocol=e.read(e.OUT/'detector_protocol.json')
    assert history[-1]['epoch']==48 and protocol['training_identity_count']==160 and protocol['real_training_images']==0
    result=dict(training=dict(total_dataset_identities=200,training_identity_count=160,old_training_identities=80,new_training_identities=80,
        training_images=3840,development_identities=20,epochs_completed=48,steps_per_epoch=64,
        selected_epoch=e.read(e.OUT/'detector_summary.json')['epoch'],checkpoint_sha256=e.sha(e.OUT/'detector_best.pt'),
        initialization_sha256=protocol['initialization_sha256'],real_training_images=0,executed_by='Agnes via AGH shell tool'),
        synthetic=syn,synthetic_detection=detection_metrics(synthetic),synthetic_physical_correspondence_audit=physical,
        real_closed=e.real_statistics(closed),real_open_set=e.real_statistics(opened),real_protocol=e.read(e.RESULTS/'real_protocol.json'),
        real_per_identity=[dict(identity=identity,**e.real_statistics([r for r in closed if r['expected_identity']==identity]))
            for identity in sorted({r['expected_identity'] for r in closed})],
        protected_source_files_unchanged=unchanged,reporting_only_source_changes=changes,
        snapshot_rewritten=False,frozen_model_and_decision_inputs_unchanged=True,
        postprocess_sha256=e.sha(Path(__file__)),
        scope='Actual balanced synthetic retraining; reused synthetic benchmark regression and same-day real source-group diagnostic. No new independent test, no real fitting or threshold tuning.')
    e.save(e.RESULTS/'summary.json',result)
    print(e.json.dumps({k:result[k] for k in ['training','synthetic','synthetic_detection','real_closed','real_open_set',
        'protected_source_files_unchanged','reporting_only_source_changes','snapshot_rewritten']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
