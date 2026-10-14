"""Isolated AGH training evaluation: synthetic calibration, real photo transfer."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import argparse,base64,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path('E:/grape_growth_workspace_20261013')
TRAIN=ROOT/'training';OUT=TRAIN/'synthetic_consumer';RESULTS=ROOT/'results'
REAL=SOURCE/'real_consumer_test_20261014'
REF=ROOT/'reference'
UNKNOWN={'FP3_17','FP3_18','FP3_19','FP3_20'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_identity_splits(splits):
    seen=set()
    for name in ['training','development','previous_test_regression','fresh_evaluation']:
        values=splits[name]
        if len(values)!=len(set(values)) or set(values)&seen:raise ValueError('Geometry identity leakage: '+name)
        seen.update(values)
    return True

def validate_real_sources(enrollment,queries):
    if {r['image_sha256'] for r in enrollment}&{r['image_sha256'] for r in queries}:
        raise ValueError('Same pixels used for enrollment and query')
    return True

def real_statistics(rows):
    known=[r for r in rows if r['expected_in_database']];unknown=[r for r in rows if not r['expected_in_database']]
    return dict(queries=len(rows),known_queries=len(known),unknown_queries=len(unknown),
        known_top1=sum(r['decision']['top1']==r['expected_identity'] for r in known),
        known_top3=sum(r['expected_identity'] in [c['identity'] for c in r['ranking']['candidates'][:3]] for r in known),
        known_correct_accepted=sum(r['decision']['predicted_identity']==r['expected_identity'] for r in known),
        known_wrong_accepted=sum(r['decision']['accepted'] and r['decision']['predicted_identity']!=r['expected_identity'] for r in known),
        known_rejected=sum(not r['decision']['accepted'] for r in known),
        unknown_correct_rejected=sum(not r['decision']['accepted'] for r in unknown),
        unknown_wrong_accepted=sum(r['decision']['accepted'] for r in unknown),
        insufficient_queries=sum(r['ranking']['status']=='insufficient' for r in rows))

def runtime():
    # Local data.py fixes the read-only source BASE and isolates all output paths.
    sys.path.insert(0,str(TRAIN))
    import engine200
    return engine200

def splits_check():
    splits=read(SOURCE/'synthetic_expansion_200_20261014/proposed_training_split.json')
    validate_identity_splits(splits)
    assert [len(splits[k]) for k in ['training','development','previous_test_regression','fresh_evaluation']]==[160,20,10,10]
    assert sum(int(s.split('_')[1])<100 for s in splits['training'])==80
    assert sum(int(s.split('_')[1])>=100 for s in splits['training'])==80
    return splits

def source_queries():return read(REF/'synthetic_training_200_20261014/synthetic_consumer/query_manifest.json')

def synthetic_infer(row,engine):
    import cv2,numpy as np
    src=SOURCE/'synthetic_training_200_20261014'
    if sha(src/row['image'])!=row['image_sha256']:raise ValueError('Synthetic JPEG changed')
    image=cv2.imdecode(np.fromfile(str(src/row['image']),np.uint8),1)
    mask=cv2.imdecode(np.fromfile(str(src/row['mask']),np.uint8),0)
    nodes=engine.detect_image(image,mask,version='new200')['nodes']
    ranking=engine.rank_nodes(nodes)
    return dict(key=row['key'],split=row['split'],identity=row['identity'],in_database=row['in_database'],
        nodes=nodes,ground_truth_nodes=row['ground_truth_nodes'],ranking=ranking,fresh_image_inference=True,
        truth_used_for_ranking=False,image_sha256=sha(src/row['image']),mask_sha256=sha(src/row['mask']))

def calibrate():
    splits_check()
    if (ROOT/'evaluation-lock.json').exists():raise ValueError('Calibration already locked; do not retune')
    if not (OUT/'detector_summary.json').exists():raise ValueError('Training has not completed')
    engine=runtime()
    save(OUT/'gallery.json',read(REF/'synthetic_training_200_20261014/synthetic_consumer/gallery.json'))
    from evaluate import calibrate as choose_threshold
    records=[]
    for row in [r for r in source_queries() if r['split']=='development']:
        record=synthetic_infer(row,engine);records.append(record)
        save(RESULTS/'development.json',records)
        print('DEVELOPMENT',row['key'],'top1',record['ranking']['candidates'][0]['identity'],flush=True)
    threshold=choose_threshold(records,'agh_retrained200')
    threshold['detector_sha256']=sha(OUT/'detector_best.pt')
    threshold['actual_selection']='AGH retraining synthetic development only, before this run real inference/rankings'
    save(OUT/'decision_threshold.json',threshold)
    paths=[OUT/'detector_best.pt',OUT/'decision_threshold.json',OUT/'gallery.json',ROOT/'experiment.py',TRAIN/'matching_fast.py',TRAIN/'engine200.py']
    save(ROOT/'evaluation-lock.json',dict(files={str(p):sha(p) for p in paths},recalibrated_on_real=False,
        evaluation_scope='Reused already-inspected synthetic benchmark and real source groups; not new independent test'))
    print('MODEL_AND_THRESHOLD_LOCKED',json.dumps(threshold,ensure_ascii=False),flush=True)

def verify_lock():
    lock=read(ROOT/'evaluation-lock.json')
    for p,h in lock['files'].items():
        if sha(p)!=h:raise ValueError('Frozen evaluation input changed: '+p)
    return lock

def synthetic_batch(index):
    verify_lock();engine=runtime()
    rows=[r for r in source_queries() if r['split']=='evaluation']
    assert len(rows)==40
    for row in rows[index*4:index*4+4]:
        result=synthetic_infer(row,engine)
        result['accepted']=engine.accepted(result['ranking'],read(OUT/'decision_threshold.json'))
        save(RESULTS/'synthetic'/(row['key']+'.json'),result)
        print('SYNTHETIC',row['key'],'top1',result['ranking']['candidates'][0]['identity'],'accepted',result['accepted'],flush=True)

def real_detect():
    verify_lock();splits_check();engine=runtime()
    import cv2,numpy as np
    enroll=read(REF/'real_consumer_test_20261014/gallery_complete.json')['records']
    queries=read(REF/'real_consumer_test_20261014/queries_complete.json')
    validate_real_sources(enroll,queries)
    assert len(enroll)==216 and len(queries)==61
    protocol=read(REF/'real_consumer_test_20261014/protocol_complete.json')
    assert protocol['source_date']=='20260707' and protocol['total_folders']==20
    records=[]
    for role,rows in [('enrollment',enroll),('query',queries)]:
        for row in rows:
            for k,h in [('source_image','image_sha256'),('source_mask','mask_sha256')]:
                if sha(row[k])!=row[h]:raise ValueError('Real source changed: '+row[k])
            image=cv2.imdecode(np.fromfile(row['source_image'],np.uint8),1)
            mask=cv2.imdecode(np.fromfile(row['source_mask'],np.uint8),0)
            detected=engine.detect_image(image,mask,crop_target=True,version='new200')
            # Ignore copied prediction fields; every gallery and query image is inferred anew.
            result={k:v for k,v in row.items() if k not in ['nodes','transform','model_version','image','overlay']}
            result.update(role=role,nodes=detected['nodes'],transform=detected['transform'],
                detector_sha256=sha(OUT/'detector_best.pt'),fresh_image_inference=True,labels_used_for_detection=False)
            prepared=cv2.imdecode(np.frombuffer(base64.b64decode(detected['image_base64']),np.uint8),1)
            path=RESULTS/'images'/(row['key']+'.jpg');path.parent.mkdir(parents=True,exist_ok=True)
            cv2.imencode('.jpg',prepared,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(path))
            result['prepared_image']=str(path.relative_to(ROOT))
            records.append(result)
            if len(records)%12==0:print('REAL_DETECTION',len(records),'/',len(enroll)+len(queries),role,row['identity'],flush=True)
    save(RESULTS/'real_detections.json',records)
    save(RESULTS/'real_protocol.json',dict(source_date='20260707',source_directory='E:/grape_data_jurong/20260707/3',
        source_folders=20,unique_source_groups=18,aliases=protocol['aliases'],enrollment_views=216,query_photos=61,
        detector_sha256=sha(OUT/'detector_best.pt'),threshold=read(OUT/'decision_threshold.json'),
        trained_on_real=False,threshold_retuned_on_real=False,masks_supplied=True,
        gallery_nodes_from_detector=True,query_nodes_from_detector=True,
        scope='real photo-to-video source-group transfer diagnostic, images previously inspected in upstream experiments'))
    print('REAL_GALLERY_AND_QUERY_REDETECTED',len(records),flush=True)

def rank_real(nodes,enrollment,engine):
    if len(nodes)<6:return dict(status='insufficient',candidates=[],margin=0.,query_node_count=len(nodes),elapsed_seconds=0.)
    start=time.time();results=[]
    for identity in sorted({r['identity'] for r in enrollment}):
        options=[dict(key=r['key'],identity=identity,**engine.match_view(nodes,r['nodes'])) for r in enrollment if r['identity']==identity]
        options.sort(key=lambda r:-r['score']);best=options[0]
        best['view_score']=best['score'];best['supporting_view_score']=options[1]['score']
        best['score']=.85*best['view_score']+.15*best['supporting_view_score'];results.append(best)
    results.sort(key=lambda r:-r['score'])
    return dict(status='ranked',candidates=results,margin=results[0]['score']-results[1]['score'],query_node_count=len(nodes),
        elapsed_seconds=time.time()-start,uses_rgb_for_matching=False,identity_confirmed=False)

def real_decision(ranking,threshold,engine):
    accepted=engine.accepted(ranking,threshold)
    return dict(accepted=accepted,predicted_identity=ranking['candidates'][0]['identity'] if accepted else None,
        top1=ranking['candidates'][0]['identity'] if ranking['candidates'] else None)

def real_batch(index):
    verify_lock();engine=runtime();records=read(RESULTS/'real_detections.json')
    enrollment=[r for r in records if r['role']=='enrollment'];queries=[r for r in records if r['role']=='query']
    validate_real_sources(enrollment,queries)
    threshold=read(OUT/'decision_threshold.json')
    for row in queries[index*5:index*5+5]:
        ranking=rank_real(row['nodes'],enrollment,engine)
        closed=dict(key=row['key'],expected_identity=row['identity'],expected_in_database=True,ranking=ranking,
            decision=real_decision(ranking,threshold,engine))
        candidates=[c for c in ranking['candidates'] if c['identity'] not in UNKNOWN]
        opened=dict(ranking,candidates=candidates,margin=candidates[0]['score']-candidates[1]['score'] if len(candidates)>1 else 0.)
        open_record=dict(key=row['key'],expected_identity=row['identity'] if row['identity'] not in UNKNOWN else None,
            expected_in_database=row['identity'] not in UNKNOWN,ranking=opened,decision=real_decision(opened,threshold,engine))
        save(RESULTS/'real'/(row['key']+'.json'),dict(closed=closed,open_set=open_record))
        print('REAL',row['key'],'truth',row['identity'],'closed',closed['decision'],'open',open_record['decision'],flush=True)

def summarize():
    verify_lock();splits=splits_check();engine=runtime()
    from evaluate import summary as synthetic_summary,legacy_review,detection_metrics
    synthetic=[read(RESULTS/'synthetic'/(r['key']+'.json')) for r in source_queries() if r['split']=='evaluation']
    assert len(synthetic)==40 and all(r['fresh_image_inference'] and not r['truth_used_for_ranking'] for r in synthetic)
    syn=synthetic_summary(synthetic,read(OUT/'decision_threshold.json'))
    physical=legacy_review.audit_correspondences(synthetic)
    detections=read(RESULTS/'real_detections.json');queries=[r for r in detections if r['role']=='query']
    records=[read(RESULTS/'real'/(r['key']+'.json')) for r in queries]
    assert len(records)==61
    closed=[r['closed'] for r in records];opened=[r['open_set'] for r in records]
    save(RESULTS/'real_rankings.json',dict(closed=closed,open_set=opened))
    save(RESULTS/'synthetic_rankings.json',synthetic)
    # Hash protected inputs again after actual training and inference, not just before.
    snapshot=read(ROOT/'source-snapshot.json');changed=[]
    for p,h in snapshot['protected'].items():
        if sha(p)!=h:changed.append(p)
    if changed:raise ValueError('Protected source changed: '+str(changed[:10]))
    history=read(OUT/'detector_history.json');protocol=read(OUT/'detector_protocol.json')
    assert history[-1]['epoch']==48 and protocol['training_identity_count']==160 and protocol['real_training_images']==0
    result=dict(training=dict(total_dataset_identities=200,training_identity_count=160,old_training_identities=80,new_training_identities=80,
        training_images=3840,development_identities=20,epochs_completed=48,steps_per_epoch=64,
        selected_epoch=read(OUT/'detector_summary.json')['epoch'],checkpoint_sha256=sha(OUT/'detector_best.pt'),
        initialization_sha256=protocol['initialization_sha256'],real_training_images=0,executed_by='Agnes via AGH shell tool'),
        synthetic=syn,synthetic_detection=detection_metrics(synthetic),synthetic_physical_correspondence_audit=physical,
        real_closed=real_statistics(closed),real_open_set=real_statistics(opened),real_protocol=read(RESULTS/'real_protocol.json'),
        real_per_identity=[dict(identity=identity,**real_statistics([r for r in closed if r['expected_identity']==identity]))
            for identity in sorted({r['expected_identity'] for r in closed})],
        protected_source_files_unchanged=len(snapshot['protected']),
        scope='Actual balanced synthetic retraining; reused synthetic benchmark regression and same-day real source-group diagnostic. No new independent test, no real fitting or threshold tuning.')
    save(RESULTS/'summary.json',result)
    print(json.dumps({k:result[k] for k in ['training','synthetic','synthetic_detection','real_closed','real_open_set','protected_source_files_unchanged']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['calibrate','synthetic','real-detect','real','summarize','verify'],required=True)
    p.add_argument('--batch',type=int);a=p.parse_args()
    if a.phase=='calibrate':calibrate()
    elif a.phase=='synthetic':
        if a.batch is None or not 0<=a.batch<10:p.error('synthetic batch must be 0..9')
        synthetic_batch(a.batch)
    elif a.phase=='real-detect':real_detect()
    elif a.phase=='real':
        if a.batch is None or not 0<=a.batch<13:p.error('real batch must be 0..12')
        real_batch(a.batch)
    elif a.phase=='summarize':summarize()
    else:splits_check();print('200 geometry identities split without leakage; local runtime output isolated.')
