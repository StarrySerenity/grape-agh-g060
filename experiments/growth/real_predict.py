"""Frozen early inputs -> held-out morphology, followed by late-only comparison."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'synthetic'))
from demo_core import predict_young
from forecast import metrics,TARGETS
from growth_data import read,save,longitudinal_pairs
from growth_geometry import align_models,arrays
from export_growth import sphere_cloud

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def predict_case(sample,pairs,group,conditional=False):
    future=bool(sample.get('uses_future_reference') or any(m.get('uses_future_reference') for m in sample['models']))
    if future and not conditional:raise ValueError('Late-assisted input requires explicit conditional illustration')
    # Same adapter pattern as the existing interactive server: geometry-only view
    # inside the unmodified predictor, with provenance explicitly restored outside.
    geometry=dict(models=[dict(id=str(m['id']),center=copy.deepcopy(m['center']),radius=m['radius'],shape='sphere') for m in sample['models']],
        bunch_id=sample.get('bunch_id'),uses_future_reference=False)
    result=predict_young(geometry,pairs,group)
    if sample.get('bunch_id') in result['forecast']['training_identities']:raise ValueError('Own identity leaked into training')
    result.update(input_uses_future_reference=future,independent_early_prediction=not future,
        mode='late_assisted_conditional_illustration' if future else 'held_out_early_forecast')
    if future:
        result['forecast'].update(uses_future_reference=True,independent_early_prediction=False)
        for part in [result['forecast']['layout'],result['grown']]:
            part['uses_future_reference']=True
            for m in part['models']:m['uses_future_reference']=True
    return result

def compare_metrics(predicted,late):
    p=metrics(predicted);q=metrics(late)
    return dict(targets=TARGETS,predicted=[p[k] for k in TARGETS],reference=[q[k] for k in TARGETS],
        absolute_errors=[abs(p[k]-q[k]) for k in TARGETS],dimensionless=True,
        actual_prediction_geometry=True,individual_berry_growth_accuracy=False)

def compare_late(predicted,late):
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import pdist
    alignment=align_models(predicted,late)
    transformed=copy.deepcopy(late)
    for m,center in zip(transformed,alignment['transformed_late_centers']):
        m['center']=center;m['radius']*=alignment['scale']
    p,_=arrays(predicted);q,_=arrays(transformed);unit=float(np.median(pdist(p)))
    chamfer=float((cKDTree(q).query(p)[0].mean()+cKDTree(p).query(q)[0].mean())/(2*unit))
    return dict(metrics=compare_metrics(predicted,late),alignment=alignment,aligned_late_models=transformed,
        normalized_center_chamfer=chamfer,registration_uses_late_only_after_prediction_saved=True,
        scale_is_for_relative_comparison_not_physical_growth=True,physical_correspondence_verified=False)

def run(identity):
    allpairs=longitudinal_pairs(read(ROOT/'data/snapshot.json')['samples'])
    pair=next(p for p in allpairs if p['identity']==identity)
    if pair['status']!='ready':raise ValueError('Unconfirmed early stage excluded')
    folder=ROOT/'real'/identity;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'record.json').exists():
        print('ALREADY_COMPLETED',identity,flush=True);return read(folder/'record.json')
    early=read(ROOT/'inputs'/identity/'early_original.json')
    repaired_raw=read(ROOT/'inputs'/identity/'candidate_for_editor.json')
    repaired=dict(sample_id=early['sample_id'],bunch_id=identity,group=pair['group'],
        models=repaired_raw['berries'],uses_future_reference=True,
        repair_provenance='Existing candidate created while comparing the same late reference; conditional illustration only.')
    save(folder/'early_original.json',early);save(folder/'early_repaired.json',repaired)
    results={};prediction_hashes={}
    for mode,sample in [('original',early),('repaired_conditional',repaired)]:
        start=time.time();result=predict_case(sample,allpairs,pair['group'],conditional=mode=='repaired_conditional')
        target=folder/(mode+'_prediction.json');save(target,result);prediction_hashes[mode]=sha(target)
        save(folder/(mode+'_grown.json'),dict(models=result['grown']['models'],
            uses_future_reference=result['input_uses_future_reference'],
            independent_early_prediction=result['independent_early_prediction'],
            absolute_scale_is_assumed=True,is_prediction_not_measurement=True))
        sphere_cloud(result['grown']['models'],folder/(mode+'_grown.ply'))
        results[mode]=dict(result=result,seconds=time.time()-start)
    # The query's own late geometry is only opened after both predictions are frozen.
    late=read(ROOT/'inputs'/identity/'late_original.json')
    save(folder/'late_reference.json',late)
    record=dict(identity=identity,group=pair['group'],days=pair['days'],early_sample=early['sample_id'],late_sample=late['sample_id'],
        original_count=len(early['models']),repaired_count=len(repaired['models']),late_count=len(late['models']),
        prediction_files_saved_before_comparison=True,prediction_sha256_before_comparison=prediction_hashes,variants={})
    for mode,item in results.items():
        result=item['result'];comparison=compare_late(result['grown']['models'],late['models'])
        save(folder/(mode+'_comparison.json'),comparison)
        assert sha(folder/(mode+'_prediction.json'))==prediction_hashes[mode]
        record['variants'][mode]=dict(independent_early_prediction=result['independent_early_prediction'],
            input_uses_future_reference=result['input_uses_future_reference'],training_identities=result['forecast']['training_identities'],
            forecast_targets=result['forecast']['prediction'],actual_generated_metrics=comparison['metrics'],
            normalized_center_chamfer=comparison['normalized_center_chamfer'],
            contact_constraint=result['grown']['contact_constraint'],
            radius_ratio=result['grown']['median_radius_ratio'],center_spacing_ratio=result['grown']['center_spacing_ratio'],
            comparison_uses_similarity_registration=True,seconds=item['seconds'])
    save(folder/'record.json',record)
    print(json.dumps(dict(identity=identity,original_count=record['original_count'],late_count=record['late_count'],
        errors={m:r['actual_generated_metrics']['absolute_errors'] for m,r in record['variants'].items()},
        contact={m:r['contact_constraint']['status'] for m,r in record['variants'].items()}),ensure_ascii=False),flush=True)
    return record

def verify():
    manifest=read(ROOT/'source-manifest.json')
    for p,h in manifest['sha256'].items():
        if sha(p)!=h:raise ValueError('Source input changed: '+p)
    records=[]
    for identity in manifest['identities']:
        folder=ROOT/'real'/identity;record=read(folder/'record.json')
        early=read(folder/'early_original.json');repair=read(folder/'early_repaired.json')
        for mode,sample in [('original',early),('repaired_conditional',repair)]:
            raw=read(folder/(mode+'_prediction.json'));grown=raw['grown']['models']
            assert [m['id'] for m in grown]==[str(m['id']) for m in sample['models']]
            assert identity not in raw['forecast']['training_identities']
            assert raw['independent_early_prediction']==(mode=='original')
            assert sha(folder/(mode+'_prediction.json'))==record['prediction_sha256_before_comparison'][mode]
            assert raw['grown']['contact_constraint']['maximum_movement_over_radius']<=.800001
            assert len(read(folder/(mode+'_comparison.json'))['metrics']['absolute_errors'])==4
            if mode!='original':assert all(m['uses_future_reference'] for m in grown)
        records.append(record)
    synthetic=read(ROOT/'synthetic/results.json');assert synthetic['samples']==6 and synthetic['berries']==306
    result=dict(real_pairs=len(records),variants_per_pair=2,synthetic_cases=6,synthetic_berries=306,
        protected_source_files_unchanged=len(manifest['sha256']),same_input_output_ids=True,own_identity_excluded=True,
        future_repair_flag_preserved=True,prediction_files_unchanged_after_late_comparison=True,
        no_physical_growth_scale=True,geometry_accuracy_is_not_individual_biological_accuracy=True,records=records)
    save(ROOT/'results/showcase_summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--identity');parser.add_argument('--all',action='store_true');parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.verify:verify()
    elif args.all:
        for identity in read(ROOT/'source-manifest.json')['identities']:run(identity)
    elif args.identity:
        if args.identity not in read(ROOT/'source-manifest.json')['identities']:parser.error('Unknown ready identity')
        run(args.identity)
    else:parser.error('Choose --identity, --all or --verify')
