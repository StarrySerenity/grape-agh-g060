"""Lock validation selection, then evaluate held-out photos and export aligned galleries."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import argparse,datetime,hashlib,json
from pathlib import Path
import numpy as np
from train_metric import ROOT,Net,embed,torch
from protocol import retrieval

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load_model(name):
    state=torch.load(ROOT/'models'/name/'best.pt',map_location='cpu',weights_only=False)
    net=Net(len(state['classes']));net.load_state_dict(state['model']);net.eval()
    return net,state

def identity_ranking(vectors,rows,qi,gi):
    ids=sorted({rows[i]['identity'] for i in gi});records=[]
    for i in qi:
        sims=vectors[gi]@vectors[i];ranked=[]
        for identity in ids:
            values=sorted([float(s) for s,j in zip(sims,gi) if rows[j]['identity']==identity],reverse=True)[:3]
            ranked.append((float(np.mean(values)),identity))
        ranked.sort(key=lambda t:(-t[0],t[1]));order=[t[1] for t in ranked]
        records.append(dict(name=rows[i]['name'],identity=rows[i]['identity'],rank=order.index(rows[i]['identity'])+1,
                            candidates=[dict(identity=t[1],cosine=t[0]) for t in ranked[:5]]))
    return dict(queries=len(records),gallery_identities=len(ids),top1=float(np.mean([r['rank']==1 for r in records])),
                top3=float(np.mean([r['rank']<=3 for r in records])),records=records)

def lock_selection():
    if (ROOT/'selection.json').exists():raise RuntimeError('Selection already locked; do not tune after opening test')
    names=['mixed_005','mixed_015','mixed_035'];choices=[dict(name=n,**read(ROOT/'models'/n/'validation.json')) for n in names]
    choices.sort(key=lambda c:(-c['score'],c['weight']))
    selected=choices[0];baseline=read(ROOT/'models/real_only/validation.json')
    # The recommended default is chosen before any test results are read.
    recommended='mixed' if selected['score']>baseline['score']+1e-9 else 'real_only'
    obj=dict(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),criterion='.6 validation frame Rank1 + .4 validation mAP',
             candidates=choices,selected_mixed=selected['name'],recommended_model=recommended,real_only_validation=baseline,
             test_not_used=True)
    save(ROOT/'selection.json',obj);print('SELECTION_LOCKED',selected['name'],recommended,flush=True)

def finalize():
    selection=read(ROOT/'selection.json');rows=read(ROOT/'real_rows.json');split=read(ROOT/'real_protocol.json')
    original=read(ROOT.parent/'fingerprint_2d/all_records.json')
    if rows!=original:raise RuntimeError('Real gallery row order changed; thumbnail indices would be wrong')
    real=np.load(ROOT/'real_features.npz')['features'];oldsummary=read(ROOT.parent/'fingerprint_2d/summary.json')
    models={};tests={};validation={}
    for key,name in [('real_only','real_only'),('mixed',selection['selected_mixed'])]:
        net,state=load_model(name);vectors=embed(net,real)
        if vectors.shape!=(len(rows),256) or not np.isfinite(vectors).all():raise RuntimeError('Invalid embeddings')
        if not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):raise RuntimeError('Embeddings not normalized')
        qi=split['test_query_indices'];gi=split['test_gallery_indices']
        tests[key]=dict(frame=retrieval(vectors[qi],[rows[i]['identity'] for i in qi],vectors[gi],[rows[i]['identity'] for i in gi]),
                        identity=identity_ranking(vectors,rows,qi,gi))
        vqi=split['val_query_indices'];vgi=split['val_gallery_indices']
        validation[key]=dict(frame=state['metrics'],identity=identity_ranking(vectors,rows,vqi,vgi))
        folder=ROOT/'gallery'/key;folder.mkdir(parents=True,exist_ok=True)
        identities=[]
        for item in oldsummary['identities']:
            entry=dict(item);entry.pop('prototype',None)
            entry['retrieval_image_indices']=[i for i in entry['image_indices'] if rows[i]['capture_kind']=='video_frame']
            if not entry['retrieval_image_indices']:raise RuntimeError('Identity has no registered video')
            prototype=vectors[entry['retrieval_image_indices']].mean(0)
            prototype=prototype/max(float(np.linalg.norm(prototype)),1e-9)
            entry['prototype']=prototype.tolist();entry['prototype_source']='registered video frames only'
            identities.append(entry)
        np.save(folder/'all_embeddings.npy',vectors);save(folder/'all_records.json',rows)
        synthetic_vectors=embed(net,np.load(ROOT/'synthetic_features.npz')['features'])
        np.save(folder/'all_synthetic_embeddings.npy',synthetic_vectors)
        save(folder/'identity_fingerprints.json',dict(dimension=256,model=key,source='normalized mean of registered video embeddings',
                fingerprints=[dict(identity=e['identity'],fingerprint=e['prototype'],views=len(e['retrieval_image_indices']))
                              for e in identities if e['identity'] not in split['excluded_ids']]))
        checkpoint=ROOT/'models'/name/'best.pt'
        save(folder/'summary.json',dict(identities=identities,total_images=len(rows),identity_count=len(identities),usable_identity_count=46,
              dimension=256,identity_conflicts_excluded=split['excluded_ids'],uses_3d_geometry=False,
              query_gallery='video frames only; photographs never self-retrieved',test=tests[key],validation=validation[key],
              checkpoint=str(checkpoint),training_real_identities=34,synthetic_identities=state['synthetic_identities'],
              synthetic_weight=state['synthetic_weight'],scope='small local identity-disjoint photo-to-video test; not cross-year/field validation'))
        models[key]=dict(checkpoint=checkpoint.relative_to(ROOT.parent).as_posix(),gallery=folder.relative_to(ROOT.parent).as_posix(),
                         checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),synthetic_weight=state['synthetic_weight'])
    result=dict(training_real_identities=34,training_real_images=len(split['train_indices']),synthetic_identities=100,
                synthetic_images=2400,valid_real_identities=46,total_real_images=len(rows),excluded_ids=split['excluded_ids'],
                test_identities=split['test_ids'],test_queries=len(split['test_query_indices']),validation=validation,test=tests,
                selection=selection,dimension=256,uses_3d_geometry=False,device='cpu',threads=2,
                caveats=['Only six unseen identities and 20 independently photographed queries from this local dataset',
                         'No cross-vineyard/year validation; gallery video frames register test identities',
                         'Frozen ImageNet backbone; new metric head trained on 34 real identities',
                         'Synthetic RGB is illustrative sphere rendering, not photorealistic imagery',
                         'Existing legacy model used these identities; legacy diagnostics are not a fair holdout comparison'])
    save(ROOT/'results.json',result)
    save(ROOT/'model_registry.json',dict(models=models,recommended_model=selection['recommended_model'],results='fingerprint_mix_20261014/results.json'))
    print(json.dumps(dict(test={k:v['identity']['top1'] for k,v in tests.items()},recommended=selection['recommended_model']),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',action='store_true');a=p.parse_args()
    if a.lock:lock_selection()
    else:finalize()
