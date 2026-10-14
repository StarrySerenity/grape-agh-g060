import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['OPENBLAS_NUM_THREADS']='2'
import collections,hashlib,json
from pathlib import Path
import numpy as np

def split_real(rows,conflicts=(),n_val=6,n_test=6,seed=42):
    excluded=set(conflicts);hash_ids=collections.defaultdict(set)
    for r in rows:hash_ids[r['image_sha256']].add(r['identity'])
    for identities in hash_ids.values():
        if len(identities)>1:excluded.update(identities)
    identities=sorted({r['identity'] for r in rows}-excluded)
    eligible=[i for i in identities if {r['capture_kind'] for r in rows if r['identity']==i}>={'photo','video_frame'}]
    rng=np.random.default_rng(seed);groups=collections.defaultdict(list)
    for identity in eligible:groups[identity.split('_')[0] if '_' in identity else 'OTHER'].append(identity)
    pools={k:list(rng.permutation(v)) for k,v in sorted(groups.items())};ordered=[]
    while any(pools.values()):
        for k in pools:
            if pools[k]:ordered.append(str(pools[k].pop()))
    if len(ordered)<n_val+n_test:raise ValueError('Insufficient independently photographed identities')
    val=ordered[:n_val];test=ordered[n_val:n_val+n_test];train=sorted(set(identities)-set(val)-set(test))
    return dict(seed=seed,train_ids=train,val_ids=val,test_ids=test,excluded_ids=sorted(excluded),
                train_indices=[j for j,r in enumerate(rows) if r['identity'] in train],
                val_query_indices=[j for j,r in enumerate(rows) if r['identity'] in val and r['capture_kind']=='photo'],
                val_gallery_indices=[j for j,r in enumerate(rows) if r['identity'] in train+val and r['capture_kind']=='video_frame'],
                test_query_indices=[j for j,r in enumerate(rows) if r['identity'] in test and r['capture_kind']=='photo'],
                test_gallery_indices=[j for j,r in enumerate(rows) if r['identity'] in identities and r['capture_kind']=='video_frame'],
                protocol='identity holdout; independent photos query registered video gallery; no test in model selection')

def retrieval(query,qids,gallery,gids):
    query=np.asarray(query,float);gallery=np.asarray(gallery,float);gids=np.asarray(gids)
    query=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-9);gallery=gallery/np.maximum(np.linalg.norm(gallery,axis=1,keepdims=True),1e-9)
    sim=query@gallery.T;records=[]
    for i,identity in enumerate(qids):
        order=np.argsort(-sim[i],kind='stable');correct=gids[order]==identity;positions=np.where(correct)[0]
        if not len(positions):raise ValueError('Query identity missing from gallery')
        first=int(positions[0]+1);ap=float(np.mean((np.arange(len(positions))+1)/(positions+1)))
        records.append(dict(identity=str(identity),rank=first,ap=ap,top_identity=str(gids[order[0]])))
    return dict(queries=len(records),gallery=len(gallery),rank1=float(np.mean([r['rank']==1 for r in records])),
                rank5=float(np.mean([r['rank']<=5 for r in records])),map=float(np.mean([r['ap'] for r in records])),records=records)

def prepare():
    root=Path(__file__).resolve().parent;project=root.parent
    rows=json.loads((project/'fingerprint_2d/all_records.json').read_text(encoding='utf-8'))
    split=split_real(rows,conflicts=['FP3_05','FP3_06','FP3_11','FP3_12'])
    (root/'real_protocol.json').write_text(json.dumps(split,indent=2,ensure_ascii=False),encoding='utf-8')
    (root/'real_rows.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in split.items() if not k.endswith('indices')},ensure_ascii=False))
    print('Training images',len(split['train_indices']),'val photos',len(split['val_query_indices']),'test photos',len(split['test_query_indices']))

if __name__=='__main__':prepare()
