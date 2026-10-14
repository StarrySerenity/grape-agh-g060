"""Actual masked-RGB fingerprints, reusing the existing CPU metric network."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,csv,hashlib,json,re,time
from functools import lru_cache
from pathlib import Path
import cv2,numpy as np
from growth_data import save,read,digest
cv2.setNumThreads(2)
ROOT=Path(__file__).resolve().parent
PROJECT=Path('E:/PyCharm/Project/grape_berry_2d/.worktrees/gbisc-baseline')
DATA=PROJECT/'datasets/fingerprint_v4_identity_downloads_v2'
WEIGHTS=PROJECT/'outputs/fingerprint_resnet18_metric_downloads/best.pt'


def imread(path,gray=False):
    x=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if x is None:raise ValueError('Unreadable image '+str(path))
    return x


def masked_crop(image,mask):
    if mask.shape!=image.shape[:2]:raise ValueError('Mask dimensions differ')
    yy,xx=np.where(mask>0)
    if not len(xx):raise ValueError('empty target mask')
    height,width=mask.shape;dx=max(1,int((xx.max()-xx.min()+1)*.1));dy=max(1,int((yy.max()-yy.min()+1)*.1))
    x0=max(0,xx.min()-dx);x1=min(width,xx.max()+dx+1);y0=max(0,yy.min()-dy);y1=min(height,yy.max()+dy+1)
    m=mask[y0:y1,x0:x1];a=image[y0:y1,x0:x1]
    return np.where(m[...,None]>0,a,128),m


def shape_descriptor(image,mask):
    crop,m=masked_crop(image,mask);m=cv2.resize(m,(128,128),interpolation=cv2.INTER_NEAREST)
    hu=cv2.HuMoments(cv2.moments(m)).flatten();hu=-np.sign(hu)*np.log10(np.maximum(abs(hu),1e-12))
    rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);fg=rgb[cv2.resize(masked_crop(image,mask)[1],(crop.shape[1],crop.shape[0]),interpolation=cv2.INTER_NEAREST)>0]
    hist=np.histogram(fg[:,0],bins=12,range=(0,180))[0].astype(float);hist/=max(hist.sum(),1)
    widths=(m>0).sum(axis=1);profile=np.array([np.mean(v)/128 for v in np.array_split(widths,12)])
    return np.r_[hu/12,hist,profile].astype(np.float32)


def rank(embeddings,rows,query_index):
    q=embeddings[query_index];length=np.linalg.norm(q)
    if length<1e-9:raise ValueError('Empty fingerprint')
    similarities=embeddings@q/np.maximum(np.linalg.norm(embeddings,axis=1)*length,1e-12)
    out=[];query=rows[query_index]
    for i,s in enumerate(similarities):
        if i==query_index or rows[i]['image_sha256']==query['image_sha256']:continue
        out.append(dict(index=i,identity=rows[i]['identity'],name=rows[i]['name'],cosine=float(s)))
    return sorted(out,key=lambda r:(-r['cosine'],r['name']))


def model_resources(model_key='legacy'):
    if model_key=='legacy':return WEIGHTS,ROOT/'fingerprint_2d'
    if model_key not in ('real_only','mixed'):raise ValueError('Unknown fingerprint model')
    registry=ROOT/'fingerprint_mix_20261014/model_registry.json'
    if not registry.exists():raise ValueError('New fingerprint training is not yet complete')
    entry=read(registry)['models'][model_key]
    weights=ROOT/entry['checkpoint'];gallery=ROOT/entry['gallery']
    if not weights.resolve().is_relative_to(ROOT) or not gallery.resolve().is_relative_to(ROOT):raise ValueError('Invalid local model paths')
    return weights,gallery

@lru_cache(maxsize=3)
def build_network(model_key='legacy'):
    import torch,torch.nn as nn
    from torchvision import models
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=2:torch.set_num_interop_threads(2)
    weights,gallery=model_resources(model_key)
    state=torch.load(weights,map_location='cpu',weights_only=False)
    class Net(nn.Module):
        def __init__(self):
            super().__init__();base=models.resnet18(weights=None);self.backbone=nn.Sequential(*list(base.children())[:-1])
            self.proj=nn.Sequential(nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(inplace=True),nn.Dropout(.15));self.classifier=nn.Linear(256,len(state['classes']))
        def forward(self,x):return torch.nn.functional.normalize(self.proj(self.backbone(x).flatten(1)),dim=1)
    network=Net();network.load_state_dict(state['model'],strict=True);network.eval()
    return torch,network,dict(epoch=int(state['epoch']),metrics=state.get('metrics'),classes=state['classes'],
        checkpoint_path=str(weights),checkpoint_sha256=digest(weights),model_key=model_key,
        caveat='Existing training selected checkpoint by test score; reused-data accuracy is not an independent generalization estimate' if model_key=='legacy' else
        'Frozen ImageNet backbone; metric head selected only on heldout validation identities; local independent test has limited identities')


def source_rows(identity=None):
    allrows=list(csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8-sig')));rows=[]
    for raw in allrows:
        if raw.get('usable','1')!='1' or not raw['image'] or (identity and raw['identity']!=identity):continue
        # Old manifests contain the former D: checkout. Resolve these files by basename
        # inside the current known dataset, without writing to the old manifest.
        image=DATA/'images'/Path(raw['image']).name;mask=DATA/'masks'/Path(raw['mask']).name
        if not image.exists() or not mask.exists():raise FileNotFoundError(image)
        rows.append(dict(identity=raw['identity'],name=raw['original_name'],image=str(image),mask=str(mask),
            split=raw['split'],source_set=raw['source_set'],image_sha256=digest(image),mask_sha256=digest(mask),
            capture_kind='photo' if '__photo_' in raw['original_name'] else 'video_frame',
            video_group=(re.search(r'video(\d+)',raw['original_name']).group(1) if re.search(r'video(\d+)',raw['original_name']) else None)))
    return rows


def extract_identity(identity):
    rows=source_rows(identity);torch,network,networkinfo=build_network();out=ROOT/'fingerprint_2d'/identity;out.mkdir(parents=True,exist_ok=True)
    tensors=[];shapes=[];thumbnails=[];accepted=[];excluded=[]
    for row in rows:
        image=imread(row['image']);mask=imread(row['mask'],True)
        try:crop,m=masked_crop(image,mask)
        except ValueError as e:excluded.append(dict(**row,reason=str(e)));continue
        resized=cv2.resize(crop,(224,224));rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        tensor=torch.from_numpy(((rgb-[.485,.456,.406])/[.229,.224,.225]).astype(np.float32).transpose(2,0,1))
        tensors.append(tensor);shapes.append(shape_descriptor(image,mask));accepted.append(row)
        thumb=cv2.resize(crop,(180,230));canvas=np.full((262,190,3),245,np.uint8);canvas[:230,:180]=thumb
        label=row['name'].split('__',1)[-1][:24];cv2.putText(canvas,label,(4,250),cv2.FONT_HERSHEY_SIMPLEX,.3,(20,20,20),1);thumbnails.append(canvas)
    if not tensors:raise ValueError('No valid masked RGB data for '+identity)
    # A tensor computation is performed within this identity only; no unattended task queue.
    embeddings=[]
    with torch.inference_mode():
        for tensor in tensors:embeddings.append(network(tensor.unsqueeze(0)).numpy()[0])
    array=np.array(embeddings);np.save(out/'embeddings.npy',array);np.save(out/'shape_features.npy',np.array(shapes))
    save(out/'records.json',dict(identity=identity,rows=accepted,excluded=excluded,network=networkinfo,uses_3d_geometry=False,
        descriptor_dimension=256,only_grown_bunch_rgb=True))
    for start in range(0,len(thumbnails),24):
        pieces=thumbnails[start:start+24]
        while len(pieces)%6:pieces.append(np.full_like(pieces[0],245))
        sheet=np.vstack([np.hstack(pieces[i:i+6]) for i in range(0,len(pieces),6)])
        cv2.imencode('.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,89])[1].tofile(str(out/f'review_{start//24}.jpg'))
    print(json.dumps(dict(identity=identity,images=len(accepted),excluded=len(excluded),fingerprint_dim=256,device='cpu',output=str(out))),flush=True)


def query_image(image,mask,model_key='legacy'):
    torch,network,info=build_network(model_key);crop,m=masked_crop(image,mask)
    rgb=cv2.cvtColor(cv2.resize(crop,(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    x=torch.from_numpy(((rgb-[.485,.456,.406])/[.229,.224,.225]).astype(np.float32).transpose(2,0,1))
    with torch.inference_mode():v=network(x[None]).numpy()[0]
    weights,gallery=model_resources(model_key)
    summary=read(gallery/'summary.json');conflicts=set(summary['identity_conflicts_excluded'])
    records=read(gallery/'all_records.json');em=np.load(gallery/'all_embeddings.npy')
    sim=em@v;ranked=[]
    for identity in summary['identities']:
        if identity['identity'] in conflicts:continue
        indices=identity.get('retrieval_image_indices',identity['image_indices']);best=sorted(indices,key=lambda i:float(sim[i]),reverse=True)[:3]
        ranked.append(dict(identity=identity['identity'],cosine=float(np.mean(sim[best])),
            best_view_index=int(best[0]),review=identity['review'],closest_name=records[best[0]]['name']))
    ranked.sort(key=lambda r:(-r['cosine'],r['identity']))
    return dict(fingerprint=v.tolist(),dimension=len(v),candidates=ranked[:5],
        model=info['checkpoint_path'],model_key=model_key,uses_3d_geometry=False,confidence_is_not_probability=True,
        prediction_accepted_as_identity=False)


def assemble():
    records=[];emb=[];shapes=[];identities=[]
    for out in sorted((ROOT/'fingerprint_2d').iterdir()):
        if not out.is_dir() or not (out/'records.json').exists():continue
        result=read(out/'records.json');start=len(records);records.extend(result['rows']);array=np.load(out/'embeddings.npy');emb.extend(array)
        shapes.extend(np.load(out/'shape_features.npy'));mean=array.mean(0);mean/=np.linalg.norm(mean)
        identities.append(dict(identity=out.name,images=len(array),prototype=mean.tolist(),review=f'fingerprint_2d/{out.name}/review_0.jpg',
            image_indices=list(range(start,len(records))),sample_names=[r['name'] for r in result['rows']],
            geometry_used=False,dimension=256))
    emb=np.array(emb);shape=np.array(shapes);np.save(ROOT/'fingerprint_2d/all_embeddings.npy',emb);np.save(ROOT/'fingerprint_2d/all_shape_features.npy',shape)
    duplicate_ids=set();byhash={}
    for row in records:byhash.setdefault(row['image_sha256'],set()).add(row['identity'])
    for ids in byhash.values():
        if len(ids)>1:duplicate_ids.update(ids)
    # Known byte-identical video source pairs from source audit are also excluded.
    conflicted={'FP3_05','FP3_06','FP3_11','FP3_12'}|duplicate_ids
    evaluations=[];cross=[];photo=[]
    for i,row in enumerate(records):
        if row['identity'] in conflicted:continue
        ranked=[r for r in rank(emb,records,i) if r['identity'] not in conflicted]
        match=next((j+1 for j,r in enumerate(ranked) if r['identity']==row['identity']),None)
        evaluations.append(dict(name=row['name'],identity=row['identity'],rank=match,candidates=ranked[:3],
            scope='reused_training_dataset_diagnostic_only'))
        separated=[r for r in ranked if records[r['index']]['capture_kind']!=row['capture_kind'] or (row['capture_kind']=='video_frame' and records[r['index']]['video_group']!=row['video_group'])]
        first=next((j+1 for j,r in enumerate(separated) if r['identity']==row['identity']),None)
        if first is not None:cross.append(dict(name=row['name'],identity=row['identity'],rank=first,scope='different_capture_group_reused_model_diagnostic'))
        if row['capture_kind']=='photo':
            vv=[r for r in ranked if records[r['index']]['capture_kind']=='video_frame']
            m=next((j+1 for j,r in enumerate(vv) if r['identity']==row['identity']),None)
            if m is not None:photo.append(dict(name=row['name'],identity=row['identity'],rank=m))
    def stats(rows):return dict(queries=len(rows),top1=float(np.mean([r['rank']==1 for r in rows])) if rows else None,
        top3=float(np.mean([r['rank'] is not None and r['rank']<=3 for r in rows])) if rows else None)
    summary=dict(identities=identities,total_images=len(records),identity_count=len(identities),dimension=256,
        uses_3d_geometry=False,checkpoint=str(WEIGHTS),reused_frame_diagnostic=stats(evaluations),different_capture_diagnostic=stats(cross),
        photo_to_video_diagnostic=stats(photo),identity_conflicts_excluded=sorted(conflicted),
        generalization_accuracy_claimed=False,limitations=['Existing model used these identities and views during training',
            'Best checkpoint chosen using old test score; the displayed diagnostic is not an unbiased new test',
            'Independent new shoot validation still required'])
    save(ROOT/'fingerprint_2d/summary.json',summary);save(ROOT/'fingerprint_2d/retrieval_diagnostics.json',evaluations)
    save(ROOT/'fingerprint_2d/all_records.json',records)
    print(json.dumps({k:v for k,v in summary.items() if k!='identities'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity');p.add_argument('--assemble',action='store_true');args=p.parse_args()
    if args.assemble:assemble()
    elif args.identity:extract_identity(args.identity)
    else:p.error('Choose one identity to inspect or --assemble after inspection')
