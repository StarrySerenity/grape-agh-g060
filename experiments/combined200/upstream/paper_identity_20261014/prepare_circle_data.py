"""Select stable real windows; preserve exact projected sphere-center supervision."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,sys,re,hashlib,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'circle_training';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT.parent/'structure_fingerprint_20261014'))
from detection import prepare_crop
from fingerprint_2d import imread
from circles import fit_circle
cv2.setNumThreads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def pic(p,im):cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,90])[1].tofile(str(p))
def letterbox(im,nodes,mask=None,size=256):
 h,w=im.shape[:2];s=min(size/w,size/h);nw,nh=round(w*s),round(h*s);dx=(size-nw)//2;dy=(size-nh)//2
 out=np.full((size,size,3),128,np.uint8);out[dy:dy+nh,dx:dx+nw]=cv2.resize(im,(nw,nh))
 labels=[dict(x=n['x']*s+dx,y=n['y']*s+dy,radius=n['radius']*s,visibility=n.get('visibility',1),confidence=n.get('confidence',1)) for n in nodes]
 fg=np.zeros((size,size),np.uint8)
 if mask is not None:fg[dy:dy+nh,dx:dx+nw]=cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST)>0
 else:fg[dy:dy+nh,dx:dx+nw]=1
 return out,labels,fg,dict(scale=s,offset=[dx,dy],source_width=w,source_height=h)

def main():
 rows=read(ROOT.parent/'fingerprint_2d/all_records.json');graphs=read(ROOT.parent/'structure_training_20261014/real_graphs.json');protocol=read(ROOT.parent/'structure_training_20261014/protocol.json')
 selected=[];tiles=[];qa=OUT/'qa';qa.mkdir(exist_ok=True)
 trainids=[v for v in protocol['train_ids'] if not v.startswith('YG')][:18]
 for identity in trainids+['FP2_09','FP3_02','GROW_03']:
  indices=[i for i,r in enumerate(rows) if r['identity']==identity and r['capture_kind']=='video_frame' and graphs[i] and graphs[i]['valid']]
  candidates=[]
  for i in indices:
   row=rows[i];image=imread(row['image']);mask=imread(row['mask'],True);yy,xx=np.where(mask>0)
   crop,target=prepare_crop(image,mask)
   sharp=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
   center=np.array([xx.mean()/image.shape[1],yy.mean()/image.shape[0]]);extent=np.array([np.ptp(xx)/image.shape[1],np.ptp(yy)/image.shape[0]])
   candidates.append(dict(index=i,frame_number=int(re.search(r'frame_(\d+)',row['name'])[1]),group=row['video_group'],sharpness=sharp,center=center,extent=extent))
  best=None
  for group in sorted({r['group'] for r in candidates}):
   group_rows=sorted([v for v in candidates if v['group']==group],key=lambda r:r['frame_number'])
   count=min(4,len(group_rows))
   for start in range(len(group_rows)-count+1):
    window=group_rows[start:start+count]
    movement=max([np.linalg.norm(a['center']-b['center']) for a,b in zip(window,window[1:])]+[0])
    change=max([float(np.max(abs(np.log((a['extent']+1e-6)/(b['extent']+1e-6))))) for a,b in zip(window,window[1:])]+[0])
    score=float(np.mean([np.log1p(v['sharpness']) for v in window])-3*movement-2*change)
    if best is None or score>best[0]:best=(score,window,movement,change)
  if best is None:continue
  _,window,movement,change=best;representatives=[]
  for entry in window:
   i=entry['index'];row=rows[i];crop,target=prepare_crop(imread(row['image']),imread(row['mask'],True));nodes=[]
   for n in graphs[i]['nodes']:
    fit=fit_circle(n.get('polygon',[]))
    if fit['accepted'] and .7*n['radius']<fit['radius']<1.8*n['radius']:
     x,y=fit['center'];nodes.append(dict(x=x,y=y,radius=fit['radius'],visibility=fit['confidence'],confidence=n['confidence']))
   if len(nodes)<6:continue
   im,labels,fg,transform=letterbox(crop,nodes,target)
   role='train_weak' if identity in trainids else 'manual_review_pending'
   stem=f'{identity}_{i}';pic(OUT/(stem+'.jpg'),im);cv2.imencode('.png',fg*255)[1].tofile(str(OUT/(stem+'_mask.png')))
   rec=dict(domain='real',identity=identity,role=role,index=i,image=str(OUT/(stem+'.jpg')),mask=str(OUT/(stem+'_mask.png')),
            source_image=row['image'],source_mask=row['mask'],source_image_sha256=row['image_sha256'],source_mask_sha256=row['mask_sha256'],
            nodes=labels,transform=transform,sharpness=entry['sharpness'],window_max_normalized_center_motion=float(movement),window_max_log_extent_change=float(change),
            label_source='automatic circle fit to existing detector polygon; weak and possibly incomplete',capture_stability='screened sampled-frame window, not proof of full raw-video stability')
   selected.append(rec)
   overlay=im.copy()
   for n in labels:cv2.circle(overlay,(round(n['x']),round(n['y'])),round(n['radius']),(40,230,70),1)
   representatives.append(overlay)
  if representatives:
   while len(representatives)<4:representatives.append(np.full_like(representatives[0],245))
   title=np.full((25,1024,3),245,np.uint8);cv2.putText(title,identity+' / '+role+' / sampled motion '+str(round(movement,3)),(6,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,50,30),1)
   tiles.append(np.vstack([title,np.hstack(representatives[:4])]))
  print(identity,'selected',len(representatives),'motion',round(movement,3),flush=True)
 for j in range(0,len(tiles),5):pic(qa/f'real_selection_{j:02d}.jpg',np.vstack(tiles[j:j+5]))
 synthetic_graphs=read(ROOT.parent/'structure_training_20261014/synthetic_graphs.json');synth=[]
 for n in range(100):
  folder=ROOT.parent/'fingerprint_mix_20261014/synthetic'/f'SYN_{n:03d}';records=read(folder/'records.json')['rows']
  views=list(range(0,24,3)) if n<80 else [0,6,12,18]
  for v in views:
   src=records[v];g=synthetic_graphs[n*24+v];nodes=[]
   for a in g['nodes']:
    x,y=a['true_projected_center'];nodes.append(dict(x=x,y=y,radius=a['true_projected_radius'],visibility=min(1,a['visible_fraction']),berry_id=a['berry_id']))
   synth.append(dict(domain='synthetic',identity=src['identity'],role='train_gt' if n<80 else 'validation_gt' if n<90 else 'heldout_gt',
                     image=src['image'],mask=src['mask'],nodes=nodes,label_source='known 3D sphere projected center/radius; visibility-filtered',view=v))
 save(OUT/'real_manifest.json',selected);save(OUT/'synthetic_manifest.json',synth)
 save(OUT/'data_protocol.json',dict(train_real_ids=trainids,real_pending_ids=['FP2_09','FP3_02','GROW_03'],synthetic_train_ids=[f'SYN_{n:03d}' for n in range(80)],
      synthetic_validation_ids=[f'SYN_{n:03d}' for n in range(80,90)],synthetic_heldout_ids=[f'SYN_{n:03d}' for n in range(90,100)],
      real_count=len(selected),synthetic_count=len(synth),manual_annotations_required=True,real_accuracy_not_measured_by_weak_labels=True))
 print('PREPARED',len(selected),'real',len(synth),'synthetic',flush=True)

if __name__=='__main__':main()
