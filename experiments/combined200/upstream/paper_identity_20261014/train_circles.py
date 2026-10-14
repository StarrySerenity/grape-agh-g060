"""Foreground CPU circle-head training, same real sampling for both comparisons."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,json,time,copy,cv2,numpy as np,torch
from pathlib import Path
from circle_network import CircleDetector,targets,loss,decode
from prepare_circle_data import letterbox
from registration import unique_pairs
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'circle_training'
torch.set_num_threads(2);cv2.setNumThreads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def imread(p):return cv2.imdecode(np.fromfile(str(p),np.uint8),1)
def normalize(im):
 a=im[...,::-1].copy().astype(np.float32)/255
 return torch.from_numpy(((a-[.485,.456,.406])/[.229,.224,.225]).transpose(2,0,1).astype(np.float32))

def load_records():
 real=[r for r in read(OUT/'real_manifest.json') if r['role']=='train_weak']
 manual=read(OUT/'manual_labels.json');synth=read(OUT/'synthetic_manifest.json')
 real+= [r for r in manual if r['role'].startswith('train')]
 return real,[r for r in synth if r['role']=='train_gt'],[r for r in synth if r['role']=='validation_gt'],[r for r in manual if r['role'].startswith('validation')],[r for r in synth if r['role']=='heldout_gt']

def prepare(record):
 im=imread(record['image']);nodes=copy.deepcopy(record['nodes'])
 if record['domain']=='synthetic':
  mask=cv2.imdecode(np.fromfile(record['mask'],np.uint8),0)
  # Rendering is 288 square: uniform resize preserves circle targets.
  im,nodes,fg,transform=letterbox(im,nodes,mask)
 else:
  fg=cv2.imdecode(np.fromfile(record['mask'],np.uint8),0)>0 if record.get('mask') else None
  if fg is None:
   fg=np.zeros((256,256),np.uint8)
   for n in nodes:cv2.circle(fg,(round(n['x']),round(n['y'])),round(n['radius']*1.35),1,-1)
  if im.shape[:2]!=(256,256):raise ValueError('Real annotation coordinates require 256 square image')
 t=targets(nodes);negative=(cv2.resize((fg==0).astype('uint8'),(64,64),interpolation=cv2.INTER_NEAREST)>0).astype(np.float32)[None]
 # A reliably annotated sphere interior should not emit multiple centers.
 # Leave other unannotated foreground unknown, rather than marking it all negative.
 for n in nodes:
  cv2.circle(negative[0],(int(n['x']/4),int(n['y']/4)),max(1,round(n['radius']*.85/4)),1,-1)
 # Unknown interior centers remain ignored. Small annotated-peak neighborhoods train peak localization.
 peak=t['heatmap'];negative=np.maximum(negative,(peak>.05).astype(np.float32))
 t['negative_valid']=negative
 return im,nodes,t

def cache_records(net,records):
 cache=[];start=time.time()
 for i,r in enumerate(records):
  im,nodes,t=prepare(r)
  with torch.no_grad():features=tuple(f[0].half().contiguous() for f in net.features(normalize(im)[None]))
  cache.append(dict(record=r,image=im,nodes=nodes,target=t,features=features))
  if (i+1)%80==0:print('features',i+1,'/',len(records),'elapsed',round(time.time()-start),flush=True)
 return cache

def batch(cache,indices,flip_flags=None):
 fs=[];ts=[]
 for k,i in enumerate(indices):
  row=cache[int(i)];f=row['features'];t={key:np.array(value,copy=True) for key,value in row['target'].items()}
  if flip_flags is not None and flip_flags[k]:
   f=tuple(torch.flip(x,[-1]) for x in f)
   nodes=[dict(n,x=255-n['x']) for n in row['nodes']];t=targets(nodes)
   t['negative_valid']=np.flip(row['target']['negative_valid'],axis=-1).copy()
  fs.append(f);ts.append(t)
 return tuple(torch.stack([f[k] for f in fs]) for k in range(3)),{key:torch.from_numpy(np.stack([v[key] for v in ts])) for key in ts[0]}

def evaluate(net,cache,threshold=.20,manual=False,return_predictions=False):
 totals=dict(annotated=0,predicted=0,matched=0);errors=[];radius_errors=[];predictions=[]
 net.eval()
 for row in cache:
  with torch.no_grad():pred=net.heads(tuple(f[None] for f in row['features']));nodes=decode(pred,threshold=threshold)
  target=row['nodes'];totals['annotated']+=len(target);totals['predicted']+=len(nodes)
  pairs=[]
  if nodes and target:
   p=np.array([[n['x'],n['y']] for n in nodes]);q=np.array([[n['x'],n['y']] for n in target]);r=np.array([n['radius'] for n in target])
   distance=np.linalg.norm(p[:,None]-q[None],axis=2)/r[None]
   from scipy.optimize import linear_sum_assignment
   n,m=distance.shape;cost=np.full((n,m+n),2.);cost[:,:m]=np.where(distance<.35,distance,1e6)
   ii,jj=linear_sum_assignment(cost)
   for i,j in zip(ii,jj):
    if j>=m or distance[i,j]>=.35:continue
    totals['matched']+=1;errors.append(float(distance[i,j]));radius_errors.append(abs(nodes[i]['radius']/target[j]['radius']-1))
    pairs.append(dict(prediction=int(i),annotation=int(j),error_radius=float(distance[i,j])))
  predictions.append(dict(image=row['record']['image'],identity=row['record']['identity'],nodes=nodes,annotations=target,pairs=pairs))
 recall=totals['matched']/max(totals['annotated'],1);precision=totals['matched']/max(totals['predicted'],1)
 out=dict(**totals,annotated_recall=recall,precision=None if manual else precision,
          f1=None if manual else 2*precision*recall/max(precision+recall,1e-9),
          median_center_error_radius=float(np.median(errors)) if errors else None,
          median_relative_radius_error=float(np.median(radius_errors)) if radius_errors else None,
          threshold=threshold,center_match_gate_radius=.35,
          caveat='manual visible subset; unmatched detections unverified; development validation, not independent test' if manual else 'known synthetic projected sphere-center labels, separate geometry identities')
 return (out,predictions) if return_predictions else out

def train(name,real,synth,val,manual,init,epochs=24,steps=24):
 folder=OUT/name;folder.mkdir(exist_ok=True);net=CircleDetector(pretrained=True);net.load_state_dict(init)
 optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
 rng=np.random.default_rng(61129);srng=np.random.default_rng(39027);history=[];best=-1;start=time.time()
 for epoch in range(1,epochs+1):
  net.train();losses=[]
  for step in range(steps):
   ids=rng.integers(0,len(real),2);flags=rng.random(2)<.5
   rf,rt=batch(real,ids,flags);rp=net.heads(rf);rl,_=loss(rp,rt,weak=True)
   total=rl
   if name=='mixed':
    ids=srng.integers(0,len(synth),2);flags=srng.random(2)<.5;sf,st=batch(synth,ids,flags)
    sp=net.heads(sf);sl,_=loss(sp,st,weak=False);total=rl+.6*sl
   optimizer.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],5);optimizer.step();losses.append(float(total.detach()))
  if epoch in [1,2] or epoch%3==0 or epoch==epochs:
   vm=evaluate(net,val);rm=evaluate(net,manual,manual=True)
   # Predeclared development selection: real visible center recall primarily, synthetic F1 auxiliary.
   error=rm['median_center_error_radius'] if rm['median_center_error_radius'] is not None else 1.
   radius=rm['median_relative_radius_error'] if rm['median_relative_radius_error'] is not None else 1.
   score=.75*rm['annotated_recall']+.25*vm['f1']-.10*error-.05*radius
   row=dict(epoch=epoch,loss=float(np.mean(losses)),synthetic_validation=vm,manual_development=rm,selection_score=score,elapsed_seconds=round(time.time()-start,2));history.append(row);save(folder/'history.json',history)
   if score>best:
    best=score;torch.save(dict(model=net.state_dict(),epoch=epoch,selection=row,architecture='frozen local ImageNet ResNet18 layer1/2/3 + trained circle heads',seed=28439),folder/'best.pt')
   print(name,'epoch',epoch,'loss',round(row['loss'],3),'real',rm['matched'],'/',rm['annotated'],'synF1',round(vm['f1'],3),'elapsed',round(time.time()-start),flush=True)
 return folder

def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=24);p.add_argument('--steps',type=int,default=24);a=p.parse_args()
 torch.manual_seed(28439);np.random.seed(28439)
 raw_real,raw_syn,raw_val,raw_manual,raw_test=load_records();net=CircleDetector(pretrained=True)
 allrows=raw_real+raw_syn+raw_val+raw_manual+raw_test;allcache=cache_records(net,allrows)
 cuts=np.cumsum([0,len(raw_real),len(raw_syn),len(raw_val),len(raw_manual),len(raw_test)])
 real,syn,val,manual,heldout=[allcache[cuts[i]:cuts[i+1]] for i in range(5)]
 init=copy.deepcopy(net.state_dict());save(OUT/'training_protocol.json',dict(epochs=a.epochs,steps_per_epoch=a.steps,real_batch=2,synthetic_batch=2,synthetic_weight=.6,
      real_training_images=len(real),real_training_ids=sorted({r['record']['identity'] for r in real}),synthetic_training_images=len(syn),synthetic_training_ids=80,
      synthetic_validation_images=len(val),synthetic_heldout_images=len(heldout),manual_development_images=len(manual),manual_development_nodes=sum(len(r['nodes']) for r in manual),
      selection='.75 manual visible-subset recall + .25 synthetic F1 - .10 median center error/radius - .05 median radius relative error',
      detector_threshold=.20,manual_labels_approximate=True,weak_real_negatives='outside target foreground plus 0.85-radius reliable annotated circle interiors; other unannotated foreground ignored',
      same_real_sampling_across_variants=True,backbone_frozen=True,cpu_threads=2,gpu_used=False,
      cache='features held in RAM only; no copied large tensors or video caches',original_circle_net=False,
      iteration=2,synthetic_heldout_previously_opened=True,heldout_metric_role='fixed synthetic regression after iteration1, not a fresh independent evaluation'))
 results={}
 for name in ['real_only','mixed']:
  folder=train(name,real,syn,val,manual,init,a.epochs,a.steps)
  state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(state['model'])
  sm=evaluate(net,heldout);rm,predictions=evaluate(net,manual,manual=True,return_predictions=True)
  results[name]=dict(epoch=state['epoch'],manual_development=rm,synthetic_heldout=sm)
  save(folder/'manual_predictions.json',predictions)
  tiles=[]
  for row in predictions:
   im=imread(row['image']);overlay=im.copy()
   for n in row['annotations']:cv2.circle(overlay,(round(n['x']),round(n['y'])),round(n['radius']),(30,210,60),1)
   for n in row['nodes']:
    cv2.circle(overlay,(round(n['x']),round(n['y'])),round(n['radius']),(0,130,245),1);cv2.circle(overlay,(round(n['x']),round(n['y'])),1,(0,0,240),-1)
   title=np.full((25,512,3),245,np.uint8);cv2.putText(title,f'{name} / {row["identity"]} / {len(row["pairs"])}/{len(row["annotations"])}',(3,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,50,30),1)
   tiles.append(np.vstack([title,np.hstack([im,overlay])]))
  cv2.imencode('.jpg',np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(folder/'manual_prediction_review.jpg'))
  save(OUT/'results.json',results)
 print('TRAINING FINISHED',json.dumps(results,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
