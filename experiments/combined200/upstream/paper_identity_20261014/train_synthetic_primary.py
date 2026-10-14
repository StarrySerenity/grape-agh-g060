"""Train only synthetic center/radius heads; geometry identity stays separate."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import copy,json,time,numpy as np,torch
from pathlib import Path
from circle_network import CircleDetector,loss
from train_circles import cache_records,batch,evaluate,save,read
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'synthetic_consumer';OUT.mkdir(exist_ok=True)

def main():
 torch.manual_seed(99237);np.random.seed(99237);torch.set_num_threads(2)
 graphs=read(ROOT.parent/'structure_training_20261014/synthetic_graphs.json');train=[];val=[]
 for i in range(90):
  src=ROOT.parent/'fingerprint_mix_20261014/synthetic'/f'SYN_{i:03d}';rows=read(src/'records.json')['rows']
  views=range(24) if i<80 else [0,6,12,18]
  for view in views:
   g=graphs[i*24+view];nodes=[]
   for n in g['nodes']:
    x,y=n['true_projected_center'];nodes.append(dict(x=x,y=y,radius=n['true_projected_radius'],visibility=min(1,n['visible_fraction'])))
   row=dict(domain='synthetic',identity=rows[view]['identity'],image=rows[view]['image'],mask=rows[view]['mask'],nodes=nodes,role='train_gt' if i<80 else 'validation_gt')
   (train if i<80 else val).append(row)
 net=CircleDetector(pretrained=True);cache=cache_records(net,train+val);traincache=cache[:len(train)];valcache=cache[len(train):]
 save(OUT/'detector_protocol.json',dict(training_images=len(train),training_identities=[f'SYN_{n:03d}' for n in range(80)],
      validation_images=len(val),validation_identities=[f'SYN_{n:03d}' for n in range(80,90)],
      geometry_holdout_identities=[f'SYN_{n:03d}' for n in range(90,100)],epochs=48,steps_per_epoch=32,batch_size=4,
      seed=99237,detector_threshold=.25,selection='synthetic validation F1 minus .1 normalized center error',
      real_training_images=0,gpu_used=False,cpu_threads=2,backbone='frozen local ImageNet ResNet18',
      architecture='CircleNet-style center heatmap, subpixel offset, radius and visibility; engineering adaptation',
      targets='known sphere projected centers, not visible-mask centroids',features_cached_in_ram_only=True))
 rng=np.random.default_rng(22719);optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
 history=[];best=-1;start=time.time()
 for epoch in range(1,49):
  net.train();values=[]
  for step in range(32):
   ids=rng.integers(0,len(traincache),4);features,target=batch(traincache,ids,rng.random(4)<.5)
   prediction=net.heads(features);value,_=loss(prediction,target,weak=False)
   optimizer.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],5);optimizer.step();values.append(float(value.detach()))
  if epoch in [1,2] or epoch%4==0:
   validation=evaluate(net,valcache,threshold=.25);score=validation['f1']-.1*(validation['median_center_error_radius'] or 0)
   row=dict(epoch=epoch,loss=float(np.mean(values)),validation=validation,selection_score=score,elapsed=round(time.time()-start,2));history.append(row);save(OUT/'detector_history.json',history)
   if score>best:best=score;torch.save(dict(model=net.state_dict(),epoch=epoch,selection=row,seed=99237,training_identities=80),OUT/'detector_best.pt')
   print('synthetic_only epoch',epoch,'loss',round(row['loss'],3),'F1',round(validation['f1'],4),'error',round(validation['median_center_error_radius'] or 0,3),'elapsed',round(time.time()-start),flush=True)
 save(OUT/'detector_summary.json',max(history,key=lambda r:r['selection_score']));print('SYNTHETIC PRIMARY DETECTOR COMPLETE',flush=True)

if __name__=='__main__':main()
