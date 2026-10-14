"""Continue the synthetic detector on balanced old/new shapes; foreground CPU."""
from data import *
import time,torch,gc
from circle_network import CircleDetector,loss
from train_circles import prepare,normalize,batch,evaluate
torch.set_num_threads(2)

def cache_light(net,records):
 result=[];start=time.time()
 for i,r in enumerate(records):
  im,nodes,target=prepare(r)
  with torch.no_grad():features=tuple(f[0].half().contiguous() for f in net.features(normalize(im)[None]))
  result.append(dict(record=r,nodes=nodes,target=target,features=features))
  if (i+1)%240==0:print('FEATURES',i+1,'/',len(records),'elapsed',round(time.time()-start),flush=True)
 return result

def main():
 protected_snapshot();available=memory_available_gib();print('AVAILABLE_RAM_GiB',round(available,2),flush=True)
 if available<5.8:raise RuntimeError('Need 5.8 GiB free RAM for in-memory features; no disk feature cache will be created')
 if (OUT/'detector_best.pt').exists():raise RuntimeError('Versioned checkpoint exists; do not overwrite trained model')
 torch.manual_seed(99237);np.random.seed(99237);splits=split_ids()
 train=load_image_records(splits['training'],range(24));val=load_image_records(splits['development'],[0,6,12,18])
 net=CircleDetector(pretrained=False);old=LEGACY/'synthetic_consumer/detector_best.pt';state=torch.load(old,map_location='cpu',weights_only=False);net.load_state_dict(state['model'])
 oldids=set(splits['training'][:80]);old_indices=np.array([i for i,r in enumerate(train) if r['identity'] in oldids]);new_indices=np.array([i for i,r in enumerate(train) if r['identity'] not in oldids])
 protocol=dict(total_dataset_identities=200,training_identities=splits['training'],training_identity_count=160,training_images=len(train),
    validation_identities=splits['development'],validation_identity_count=20,validation_images=len(val),
    previous_test_regression=splits['previous_test_regression'],fresh_evaluation=splits['fresh_evaluation'],
    epochs=48,steps_per_epoch=64,batch_size=4,old_images_per_batch=2,new_images_per_batch=2,
    learning_rate=.00035,optimizer='AdamW',weight_decay=.0001,seed=99237,real_training_images=0,cpu_threads=2,gpu_used=False,
    initialization='previous 80-bunch synthetic detector; balanced mixed-shape continuation, not a causal data-only ablation',
    initialization_sha256=sha(old),backbone='frozen local ImageNet ResNet18',architecture='CircleNet-style heatmap, center offsets, sphere radius, visibility',
    selection='mean(old-domain F1,new-domain F1) minus .1 mean normalized center error; development only',
    detector_threshold=.25,targets='projected full sphere centers, visibility >=25%; not visible-mask centroids',
    features_cached_in_ram_only=True,original_weights_overwritten=False)
 save(OUT/'detector_protocol.json',protocol);save(OUT/'training_records.json',train);save(OUT/'validation_records.json',val)
 cached=cache_light(net,train+val);traincache=cached[:len(train)];valcache=cached[len(train):]
 oldval=[r for r in valcache if int(r['record']['identity'].split('_')[1])<100];newval=[r for r in valcache if int(r['record']['identity'].split('_')[1])>=100]
 rng=np.random.default_rng(22719);optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.00035,weight_decay=.0001)
 best=-1;history=[];start=time.time()
 for epoch in range(1,49):
  net.train();losses=[]
  for step in range(64):
   indices=np.r_[rng.choice(old_indices,2),rng.choice(new_indices,2)];features,target=batch(traincache,indices,rng.random(4)<.5)
   value,_=loss(net.heads(features),target,weak=False);optimizer.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],5);optimizer.step();losses.append(float(value.detach()))
  if epoch in [1,2] or epoch%4==0:
   ov=evaluate(net,oldval,threshold=.25);nv=evaluate(net,newval,threshold=.25);validation=evaluate(net,valcache,threshold=.25)
   score=(ov['f1']+nv['f1'])/2-.05*((ov['median_center_error_radius'] or 0)+(nv['median_center_error_radius'] or 0))
   row=dict(epoch=epoch,loss=float(np.mean(losses)),validation=validation,old_shape_validation=ov,new_shape_validation=nv,selection_score=score,elapsed_seconds=round(time.time()-start,2));history.append(row)
   save(OUT/'detector_history.json',history)
   if score>best:
    best=score;torch.save(dict(model=net.state_dict(),epoch=epoch,selection=row,training_identities=160,total_dataset_identities=200,seed=99237),OUT/'detector_best.pt')
   print('TRAIN',epoch,'loss',round(row['loss'],3),'oldF1',round(ov['f1'],4),'newF1',round(nv['f1'],4),'elapsed',round(time.time()-start),flush=True)
 save(OUT/'detector_summary.json',max(history,key=lambda r:r['selection_score']));del cached,traincache,valcache;gc.collect()
 print('TRAINING_200_FINISHED',flush=True)

if __name__=='__main__':main()
