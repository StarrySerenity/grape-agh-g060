"""New query poses/appearances; disjoint geometry IDs for decision calibration/eval."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys,json,math,hashlib,time,cv2,numpy as np,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'synthetic_consumer';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT.parent/'structure_fingerprint_20261014'));sys.path.insert(0,str(ROOT.parent/'fingerprint_mix_20261014'))
from projection import project_instances
from render_synthetic import render
from circle_network import CircleDetector,decode
from prepare_circle_data import letterbox
from train_circles import normalize
from consumer_geometry import descriptor
cv2.setNumThreads(2);torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def model(i):return read(ROOT.parent/'fingerprint_mix_20261014/synthetic'/f'SYN_{i:03d}'/'model.json')

def main():
 registered=list(range(50,86))+list(range(90,95));models={i:model(i) for i in range(50,100)}
 gallery=[];start=time.time()
 for i in registered:
  for pitch_index,pitch in enumerate([-.24,0,.24]):
   for view in range(24):
    yaw=view*math.pi/12;projection=project_instances(models[i],yaw,pitch)
    nodes=[dict(x=n['true_projected_center'][0]*256/288,y=n['true_projected_center'][1]*256/288,
                radius=n['true_projected_radius']*256/288,visibility=n['visible_fraction'],berry_id=n['berry_id']) for n in projection['nodes'] if n['visible_fraction']>=.35]
    gallery.append(dict(key=f'SYN_{i:03d}__p{pitch_index}_v{view:02d}',identity=f'SYN_{i:03d}',yaw=yaw,pitch=pitch,
                        nodes=nodes,width=256,height=256,descriptor=descriptor(nodes).tolist(),geometry_hash=models[i]['geometry_hash']))
  print('GALLERY',i,72,'views','elapsed',round(time.time()-start),flush=True)
 save(OUT/'gallery.json',dict(identities=[f'SYN_{i:03d}' for i in registered],records=gallery,uses_color=False,
       templates='known projected sphere centers, visible area >=35%; RGB not used for matching',original_geometry_read_only=True))
 state=torch.load(OUT/'detector_best.pt',map_location='cpu',weights_only=False);net=CircleDetector(pretrained=False);net.load_state_dict(state['model']);net.eval()
 tasks=[('development',i,v) for i in range(80,90) for v in range(2)]+[('evaluation',i,v) for i in range(90,100) for v in range(4)]
 queries=[];images=OUT/'queries';images.mkdir(exist_ok=True);tiles=[]
 for split,i,view in tasks:
  seed=(724119 if split=='development' else 991791)+i*1931+view*117;rng=np.random.default_rng(seed)
  yaw=float(rng.uniform(0,2*np.pi));pitch=float(rng.uniform(-.25,.25));appearance=seed+300001
  im,mask=render(models[i],yaw,pitch,appearance)
  projection=project_instances(models[i],yaw,pitch,visible_mask=mask)
  roll=float(rng.uniform(-14,14));zoom=float(rng.uniform(.92,1.05));M=cv2.getRotationMatrix2D((144,144),roll,zoom)
  im=cv2.warpAffine(im,M,(288,288),borderValue=(128,128,128));mask=cv2.warpAffine(mask,M,(288,288),flags=cv2.INTER_NEAREST,borderValue=0)
  hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);hue=int(rng.integers(-25,26));hsv[...,0]=(hsv[...,0].astype(int)+hue)%180
  im=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);brightness=float(rng.uniform(.70,1.25));im=np.clip(im.astype(float)*brightness,0,255).astype(np.uint8);im[mask==0]=128
  nodes=[]
  for n in projection['nodes']:
   if n['visible_fraction']<.35:continue
   xy=M@np.r_[n['true_projected_center'],1.]
   if not (0<=xy[0]<288 and 0<=xy[1]<288):continue
   nodes.append(dict(x=float(xy[0]),y=float(xy[1]),radius=n['true_projected_radius']*zoom,visibility=n['visible_fraction'],berry_id=n['berry_id']))
  im,gt,fg,transform=letterbox(im,nodes,mask)
  for dst,src in zip(gt,nodes):dst['berry_id']=src['berry_id']
  with torch.no_grad():prediction=net(normalize(im)[None]);detected=decode(prediction,threshold=.25)
  detected=[n for n in detected if 0<=n['x']<256 and 0<=n['y']<256 and fg[int(n['y']),int(n['x'])]>0]
  for j,n in enumerate(detected):n['id']=f'D{j:02d}'
  key=f'{split}_{i:03d}_{view:02d}';path=images/(key+'.jpg');cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(path))
  maskpath=images/(key+'_mask.png');cv2.imencode('.png',fg*255)[1].tofile(str(maskpath))
  rec=dict(key=key,identity=f'SYN_{i:03d}',in_database=i in registered,split=split,image=path.relative_to(ROOT).as_posix(),mask=maskpath.relative_to(ROOT).as_posix(),
           nodes=detected,ground_truth_nodes=gt,yaw=yaw,pitch=pitch,roll_degrees=roll,zoom=zoom,brightness=brightness,hue_shift=hue,
           appearance_seed=appearance,geometry_hash=models[i]['geometry_hash'],image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
           labels_used_for_matching=False,query_generated_after_detector_training=True,novel_pose_and_appearance=True)
  queries.append(rec)
  overlay=im.copy()
  for n in detected:cv2.circle(overlay,(round(n['x']),round(n['y'])),round(n['radius']),(35,230,90),1)
  title=np.full((24,256,3),245,np.uint8);cv2.putText(title,key+f' / {len(detected)} centers',(3,17),cv2.FONT_HERSHEY_SIMPLEX,.35,(30,50,30),1)
  tiles.append(np.vstack([title,overlay]));print('QUERY',key,'detected',len(detected),flush=True)
 save(OUT/'query_manifest.json',queries)
 qa=OUT/'qa';qa.mkdir(exist_ok=True)
 for start in range(0,len(tiles),12):
  group=tiles[start:start+12]
  while len(group)%4:group.append(np.full_like(group[0],245))
  cv2.imencode('.jpg',np.vstack([np.hstack(group[j:j+4]) for j in range(0,len(group),4)]),[cv2.IMWRITE_JPEG_QUALITY,92])[1].tofile(str(qa/f'query_review_{start:02d}.jpg'))
 save(OUT/'identity_protocol.json',dict(registered_ids=[f'SYN_{i:03d}' for i in registered],registered_count=len(registered),views_per_identity=72,
       development_known_ids=[f'SYN_{i:03d}' for i in range(80,86)],development_unknown_ids=[f'SYN_{i:03d}' for i in range(86,90)],
       evaluation_known_ids=[f'SYN_{i:03d}' for i in range(90,95)],evaluation_unknown_ids=[f'SYN_{i:03d}' for i in range(95,100)],
       query_known_evaluation=20,query_unknown_evaluation=20,detector_training_geometry_ids='SYN000..079',
       gallery_distractor_ids='SYN050..079 also seen by detector, identity matching has no class head',
       decision_threshold_selected_on_development_only=True,query_nodes_from_trained_detector=True,gallery_nodes_from_simulation=True,
       perspective='orthographic renderer; perspective phone photos not yet validated',random_seed_protocol='724119/991791 + identity*1931 + view*117'))

if __name__=='__main__':main()
