import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,argparse,cv2,numpy as np
from pathlib import Path
from tracking import track_frames
from sfm import reconstruct,_project
from registration import register_clouds
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def picture(path,im):cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,88])[1].tofile(str(path))
def ply(path,points):
 with path.open('w',encoding='ascii') as f:
  f.write('ply\nformat ascii 1.0\nelement vertex '+str(len(points))+'\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
  for p in points:f.write(' '.join(map(str,p))+'\n')

def run(key):
 out=ROOT/'real'/key;data=read(out/'detections.json');frames=track_frames(data['frames']);save(out/'tracks.json',dict(**{k:v for k,v in data.items() if k!='frames'},frames=frames))
 counts={}
 for f in frames:
  for n in f['nodes']:counts[n['track_id']]=counts.get(n['track_id'],0)+1
 persistent={id for id,c in counts.items() if c>=10};tiles=[]
 for f in frames:
  im=cv2.imdecode(np.fromfile(f['image'],np.uint8),1)
  for n in f['nodes']:
   color=(40,220,90) if n['track_id'] in persistent else (0,140,245)
   xy=(round(n['x']),round(n['y']));cv2.circle(im,xy,round(n['radius']),color,1)
   cv2.putText(im,str(n['track_id']),xy,cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
  picture(out/f'tracks_{f["frame"]:03d}.jpg',im)
  if f['frame']%4==0:tiles.append(cv2.resize(im,(180,320)))
 while len(tiles)%3:tiles.append(np.full_like(tiles[0],245))
 picture(out/'track_review.jpg',np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
 print(key,'tracked',len(counts),'persistent >=10',len(persistent),flush=True)
 h,w=frames[0]['height'],frames[0]['width'];all_results=[]
 # Existing camera intrinsics are a useful prior; camera poses/landmarks remain
 # reconstructed independently here. Do not infer unknown focal from reprojection
 # error alone: erroneous focal can still fit a sparse short-baseline sequence.
 if key.startswith('grow'):
  prior_path=Path('E:/PyCharm/Project/grape_berry_2d/.worktrees/gbisc-baseline/outputs/colmap_joint_20261005/jobs/g1_02_joint/attempt_0/sparse/0/cameras.txt')
  camera_id=1 if key.startswith('grow_A') else 2
  line=next(v for v in prior_path.read_text().splitlines() if v.startswith(str(camera_id)+' '));values=line.split()
  original_w,original_h=int(values[2]),int(values[3]);focal=float(values[4])*w/original_w
  Kprior=np.array([[focal,0,float(values[5])*w/original_w],[0,focal,float(values[6])*h/original_h],[0,0,1.]])
  focal_trials=[(focal/max(h,w),Kprior)]
  prior=dict(source=str(prior_path),camera_id=camera_id,kind='existing COLMAP-estimated intrinsics; NOT physical camera calibration',radial_coefficient=float(values[7]),radial_distortion_ignored_in_center_pilot=True)
 elif key.startswith('fp203'):
  prior_path=Path('E:/PyCharm/Project/grape_berry_2d/.worktrees/gbisc-baseline/outputs/berry_position_batch_20261006/alignment/g2_03_v1/transformed_views.json')
  raw=read(prior_path)['views'][0]['camera'];params=raw['params'];focal=params[0]*w/raw['w']
  Kprior=np.array([[focal,0,params[1]*w/raw['w']],[0,focal,params[2]*h/raw['h']],[0,0,1.]])
  focal_trials=[(focal/max(h,w),Kprior)]
  prior=dict(source=str(prior_path),kind='old model intrinsics only; uncertain pose alignment not imported',radial_coefficient=params[3],radial_distortion_ignored_in_center_pilot=True)
 else:
  focal_trials=[(ratio,np.array([[max(h,w)*ratio,0,w/2],[0,max(h,w)*ratio,h/2],[0,0,1.]])) for ratio in [.55,.70,.85]]
  prior=dict(kind='uncalibrated generic focal assumptions')
 for focal_ratio,K in focal_trials:
  result=reconstruct(frames,K,max_candidates=6)
  result['focal_long_side_ratio']=focal_ratio;result['intrinsics_calibrated']=False
  result['intrinsics_prior']=prior
  save(out/f'reconstruction_focal_{focal_ratio:.2f}.json',result)
  all_results.append(result)
  print(key,'focal',focal_ratio,'status',result['status'],'points',len(result['points']),'error',result.get('mean_error_radius'),flush=True)
 valid=[r for r in all_results if r['status']=='reconstructed']
 result=min(valid,key=lambda r:r['mean_error_radius']) if valid else all_results[0]
 result.update(clip=key,identity=data['identity'],source_video=data['source_video'],
               camera_assumption_trials=[{k:r[k] for k in ['focal_long_side_ratio','status']}|dict(points=len(r['points']),error=r.get('mean_error_radius')) for r in all_results],
               track_count=len(counts),persistent_track_count=len(persistent),quality_manually_verified=False)
 save(out/'reconstruction.json',result)
 if result['status']!='reconstructed':return result
 ply(out/'berry_centers.ply',result['points'])
 points=np.array(result['points']);ids=result['track_ids'];K=np.array(result['K']);reproject_tiles=[]
 for camera in result['cameras']:
  f=frames[camera['frame']];im=cv2.imdecode(np.fromfile(f['image'],np.uint8),1);uv,depth=_project(points,np.array(camera['rotation']),np.array(camera['translation']),K)
  nodes={n['track_id']:n for n in f['nodes']}
  for i,id in enumerate(ids):
   if id not in nodes or depth[i]<=0:continue
   n=nodes[id];a=(round(n['x']),round(n['y']));b=tuple(np.rint(uv[i]).astype(int))
   cv2.line(im,a,b,(30,180,230),1);cv2.circle(im,a,4,(245,120,30),-1);cv2.circle(im,b,2,(30,230,60),-1)
   cv2.putText(im,str(id),a,cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
  picture(out/f'reproject_{f["frame"]:03d}.jpg',im)
  if f['frame']%4==0:reproject_tiles.append(cv2.resize(im,(180,320)))
 if reproject_tiles:
  while len(reproject_tiles)%3:reproject_tiles.append(np.full_like(reproject_tiles[0],245))
  picture(out/'reprojection_review.jpg',np.vstack([np.hstack(reproject_tiles[i:i+3]) for i in range(0,len(reproject_tiles),3)]))
 return result

def compare(a,b):
 x=read(ROOT/'real'/a/'reconstruction.json');y=read(ROOT/'real'/b/'reconstruction.json')
 if x['status']!='reconstructed' or y['status']!='reconstructed':result=dict(status='reconstruction_failed',matched=0,identity_confirmed=False)
 else:result=register_clouds(x['points'],y['points'])
 result.update(source_clip=a,target_clip=b,known_same_identity=x['identity']==y['identity'],
               different_source_videos=x['source_video']!=y['source_video'],quality_manually_verified=False)
 out=ROOT/'comparisons';out.mkdir(exist_ok=True);save(out/f'{a}__{b}.json',result)
 print('COMPARE',a,b,result['status'],result['matched'],result.get('same_cluster_candidate'),flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('key');p.add_argument('--compare');a=p.parse_args()
 compare(a.key,a.compare) if a.compare else run(a.key)
