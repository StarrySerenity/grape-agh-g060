"""Foreground, one explicitly selected clip at a time. Source videos stay read-only."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys,json,argparse,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'structure_fingerprint_20261014'))
from detection import detector
from circles import fit_circle
cv2.setNumThreads(2)

CONFIGS={
 'grow_A':dict(identity='GROW_02',video=1,start=.4,end=2.,boxes=[[.17,.26,.77,.85],[.05,.12,.70,.81]]),
 'grow_B':dict(identity='GROW_02',video=2,start=.5,end=3.7,boxes=[[.17,.16,.76,.77],[.14,.13,.73,.74]]),
 'grow_C':dict(identity='GROW_02',video=2,start=4.5,end=7.7,boxes=[[.13,.14,.74,.77],[.16,.17,.79,.85]]),
 'fp203_A':dict(identity='FP2_03',video=1,start=10.2,end=13.4,boxes=[[.15,.29,.64,.72],[.30,.28,.73,.72]])}

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def picture(path,im):cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,88])[1].tofile(str(path))
def reliable_size_nodes(nodes):
 if not nodes:return [],[]
 median=float(np.median([n['radius'] for n in nodes]));accepted=[];rejected=[]
 for n in sorted(nodes,key=lambda v:-v['confidence']):
  if n['radius']<.70*median:
   rejected.append(dict(visible_center=n['visible_centroid'],fit=n['circle_fit'],reason='small candidate / possible neighbor bunch or mask fragment'));continue
  if any(np.linalg.norm(np.array([n['x'],n['y']])-[v['x'],v['y']])<.75*max(n['radius'],v['radius']) for v in accepted):
   rejected.append(dict(visible_center=n['visible_centroid'],fit=n['circle_fit'],reason='near-duplicate center'));continue
  accepted.append(n)
 return accepted,rejected

def render_review(key,data):
 out=ROOT/'real'/key;qa=[]
 for f in data['frames']:
  im=cv2.imdecode(np.fromfile(f['image'],np.uint8),1)
  for n in f['nodes']:
   xy=(round(n['x']),round(n['y']));cv2.circle(im,xy,round(n['radius']),(30,230,90),2);cv2.circle(im,xy,3,(40,30,240),-1)
  picture(out/f'circles_{f["frame"]:03d}.jpg',im)
  if f['frame']%4==0:
   tile=cv2.resize(im,(180,320));label=np.full((24,180,3),245,np.uint8)
   cv2.putText(label,f'f{f["frame"]} {len(f["nodes"])} circles',(3,17),cv2.FONT_HERSHEY_SIMPLEX,.38,(30,50,30),1);qa.append(np.vstack([label,tile]))
 while len(qa)%3:qa.append(np.full_like(qa[0],245))
 picture(out/'circle_review.jpg',np.vstack([np.hstack(qa[i:i+3]) for i in range(0,len(qa),3)]))

def filter_saved(key):
 out=ROOT/'real'/key;data=json.loads((out/'detections.json').read_text(encoding='utf-8'))
 for f in data['frames']:
  f['unfiltered_nodes']=f.get('unfiltered_nodes',f['nodes'])
  f['nodes'],extra=reliable_size_nodes(f['unfiltered_nodes']);f['rejected_size']=extra
 data['size_filter']=dict(min_radius_to_frame_median=.70,deduplicate_distance_max_radius=.75,
                         caveat='pilot mature bunch consistency assumption; may exclude true small berries')
 save(out/'detections.json',data);render_review(key,data)
 print('FILTERED',key,'counts',[len(f['nodes']) for f in data['frames']],flush=True)
def extract(key):
 c=CONFIGS[key];inventory=json.loads((ROOT/'video_inventory.json').read_text(encoding='utf-8'))
 src=next(v for v in inventory if v['identity']==c['identity'] and v['video']==c['video'])
 out=ROOT/'real'/key;out.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(src['path']);frames=[];tiles=[]
 for i,t in enumerate(np.linspace(c['start'],c['end'],33)):
  cap.set(cv2.CAP_PROP_POS_MSEC,float(t*1000));ok,im=cap.read()
  if not ok:raise ValueError('Cannot decode selected source frame')
  h,w=im.shape[:2];factor=960/max(h,w);im=cv2.resize(im,(round(w*factor),round(h*factor)));h,w=im.shape[:2]
  phase=i/32*(len(c['boxes'])-1);left=min(int(phase),len(c['boxes'])-2);fraction=phase-left
  b=np.array(c['boxes'][left])*(1-fraction)+np.array(c['boxes'][left+1])*fraction
  box=np.rint(b*np.array([w,h,w,h])).astype(int).tolist()
  path=out/f'frame_{i:03d}.jpg';picture(path,im)
  frames.append(dict(frame=i,time_seconds=float(t),source_frame=round(t*src['fps']),image=str(path),width=w,height=h,box=box))
  if i%8==0:
   overlay=im.copy();cv2.rectangle(overlay,tuple(box[:2]),tuple(box[2:]),(0,220,240),3)
   tile=cv2.resize(overlay,(216,384));label=np.full((24,216,3),245,np.uint8)
   cv2.putText(label,f'{key} / {t:.2f}s',(4,17),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,50,30),1);tiles.append(np.vstack([label,tile]))
 cap.release();picture(out/'target_review.jpg',np.hstack(tiles))
 save(out/'frames.json',dict(clip=key,identity=c['identity'],source_video=src['path'],config=c,frames=frames,
      target_review='rectangular region proposed from video overview; must be visually verified',intrinsics_calibrated=False))
 print('EXTRACTED',key,33,'frames',out/'target_review.jpg',flush=True)

def detect(key):
 out=ROOT/'real'/key;data=json.loads((out/'frames.json').read_text(encoding='utf-8'));qa=[]
 net=detector()
 for frame in data['frames']:
  im=cv2.imdecode(np.fromfile(frame['image'],np.uint8),1);x0,y0,x1,y1=frame['box'];crop=im[y0:y1,x0:x1]
  result=net.predict(crop,device='cpu',half=False,imgsz=640,conf=.25,iou=.50,retina_masks=True,verbose=False)[0]
  nodes=[];rejected=[]
  if result.masks is not None:
   for binary,conf in zip(result.masks.data.cpu().numpy()>.5,result.boxes.conf.cpu().numpy()):
    m=binary.astype(np.uint8)
    if m.shape!=crop.shape[:2]:m=cv2.resize(m,(crop.shape[1],crop.shape[0]),interpolation=cv2.INTER_NEAREST)
    contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:continue
    contour=max(contours,key=cv2.contourArea).reshape(-1,2);area=cv2.contourArea(contour)
    if area<80:continue
    fit=fit_circle(contour);mom=cv2.moments(contour)
    visible=[mom['m10']/max(mom['m00'],1),mom['m01']/max(mom['m00'],1)]
    if not fit['accepted'] or not .65*np.sqrt(area/np.pi)<fit['radius']<2*np.sqrt(area/np.pi):
     rejected.append(dict(visible_center=[visible[0]+x0,visible[1]+y0],fit=fit,confidence=float(conf)));continue
    cx,cy=fit['center'];r=fit['radius']
    if not (0<=cx<crop.shape[1] and 0<=cy<crop.shape[0]) or r>min(crop.shape[:2])*.24:continue
    n=dict(x=float(cx+x0),y=float(cy+y0),radius=r,confidence=float(conf)*fit['confidence'],
           visible_centroid=[visible[0]+x0,visible[1]+y0],circle_fit=fit,
           polygon=(contour[np.linspace(0,len(contour)-1,min(len(contour),80),dtype=int)]+[x0,y0]).tolist())
    if any(np.linalg.norm(np.array([n['x'],n['y']])-[v['x'],v['y']])<.65*min(r,v['radius']) for v in nodes):continue
    nodes.append(n)
  frame.update(nodes=nodes,rejected=rejected)
  overlay=im.copy()
  for n in nodes:
   center=(round(n['x']),round(n['y']));cv2.circle(overlay,center,round(n['radius']),(30,230,90),2)
   cv2.circle(overlay,center,3,(40,30,240),-1)
   cv2.circle(overlay,tuple(map(round,n['visible_centroid'])),2,(230,120,20),-1)
  for n in rejected:cv2.circle(overlay,tuple(map(round,n['visible_center'])),3,(0,160,240),-1)
  picture(out/f'circles_{frame["frame"]:03d}.jpg',overlay)
  if frame['frame']%4==0:
   tile=cv2.resize(overlay,(180,320));label=np.full((24,180,3),245,np.uint8)
   cv2.putText(label,f'f{frame["frame"]} {len(nodes)} circles',(3,17),cv2.FONT_HERSHEY_SIMPLEX,.38,(30,50,30),1);qa.append(np.vstack([label,tile]))
  print(key,frame['frame'],'circles',len(nodes),'rejected',len(rejected),flush=True)
 while len(qa)%3:qa.append(np.full_like(qa[0],245))
 picture(out/'circle_review.jpg',np.vstack([np.hstack(qa[i:i+3]) for i in range(0,len(qa),3)]))
 data.update(detector='existing GBISC YOLOv8m instance masks on RGB + robust arc fitting; NOT trained CircleNet',
             circles_manually_verified=False,original_pixels_restored=True)
 save(out/'detections.json',data)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('key',choices=list(CONFIGS));p.add_argument('--operation',choices=['extract','detect','filter'],required=True);a=p.parse_args()
 {'extract':extract,'detect':detect,'filter':filter_saved}[a.operation](a.key)
