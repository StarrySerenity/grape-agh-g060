"""Use trained circle heads on selected raw-video clips, keeping baseline outputs."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import argparse,json,copy,cv2,numpy as np,torch
from pathlib import Path
from circle_network import CircleDetector,decode
from train_circles import normalize,save
from prepare_circle_data import letterbox
from prepare_real import render_review
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('source_clip');p.add_argument('--model',choices=['real_only','mixed'],default='mixed');a=p.parse_args()
 src=ROOT/'real'/a.source_clip;data=json.loads((src/'frames.json').read_text(encoding='utf-8'))
 state=torch.load(ROOT/'circle_training'/a.model/'best.pt',map_location='cpu',weights_only=False)
 net=CircleDetector(pretrained=False);net.load_state_dict(state['model']);net.eval()
 key=a.source_clip+'_'+a.model;out=ROOT/'real'/key;out.mkdir(exist_ok=True)
 baseline=json.loads((src/'detections.json').read_text(encoding='utf-8'))['frames']
 masks=[]
 for f in data['frames']:
  im=cv2.imdecode(np.fromfile(f['image'],np.uint8),1);x0,y0,x1,y1=f['box'];crop=im[y0:y1,x0:x1]
  # Training uses isolated bunches. An arbitrary rectangular ROI still contains
  # leaves/neighbors; restore the same target-selection contract at inference.
  support=[]
  for n in baseline[f['frame']]['nodes']:
   for angle in np.linspace(0,2*np.pi,24,endpoint=False):
    support.append([n['x']-x0+1.15*n['radius']*np.cos(angle),n['y']-y0+1.15*n['radius']*np.sin(angle)])
  mask=np.zeros(crop.shape[:2],np.uint8)
  if len(support)>=3:
   hull=cv2.convexHull(np.rint(support).astype(np.int32));cv2.fillConvexPoly(mask,hull,255)
  isolated=np.where(mask[...,None]>0,crop,128)
  pic,__,___,transform=letterbox(isolated,[],mask)
  masks.append(dict(frame=f['frame'],polygon=(hull.reshape(-1,2)+[x0,y0]).tolist() if support else [],source='automatic expanded circle-support hull; target proposal, not manual instance segmentation'))
  with torch.no_grad():pred=net(normalize(pic)[None]);candidates=decode(pred,threshold=.30)
  nodes=[];s=transform['scale'];dx,dy=transform['offset']
  for n in candidates:
   x=(n['x']-dx)/s;y=(n['y']-dy)/s;r=n['radius']/s
   if not 0<=x<crop.shape[1] or not 0<=y<crop.shape[0]:continue
   if mask[int(y),int(x)]==0:continue
   nodes.append(dict(n,x=float(x+x0),y=float(y+y0),radius=float(r),visible_centroid=[float(x+x0),float(y+y0)],circle_fit=dict(accepted=True,center=[x,y],radius=r),model=a.model,epoch=state['epoch']))
  # Predeclared mature-berry consistency filter, same across both trained variants.
  if nodes:
   med=np.median([n['radius'] for n in nodes]);nodes=[n for n in nodes if .70*med<=n['radius']<=1.5*med]
  f.update(nodes=nodes,rejected=[])
  print(key,f['frame'],'centers',len(nodes),flush=True)
 data.update(clip=key,detector='locally trained frozen-ResNet18 circle heads, not original CircleNet',
             model=a.model,epoch=state['epoch'],detection_threshold=.30,original_pixels_restored=True,
             circles_manually_verified=False,target_mask_source='baseline circle-support convex hull proposal; uses baseline localization and is not standalone target segmentation',
             training_overlap='GROW_02 video2 frame0 was used for manual detector training; video matching is a reuse diagnostic, not heldout accuracy' if a.source_clip.startswith('grow') else 'FP2_03 frame20 manual detector training; not heldout identity accuracy')
 save(out/'frames.json',data);save(out/'detections.json',data)
 save(out/'target_masks.json',masks)
 # Current render_review expects source centroid and fit metadata, provided explicitly above.
 render_review(key,data)
 print('APPLIED',key,flush=True)

if __name__=='__main__':main()
