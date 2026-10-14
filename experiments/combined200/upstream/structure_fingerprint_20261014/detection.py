"""Explicit CPU berry segmentation restricted to a user-selected bunch."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys,hashlib
from functools import lru_cache
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent))
from fingerprint_2d import masked_crop
cv2.setNumThreads(2)
WEIGHTROOT=Path('E:/PyCharm/Project/grape_berry_2d/.worktrees/gbisc-baseline/outputs')

@lru_cache(maxsize=2)
def detector(name='gbisc_yolov8m'):
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=2:torch.set_num_interop_threads(2)
    from ultralytics import YOLO
    model=YOLO(WEIGHTROOT/name/'weights/best.pt')
    if model.names!={0:'grape_berry'}:raise ValueError('Expected single berry class')
    return model

def nodes_from_masks(masks,confidences,target):
    area=int((target>0).sum());out=[]
    for binary,confidence in zip(masks,confidences):
        binary=np.asarray(binary,dtype=np.uint8)
        if binary.shape!=target.shape:binary=cv2.resize(binary,(target.shape[1],target.shape[0]),interpolation=cv2.INTER_NEAREST)
        original=int((binary>0).sum());cut=((binary>0)&(target>0)).astype(np.uint8)
        n=int(cut.sum())
        if original==0 or n/original<.75 or n<max(35,area*.0015) or n>area*.20:continue
        contours,_=cv2.findContours(cut,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea);mom=cv2.moments(contour)
        if mom['m00']<20:continue
        x=mom['m10']/mom['m00'];y=mom['m01']/mom['m00'];radius=np.sqrt(mom['m00']/np.pi)
        poly=cv2.approxPolyDP(contour,.015*cv2.arcLength(contour,True),True).reshape(-1,2)
        # Preserve the visible-mask convention; no hidden sphere center inference.
        node=dict(x=float(x),y=float(y),radius=float(radius),confidence=float(confidence),polygon=poly.tolist(),
                  center_type='visible mask centroid estimate')
        if any(np.hypot(x-b['x'],y-b['y'])<.5*min(radius,b['radius']) for b in out):continue
        out.append(node)
    out.sort(key=lambda b:(b['y'],b['x']))
    for i,n in enumerate(out):n['id']=f'B{i+1:02d}'
    return out

def prepare_crop(image,mask):
    crop,target=masked_crop(image,mask);ratio=min(1,640/max(crop.shape[:2]))
    if ratio<1:
        crop=cv2.resize(crop,(max(1,round(crop.shape[1]*ratio)),max(1,round(crop.shape[0]*ratio))))
        target=cv2.resize(target,(crop.shape[1],crop.shape[0]),interpolation=cv2.INTER_NEAREST)
    return crop,target

def detector_input(crop,target):
    target=np.asarray(target).squeeze()
    if target.shape!=crop.shape[:2]:raise ValueError('Target mask dimensions differ')
    value=np.max(crop,axis=2).astype(float);fg=target>0
    if not fg.any():raise ValueError('Empty target')
    level=max(float(np.quantile(value[fg],.85)),1.)
    value=np.clip(np.rint(value*190/level),0,255).astype(np.uint8);value[~fg]=128
    return np.repeat(value[...,None],3,axis=2)

def detect_crop(crop,target,name='gbisc_yolov8m'):
    # Only value/shape enters segmentation; original RGB is used for display only.
    results=detector(name).predict(detector_input(crop,target),device='cpu',half=False,imgsz=640,conf=.18,iou=.50,retina_masks=True,verbose=False)
    result=results[0]
    if result.masks is None:return []
    return nodes_from_masks(result.masks.data.cpu().numpy()>.5,result.boxes.conf.cpu().numpy(),target)

def overlay(crop,nodes):
    image=crop.copy()
    for n in nodes:
        if n.get('polygon'):cv2.polylines(image,[np.rint(n['polygon']).astype(np.int32)],True,(55,240,100),1,cv2.LINE_AA)
        cv2.circle(image,(round(n['x']),round(n['y'])),max(1,round(n['radius'])),(180,210,255),1,cv2.LINE_AA)
        cv2.circle(image,(round(n['x']),round(n['y'])),2,(50,40,220),-1)
        cv2.putText(image,n['id'],(round(n['x'])-10,round(n['y'])),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1,cv2.LINE_AA)
    return image
