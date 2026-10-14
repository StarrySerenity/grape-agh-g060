"""CPU orthographic RGB and visible instance masks; illustrative materials."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import math
import cv2,numpy as np
cv2.setNumThreads(2)

def render(bunch,yaw,pitch,appearance_seed,size=288):
    rng=np.random.default_rng(appearance_seed);centers=np.array([m['center'] for m in bunch['models']]);r=np.array([m['radius'] for m in bunch['models']]);origin=centers.mean(0)
    ext=np.max(np.linalg.norm(centers-origin,axis=1)+r);factor=size*.90/(2*ext)
    cy,sy=math.cos(yaw),math.sin(yaw);cp,sp=math.cos(pitch),math.sin(pitch)
    R=np.array([[cy,0,sy],[sp*sy,cp,-sp*cy],[-cp*sy,sp,cp*cy]])
    def project(point):
        q=R@(np.asarray(point)-origin);return np.array([size/2+q[0]*factor,size/2-q[1]*factor,q[2]])
    image=np.full((size,size,3),128.,np.float32);depth=np.full((size,size),-1e9,np.float32);mask=np.zeros((size,size),np.uint8)
    light=np.array([rng.uniform(-.6,.6),rng.uniform(.1,.7),rng.uniform(.6,1.)]);light/=np.linalg.norm(light)
    base=np.array([98,173,154]) if rng.random()<.5 else np.array([120,82,142]);base=base*rng.uniform(.85,1.15)
    ambient=rng.uniform(.42,.58)
    def splat(point,radius,color):
        x,y,z=project(point);rad=radius*factor;x0=max(0,int(x-rad-1));x1=min(size,int(x+rad+2));y0=max(0,int(y-rad-1));y1=min(size,int(y+rad+2))
        if x0>=x1 or y0>=y1:return
        yy,xx=np.mgrid[y0:y1,x0:x1];nx=(xx+.5-x)/factor/radius;ny=-(yy+.5-y)/factor/radius;distance=nx*nx+ny*ny
        nz=np.sqrt(np.maximum(0,1-distance));surface=z+nz*radius;visible=(distance<=1)&(surface>depth[y0:y1,x0:x1])
        if not visible.any():return
        diffuse=np.maximum(0,nx*light[0]+ny*light[1]+nz*light[2]);intensity=ambient+(1-ambient)*diffuse
        reflection=np.maximum(0,(nz+diffuse)/2)**24
        pixels=np.clip(color[None,None,:]*intensity[...,None]+reflection[...,None]*18,0,255)
        view=image[y0:y1,x0:x1];view[visible]=pixels[visible];depth[y0:y1,x0:x1][visible]=surface[visible];mask[y0:y1,x0:x1][visible]=255
    for stem in bunch['stems']:
        for a,b in zip(stem['points'],stem['points'][1:]):
            a,b=np.array(a),np.array(b);steps=max(2,int(np.linalg.norm(b-a)*factor/2))
            for t in np.linspace(0,1,steps):splat(a+t*(b-a),stem['radius'],np.array([53,98,80]))
    for berry in bunch['models']:
        splat(berry['center'],berry['radius'],base*rng.uniform(.94,1.06))
    # Leaf-shaped foreground occlusion changes observation only, never identity geometry.
    if rng.random()<.22:
        leaf=np.zeros_like(mask);fg=np.argwhere(mask>0)
        if len(fg):
            cy,cx=fg[int(rng.integers(len(fg)))];axes=(int(rng.integers(13,27)),int(rng.integers(8,17)))
            cv2.ellipse(leaf,(int(cx),int(cy)),axes,float(rng.uniform(0,180)),0,360,255,-1)
            if ((leaf>0)&(mask>0)).sum()/(mask>0).sum()<.18:mask[leaf>0]=0
    image=np.clip(image+rng.normal(0,1.1,image.shape),0,255).astype(np.uint8)
    if rng.random()<.25:image=cv2.GaussianBlur(image,(3,3),float(rng.uniform(.25,.65)))
    image[mask==0]=128
    return image,mask
