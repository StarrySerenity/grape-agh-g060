"""Known berry ID depth buffer: visible centroids separate from true centers."""
import math
import numpy as np

def project_instances(bunch,yaw,pitch,size=288,visible_mask=None):
    models=bunch['models'];centers=np.array([m['center'] for m in models]);r=np.array([m['radius'] for m in models]);origin=centers.mean(0)
    extent=np.max(np.linalg.norm(centers-origin,axis=1)+r);factor=size*.9/(2*extent)
    cy,sy=math.cos(yaw),math.sin(yaw);cp,sp=math.cos(pitch),math.sin(pitch)
    rot=np.array([[cy,0,sy],[sp*sy,cp,-sp*cy],[-cp*sy,sp,cp*cy]])
    coords=(centers-origin)@rot.T;projected=np.c_[size/2+coords[:,0]*factor,size/2-coords[:,1]*factor]
    depth=np.full((size,size),-1e9);labels=np.zeros((size,size),np.int32)
    for index,(c,xy,radius) in enumerate(zip(coords,projected,r)):
        x,y=xy;rad=radius*factor;x0=max(0,int(x-rad-1));x1=min(size,int(x+rad+2));y0=max(0,int(y-rad-1));y1=min(size,int(y+rad+2))
        yy,xx=np.mgrid[y0:y1,x0:x1];d=((xx+.5-x)**2+(yy+.5-y)**2)/(rad*rad)
        surface=c[2]+radius*np.sqrt(np.maximum(0,1-d));visible=(d<=1)&(surface>depth[y0:y1,x0:x1])
        labels[y0:y1,x0:x1][visible]=index+1;depth[y0:y1,x0:x1][visible]=surface[visible]
    if visible_mask is not None:
        visible_mask=np.asarray(visible_mask).squeeze()
        if visible_mask.shape!=labels.shape:raise ValueError('Visible mask dimensions differ')
        labels[visible_mask==0]=0
    nodes=[]
    for index,(model,xy,radius) in enumerate(zip(models,projected,r)):
        yy,xx=np.where(labels==index+1);fraction=len(xx)/max(np.pi*(radius*factor)**2,1)
        if fraction<.25:continue
        nodes.append(dict(id=f'B{len(nodes)+1:02d}',berry_id=model['id'],x=float((xx+.5).mean()),y=float((yy+.5).mean()),
            radius=float(np.sqrt(len(xx)/np.pi)),true_projected_center=xy.tolist(),true_projected_radius=float(radius*factor),
            visible_fraction=float(fraction),center_type='visible pixel centroid, not true sphere center'))
    nodes.sort(key=lambda n:(n['y'],n['x']))
    for i,n in enumerate(nodes):n['id']=f'B{i+1:02d}'
    return dict(nodes=nodes,labels=labels,projected_centers=projected.tolist(),factor=factor)
