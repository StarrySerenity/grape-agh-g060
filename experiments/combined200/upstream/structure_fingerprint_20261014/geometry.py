"""Geometry-only partial berry correspondences. No RGB, texture or classifier."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

def arrays(nodes):
    if not isinstance(nodes,list) or len(nodes)>160:raise ValueError('Provide at most 160 berry nodes')
    try:p=np.array([[n['x'],n['y']] for n in nodes],float).reshape(-1,2);r=np.array([n['radius'] for n in nodes],float)
    except (KeyError,TypeError,ValueError):raise ValueError('Each berry needs x, y and radius')
    if not np.isfinite(p).all() or not np.isfinite(r).all() or (r<=0).any():raise ValueError('Invalid berry coordinates or radius')
    return p,r

def normalized(nodes):
    p,r=arrays(nodes);unit=max(float(np.median(r)),1e-9) if len(r) else 1.
    origin=np.median(p,axis=0) if len(p) else np.zeros(2)
    return (p-origin)/unit,r/unit,origin,unit

def fingerprint(nodes):
    p,r,_,_=normalized(nodes)
    if len(p)<4:raise ValueError('At least four visible berries are needed')
    d=np.linalg.norm(p[:,None]-p[None,:],axis=2);np.fill_diagonal(d,np.inf)
    neighbors=np.sort(d,axis=1)[:,:min(4,len(p)-1)]
    eigen=np.linalg.eigvalsh(np.cov(p.T));eigen=np.maximum(eigen,0)
    vector=np.r_[np.quantile(neighbors,[.1,.25,.5,.75,.9]),np.quantile(r,[.1,.25,.5,.75,.9]),np.sqrt(eigen),len(p)/50]
    edges=set()
    for i in range(len(p)):
        for j in np.argsort(d[i])[:min(3,len(p)-1)]:edges.add(tuple(sorted((i,int(j)))))
    return dict(version='geometry-v1',dimension=len(vector),vector=np.round(vector,7).tolist(),
                vector_role='auxiliary distribution summary, not sufficient for instance identity by itself',
                normalized_nodes=[dict(index=i,x=round(float(x),7),y=round(float(y),7),radius=round(float(radius),7))
                                  for i,((x,y),radius) in enumerate(zip(p,r))],
                neighbor_edges=[dict(a=i,b=j,distance=round(float(d[i,j]),7)) for i,j in sorted(edges)],
                inputs='relative centers/radii and neighbor geometry only; no image values',visible_berries=len(p))

def correspond(p,r,q,s,alpha,beta,tolerance=.65):
    z=p[:,0]+1j*p[:,1];trans=z*alpha+beta;t=np.c_[trans.real,trans.imag]
    d=np.linalg.norm(t[:,None]-q[None,:],axis=2)
    # Radius discrepancy discourages matching a tiny fragment to a whole berry.
    cost=d+.12*np.abs(r[:,None]*abs(alpha)-s[None,:])
    ii,jj=linear_sum_assignment(cost);keep=d[ii,jj]<tolerance
    return ii[keep],jj[keep],d[ii[keep],jj[keep]]

def fit_transform(p,q):
    a=p[:,0]+1j*p[:,1];b=q[:,0]+1j*q[:,1];ac=a-a.mean();bc=b-b.mean()
    alpha=np.sum(bc*np.conj(ac))/max(float(np.sum(abs(ac)**2)),1e-9)
    return alpha,b.mean()-alpha*a.mean()

def match_graphs(query,gallery):
    p,r,po,pu=normalized(query);q,s,qo,qu=normalized(gallery)
    empty=dict(status='insufficient',score=0.,coverage=0.,query_coverage=0.,gallery_coverage=0.,residual_radius=None,pairs=[],
               identity_confirmed=False,uses_color=False)
    if min(len(p),len(q))<6:return empty
    za=p[:,0]+1j*p[:,1];zb=q[:,0]+1j*q[:,1];tree=cKDTree(q)
    da=np.linalg.norm(p[:,None]-p[None,:],axis=2);db=np.linalg.norm(q[:,None]-q[None,:],axis=2)
    np.fill_diagonal(da,np.inf);np.fill_diagonal(db,np.inf)
    descriptors_a=np.sort(da,axis=1)[:,:3];descriptors_b=np.sort(db,axis=1)[:,:3]
    difference=np.linalg.norm(descriptors_a[:,None]-descriptors_b[None,:],axis=2)
    possible=np.argsort(difference,axis=1)[:,:3];near_a=np.argsort(da,axis=1)[:,:3];near_b=np.argsort(db,axis=1)[:,:4]
    hypotheses=[(1.+0j,0.+0j)];best=[]
    for i in range(len(p)):
        for j in possible[i]:
            for k in near_a[i]:
                if abs(za[k]-za[i])<.1:continue
                for l in near_b[j]:
                    alpha=(zb[l]-zb[j])/(za[k]-za[i])
                    if .65<=abs(alpha)<=1.55 and abs(np.angle(alpha))<=.85:
                        hypotheses.append((alpha,zb[j]-alpha*za[i]))
    for alpha,beta in hypotheses:
        z=za*alpha+beta;distance,indices=tree.query(np.c_[z.real,z.imag]);valid=distance<.65
        count=len(set(indices[valid].tolist()))
        fast=count/max(len(p),len(q))-.12*float(np.mean(distance[valid])) if count else 0
        best.append((fast,alpha,beta))
    best.sort(key=lambda h:-h[0]);out=None
    for _,alpha,beta in best[:12]:
        for _ in range(3):
            ii,jj,dist=correspond(p,r,q,s,alpha,beta)
            if len(ii)<6:break
            updated,b=fit_transform(p[ii],q[jj])
            if not .65<=abs(updated)<=1.55 or abs(np.angle(updated))>.85:break
            alpha,beta=updated,b
        ii,jj,dist=correspond(p,r,q,s,alpha,beta)
        if len(ii)<6:continue
        cq=len(ii)/len(p);cg=len(ii)/len(q);coverage=2*cq*cg/max(cq+cg,1e-9);residual=float(np.median(dist))
        spread=float(np.linalg.norm(np.ptp(q[jj],axis=0))/max(np.linalg.norm(np.ptp(q,axis=0)),1e-9))
        score=coverage*np.exp(-.65*residual)*min(1,spread/.75)
        if out is None or score>out['score']:
            out=dict(status='candidate',score=float(score),coverage=float(coverage),query_coverage=cq,gallery_coverage=cg,
                     residual_radius=residual,pairs=[dict(query=int(a),gallery=int(b),residual_radius=float(d)) for a,b,d in zip(ii,jj,dist)],
                     transform=dict(scale=float(abs(alpha)*qu/pu),angle=float(np.angle(alpha)),
                         query_origin=po.tolist(),gallery_origin=qo.tolist(),query_unit=pu,gallery_unit=qu,
                         alpha=[float(alpha.real),float(alpha.imag)],beta=[float(beta.real),float(beta.imag)]),
                     identity_confirmed=False,uses_color=False)
    return out or empty
