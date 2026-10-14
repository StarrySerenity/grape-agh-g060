"""Geometry-only single-photo vs multi-view template matching; no identity training."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

def normalized(nodes):
    if not isinstance(nodes,list) or len(nodes)>160:raise ValueError('Expected up to 160 circle nodes')
    p=np.array([[n['x'],n['y']] for n in nodes],float).reshape(-1,2);r=np.array([n['radius'] for n in nodes],float)
    if not np.isfinite(p).all() or not np.isfinite(r).all() or (r<=0).any():raise ValueError('Nonfinite or nonpositive circle')
    unit=float(np.median(r)) if len(r) else 1.;origin=np.median(p,axis=0) if len(p) else np.zeros(2)
    return (p-origin)/max(unit,1e-8),r/max(unit,1e-8),origin,unit

def descriptor(nodes):
    p,r,_,_=normalized(nodes)
    if len(p)<4:return np.zeros(19)
    d=np.linalg.norm(p[:,None]-p[None],axis=2);distances=d[np.triu_indices(len(p),1)]
    cov=np.linalg.eigvalsh(np.cov(p.T))
    return np.r_[np.quantile(distances,np.linspace(.05,.95,9))/8,
                 np.quantile(np.sort(d+np.eye(len(p))*1e8,axis=1)[:,:3],[.2,.5,.8])/3,
                 np.quantile(r,[.2,.5,.8]),np.sqrt(np.maximum(cov,0))/5,
                 len(p)/50,np.ptp(p,axis=0).max()/12]

def _fit(p,q):
    a=p[:,0]+1j*p[:,1];b=q[:,0]+1j*q[:,1];ac=a-a.mean();bc=b-b.mean()
    alpha=np.sum(bc*np.conj(ac))/max(float(np.sum(abs(ac)**2)),1e-9)
    return alpha,b.mean()-alpha*a.mean()

def _pairs(p,r,q,s,alpha,beta,gate=.65):
    z=(p[:,0]+1j*p[:,1])*alpha+beta;points=np.c_[z.real,z.imag]
    d=np.linalg.norm(points[:,None]-q[None],axis=2);radius=abs(r[:,None]*abs(alpha)-s[None])
    n,m=d.shape;cost=np.full((n,m+n),2*gate);cost[:,:m]=np.where(d<gate,d+.10*radius,1e6)
    ii,jj=linear_sum_assignment(cost);keep=jj<m;ii,jj=ii[keep],jj[keep]
    keep=d[ii,jj]<gate;ii,jj=ii[keep],jj[keep]
    return ii,jj,d[ii,jj],radius[ii,jj]

def match_view(query,gallery):
    p,r,po,pu=normalized(query);q,s,qo,qu=normalized(gallery)
    empty=dict(score=0.,pairs=[],identity_confirmed=False,coverage=0.,query_coverage=0.,gallery_coverage=0.,residual_radius=None)
    if min(len(p),len(q))<6:return empty
    za=p[:,0]+1j*p[:,1];zb=q[:,0]+1j*q[:,1]
    da=np.linalg.norm(p[:,None]-p[None],axis=2);db=np.linalg.norm(q[:,None]-q[None],axis=2)
    np.fill_diagonal(da,np.inf);np.fill_diagonal(db,np.inf)
    ad=np.sort(da,axis=1)[:,:3];bd=np.sort(db,axis=1)[:,:3]
    jchoices=np.argsort(np.linalg.norm(ad[:,None]-bd[None],axis=2),axis=1)[:,:3]
    kn=np.argsort(da,axis=1)[:,:3];ln=np.argsort(db,axis=1)[:,:3]
    alphas=[1+0j];betas=[0+0j]
    for i in range(len(p)):
        for j in jchoices[i]:
            for k in kn[i]:
                if abs(za[k]-za[i])<.1:continue
                for l in ln[j]:
                    a=(zb[l]-zb[j])/(za[k]-za[i])
                    if .65<abs(a)<1.55 and abs(np.angle(a))<.70:
                        alphas.append(a);betas.append(zb[j]-a*za[i])
    alphas=np.asarray(alphas);betas=np.asarray(betas);z=alphas[:,None]*za[None]+betas[:,None]
    distances,indices=cKDTree(q).query(np.c_[z.real.ravel(),z.imag.ravel()]);distances=distances.reshape(z.shape);indices=indices.reshape(z.shape)
    valid=distances<.65;sorted_indices=np.sort(np.where(valid,indices,len(q)),axis=1)
    unique=(sorted_indices[:,0]<len(q)).astype(int)+((sorted_indices[:,1:]!=sorted_indices[:,:-1])&(sorted_indices[:,1:]<len(q))).sum(1)
    residual=(distances*valid).sum(1)/np.maximum(valid.sum(1),1)
    rank=np.argsort(-(unique-.25*residual))[:8];best=None
    for idx in rank:
        a,b=alphas[idx],betas[idx]
        for _ in range(3):
            ii,jj,d,dr=_pairs(p,r,q,s,a,b)
            if len(ii)<6:break
            aa,bb=_fit(p[ii],q[jj])
            if not .65<abs(aa)<1.55 or abs(np.angle(aa))>.70:break
            a,b=aa,bb
        ii,jj,d,dr=_pairs(p,r,q,s,a,b)
        if len(ii)<6:continue
        cq=len(ii)/len(p);cg=len(ii)/len(q);coverage=2*cq*cg/(cq+cg)
        residual=float(np.median(d));radius_error=float(np.median(dr))
        spread=float(np.linalg.norm(np.ptp(q[jj],axis=0))/max(np.linalg.norm(np.ptp(q,axis=0)),1e-9))
        score=coverage*np.exp(-1.4*residual)*max(.5,1-.20*radius_error)*min(1,spread/.8)
        if best is None or score>best['score']:
            best=dict(score=float(score),pairs=[dict(query=int(i),gallery=int(j),residual_radius=float(v)) for i,j,v in zip(ii,jj,d)],
                      coverage=float(coverage),query_coverage=float(cq),gallery_coverage=float(cg),residual_radius=residual,
                      relative_radius_error=radius_error,identity_confirmed=False,
                      transform=dict(alpha=[float(a.real),float(a.imag)],beta=[float(b.real),float(b.imag)],query_origin=po.tolist(),gallery_origin=qo.tolist(),query_unit=pu,gallery_unit=qu),
                      uses_rgb_for_matching=False)
    return best or empty
