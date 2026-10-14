"""Equivalent all-view matcher: vectorize hypotheses, cache immutable layouts."""
from data import *
from functools import lru_cache
from scipy.spatial import cKDTree
from consumer_geometry import normalized,descriptor,_pairs,_fit

@lru_cache(maxsize=4096)
def _layout(key):
 nodes=[dict(x=x,y=y,radius=r) for x,y,r in key]
 p,r,origin,unit=normalized(nodes);z=p[:,0]+1j*p[:,1]
 d=np.linalg.norm(p[:,None]-p[None],axis=2);np.fill_diagonal(d,np.inf)
 nearest=np.argsort(d,axis=1)[:,:3];signature=np.sort(d,axis=1)[:,:3]
 return p,r,origin,unit,z,nearest,signature,cKDTree(p)

def layout(nodes):return _layout(tuple((float(n['x']),float(n['y']),float(n['radius'])) for n in nodes))

def match_view(query,gallery):
 p,r,po,pu,za,kn,ad,_=layout(query);q,s,qo,qu,zb,ln,bd,tree=layout(gallery)
 empty=dict(score=0.,pairs=[],identity_confirmed=False,coverage=0.,query_coverage=0.,gallery_coverage=0.,residual_radius=None)
 if min(len(p),len(q))<6:return empty
 choices=np.argsort(np.linalg.norm(ad[:,None]-bd[None],axis=2),axis=1)[:,:3]
 # C-order flattening preserves the legacy i / choice / k / l loop order.
 grid=np.indices((len(p),3,3,3)).reshape(4,-1);i=grid[0];j=choices[i,grid[1]];k=kn[i,grid[2]];l=ln[j,grid[3]]
 denominator=za[k]-za[i];a=np.divide(zb[l]-zb[j],denominator,out=np.zeros(len(i),complex),where=abs(denominator)>=.1)
 keep=(abs(denominator)>=.1)&(abs(a)>.65)&(abs(a)<1.55)&(abs(np.angle(a))<.70)
 alphas=np.r_[1+0j,a[keep]];betas=np.r_[0+0j,zb[j[keep]]-a[keep]*za[i[keep]]]
 z=alphas[:,None]*za[None]+betas[:,None]
 distances,indices=tree.query(np.c_[z.real.ravel(),z.imag.ravel()]);distances=distances.reshape(z.shape);indices=indices.reshape(z.shape)
 valid=distances<.65;sorted_indices=np.sort(np.where(valid,indices,len(q)),axis=1)
 unique=(sorted_indices[:,0]<len(q)).astype(int)+((sorted_indices[:,1:]!=sorted_indices[:,:-1])&(sorted_indices[:,1:]<len(q))).sum(1)
 residual=(distances*valid).sum(1)/np.maximum(valid.sum(1),1)
 rank=np.argsort(-(unique-.25*residual))[:8];best=None
 for idx in rank:
  a,b=alphas[idx],betas[idx];converged=False
  for _ in range(3):
   ii,jj,d,dr=_pairs(p,r,q,s,a,b)
   if len(ii)<6:break
   aa,bb=_fit(p[ii],q[jj])
   if not .65<abs(aa)<1.55 or abs(np.angle(aa))>.70:break
   if aa==a and bb==b:
    # Exact deterministic convergence: the current assignments/residuals
    # are already the final result; avoid identical repeated solves.
    converged=True;break
   a,b=aa,bb
  if not converged:ii,jj,d,dr=_pairs(p,r,q,s,a,b)
  if len(ii)<6:continue
  cq=len(ii)/len(p);cg=len(ii)/len(q);coverage=2*cq*cg/(cq+cg)
  residual=float(np.median(d));radius_error=float(np.median(dr));spread=float(np.linalg.norm(np.ptp(q[jj],axis=0))/max(np.linalg.norm(np.ptp(q,axis=0)),1e-9))
  score=coverage*np.exp(-1.4*residual)*max(.5,1-.20*radius_error)*min(1,spread/.8)
  if best is None or score>best['score']:
   best=dict(score=float(score),pairs=[dict(query=int(i),gallery=int(j),residual_radius=float(v)) for i,j,v in zip(ii,jj,d)],coverage=float(coverage),query_coverage=float(cq),gallery_coverage=float(cg),
    residual_radius=residual,relative_radius_error=radius_error,identity_confirmed=False,uses_rgb_for_matching=False,
    transform=dict(alpha=[float(a.real),float(a.imag)],beta=[float(b.real),float(b.imag)],query_origin=po.tolist(),gallery_origin=qo.tolist(),query_unit=pu,gallery_unit=qu))
 return best or empty
