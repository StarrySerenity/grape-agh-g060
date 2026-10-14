"""Partial similarity registration with unique matches and density-scaled gates.

Paper adaptations: triangle seeds augment canonical yaw seeds; no learned acceptor.
"""
import itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

def similarity(p,q):
    p=np.asarray(p,float);q=np.asarray(q,float);pc=p-p.mean(0);qc=q-q.mean(0)
    U,sv,Vt=np.linalg.svd(qc.T@pc)
    fix=np.eye(p.shape[1]);fix[-1,-1]=np.sign(np.linalg.det(U@Vt))
    R=U@fix@Vt;s=float(np.sum(sv*np.diag(fix))/max(np.sum(pc**2),1e-12))
    return s,R,q.mean(0)-s*(R@p.mean(0))

def spacing(p):
    k=min(6,len(p));d=cKDTree(p).query(p,k=k)[0][:,1:]
    return max(float(d.mean()),1e-9)

def unique_pairs(p,q,gate):
    distance=np.linalg.norm(p[:,None]-q[None,:],axis=2)
    # Explicit unmatched dummies maximize valid match count before minimizing error.
    n,m=distance.shape
    cost=np.full((n,m+n),2.*gate)
    cost[:,:m]=np.where(distance<=gate,distance,1e6*gate)
    ii,jj=linear_sum_assignment(cost)
    keep=(jj<m)
    ii=ii[keep];jj=jj[keep];valid=distance[ii,jj]<=gate
    return ii[valid],jj[valid],distance[ii[valid],jj[valid]]

def _triangles(p):
    triples=np.array(list(itertools.combinations(range(len(p)),3)),int)
    t=p[triples]
    edges=np.c_[np.linalg.norm(t[:,1]-t[:,2],axis=1),np.linalg.norm(t[:,0]-t[:,2],axis=1),np.linalg.norm(t[:,0]-t[:,1],axis=1)]
    order=np.argsort(edges,axis=1);edges=np.take_along_axis(edges,order,axis=1)
    good=(edges[:,0]>.25)&(edges[:,0]+edges[:,1]>edges[:,2]*1.03)
    return triples[good],order[good],edges[good,:2]/edges[good,2,None]

def register_clouds(source,target,min_matches=11):
    p=np.asarray(source,float);q=np.asarray(target,float)
    if p.ndim!=2 or q.ndim!=2 or p.shape[1]!=3 or q.shape[1]!=3 or not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('Expected finite N x 3 points')
    if max(len(p),len(q))>160:raise ValueError('At most 160 landmarks per cloud')
    if min(len(p),len(q))<6:return dict(status='insufficient',matched=0,pairs=[],same_cluster_candidate=False)
    up,uq=spacing(p),spacing(q);po=p.mean(0);qo=q.mean(0);a=(p-po)/up;b=(q-qo)/uq
    _,_,Vp=np.linalg.svd(a,full_matrices=False);_,_,Vq=np.linalg.svd(b,full_matrices=False)
    seeds=[]
    for signs in itertools.product([-1,1],repeat=3):
        S=np.diag(signs)
        if np.linalg.det(S)<0:continue
        for angle in np.arange(8)*np.pi/4:
            Y=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            R=Vq.T@S@Y@Vp
            if np.linalg.det(R)>0:seeds.append((1.,R,np.zeros(3)))
    at,ao,ad=_triangles(a);bt,bo,bd=_triangles(b)
    if len(at) and len(bt):
        choice=np.random.default_rng(172).choice(len(at),min(130,len(at)),replace=False)
        neighbors=cKDTree(bd).query(ad[choice],k=min(3,len(bt)))[1]
        for u,js in zip(choice,np.asarray(neighbors).reshape(len(choice),-1)):
            for v in js:
                s,R,t=similarity(a[at[u][ao[u]]],b[bt[v][bo[v]]])
                if .5<s<2:seeds.append((s,R,t))
    tree=cKDTree(b);rank=[]
    for s,R,t in seeds:
        z=s*a@R.T+t;d,j=tree.query(z);gate=.20
        rank.append((len(set(j[d<gate])), -float(np.minimum(d,1).mean()),s,R,t))
    rank.sort(key=lambda v:(v[0],v[1]),reverse=True);best=None
    for _,__,s,R,t in rank[:16]:
        for step in range(20):
            z=s*a@R.T+t
            gate=max(.1*spacing(z),.35*(.85**step))
            ii,jj,d=unique_pairs(z,b,gate)
            if len(ii)<4:break
            sn,Rn,tn=similarity(a[ii],b[jj])
            if not .5<sn<2:break
            move=float(np.linalg.norm((sn*a@Rn.T+tn)-z));s,R,t=sn,Rn,tn
            if move<1e-7:break
        z=s*a@R.T+t;gate=.1*spacing(z);ii,jj,d=unique_pairs(z,b,gate)
        score=(len(ii),-float(np.median(d)) if len(d) else -1)
        if best is None or score>best[0]:best=(score,s,R,t,ii,jj,d,gate,z)
    _,s,R,t,ii,jj,d,gate,z=best
    scale=s*uq/up;translation=qo+uq*t-scale*(R@po)
    return dict(status='candidate',matched=len(ii),source_count=len(p),target_count=len(q),
                same_cluster_candidate=bool(len(ii)>=min_matches),threshold=min_matches,threshold_validated_for_project=False,
                pairs=[dict(source=int(i),target=int(j),distance=float(x*uq)) for i,j,x in zip(ii,jj,d)],
                scale=float(scale),rotation=R.tolist(),translation=translation.tolist(),
                aligned_source=(scale*p@R.T+translation).tolist(),gate=float(gate*uq),
                median_residual_spacing=float(np.median(d)) if len(d) else None,
                source_coverage=len(ii)/len(p),target_coverage=len(ii)/len(q),
                uses_color=False,identity_confirmed=False)
