"""Calibrated berry-center SfM adaptation, with parallax and quality rejection."""
import numpy as np,cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
cv2.setNumThreads(2)

def _map(frame):return {n['track_id']:n for n in frame['nodes']}
def _xy(nodes,ids):return np.array([[nodes[i]['x'],nodes[i]['y']] for i in ids],float)
def _project(points,R,t,K):
    z=points@R.T+t
    return (z@K.T)[:,:2]/np.maximum(z[:,2,None],1e-8),z[:,2]

def _triangulate(a,b,A,B,K):
    P=K@np.c_[A[0],A[1]];Q=K@np.c_[B[0],B[1]]
    h=cv2.triangulatePoints(P,Q,a.T,b.T);points=(h[:3]/h[3]).T
    ca=-A[0].T@A[1];cb=-B[0].T@B[1]
    va=points-ca;vb=points-cb
    cos=(va*vb).sum(1)/np.maximum(np.linalg.norm(va,axis=1)*np.linalg.norm(vb,axis=1),1e-12)
    angles=np.degrees(np.arccos(np.clip(cos,-1,1)))
    pa,za=_project(points,*A,K);pb,zb=_project(points,*B,K)
    error=np.maximum(np.linalg.norm(pa-a,axis=1),np.linalg.norm(pb-b,axis=1))
    good=np.isfinite(points).all(1)&(za>0)&(zb>0)&(angles>.35)&(error<5)
    return points,good,angles

def _bundle(frames,maps,poses,points,ids,K,first,second):
    camera_ids=sorted(poses);variable=[i for i in camera_ids if i!=first]
    camera_index={i:k for k,i in enumerate(variable)};point_index={v:k for k,v in enumerate(ids)}
    observations=[]
    for f in camera_ids:
        for id,n in maps[f].items():
            if id in point_index:observations.append((f,point_index[id],n['x'],n['y'],n['radius']))
    if len(observations)<2*len(ids):return poses,points
    cam=np.array([np.r_[cv2.Rodrigues(poses[f][0])[0].ravel(),poses[f][1]] for f in variable])
    x=np.r_[cam.ravel(),points.ravel()];nc=6*len(variable)
    sparsity=lil_matrix((2*len(observations)+1,len(x)),dtype=int)
    for k,(f,j,_,__,r) in enumerate(observations):
        if f in camera_index:sparsity[2*k:2*k+2,6*camera_index[f]:6*camera_index[f]+6]=1
        sparsity[2*k:2*k+2,nc+3*j:nc+3*j+3]=1
    sparsity[-1,6*camera_index[second]+3:6*camera_index[second]+6]=1
    def unpack(v):
        ps={first:poses[first]}
        for f,k in camera_index.items():ps[f]=(cv2.Rodrigues(v[k*6:k*6+3])[0],v[k*6+3:k*6+6])
        return ps,v[nc:].reshape(-1,3)
    def residual(v):
        ps,pts=unpack(v);out=[]
        for f,j,u,w,r in observations:
            uv,_=_project(pts[j:j+1],*ps[f],K);out.extend((uv[0]-[u,w])/max(r,1))
        out.append(10*(np.linalg.norm(ps[second][1])-1))
        return np.asarray(out)
    result=least_squares(residual,x,jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=.1,x_scale='jac',max_nfev=45,ftol=1e-5)
    return unpack(result.x)

def reconstruct(frames,K,max_candidates=6):
    K=np.asarray(K,float);maps=[_map(f) for f in frames];n=len(frames)
    fail=dict(status='failed',reason='No baseline with stable triangulation and sufficient tracks',points=[],candidates=[])
    if n<3:return fail
    counts={}
    for m in maps:
        for id in m:counts[id]=counts.get(id,0)+1
    allowed={id for id,c in counts.items() if c>=max(3,int(np.ceil(.30*n)))}
    maps=[{id:v for id,v in m.items() if id in allowed} for m in maps]
    pairs=[]
    gaps=sorted(set([max(2,n//6),max(3,n//3),max(4,n//2),n-2]))
    for gap in gaps:
        options=[]
        for i in range(n-gap):
            j=i+gap;shared=sorted(set(maps[i])&set(maps[j]))
            if len(shared)<8:continue
            shift=float(np.median(np.linalg.norm(_xy(maps[i],shared)-_xy(maps[j],shared),axis=1)))
            if shift<.5:continue
            options.append((len(shared),shift,i,j,shared))
        options.sort(reverse=True)
        pairs.extend(options[:2])
    pairs.sort(key=lambda r:(-r[0],-r[1]));candidates=[];diagnostics=[]
    for _,__,i,j,shared in pairs[:max_candidates]:
        a=_xy(maps[i],shared);b=_xy(maps[j],shared)
        diag=dict(initial_frames=[i,j],shared=len(shared));diagnostics.append(diag)
        E,mask=cv2.findEssentialMat(a,b,K,method=cv2.RANSAC,prob=.999,threshold=1.5)
        if E is None:diag['rejected_at']='essential_matrix';continue
        try:_,R,t,mask=cv2.recoverPose(E[:3],a,b,K,mask=mask)
        except cv2.error:diag['rejected_at']='recover_pose';continue
        poses={i:(np.eye(3),np.zeros(3)),j:(R,t.ravel())}
        p,good,angles=_triangulate(a,b,poses[i],poses[j],K);good &= mask.ravel()>0
        diag.update(triangulated=int(good.sum()),pose_inliers=int((mask.ravel()>0).sum()),parallax=float(np.median(angles[good])) if good.any() else None)
        if good.sum()<8 or np.median(angles[good])<.45:diag['rejected_at']='initial_triangulation';continue
        cloud={id:pt for id,pt,g in zip(shared,p,good) if g}
        # Revisit unregistered frames as more landmarks become available.
        for iteration in range(2):
            for f in range(n):
                if f in poses:continue
                known=sorted(set(maps[f])&set(cloud))
                if len(known)<6:continue
                xyz=np.array([cloud[id] for id in known]);uv=_xy(maps[f],known)
                ok,rv,tv,inliers=cv2.solvePnPRansac(xyz,uv,K,None,iterationsCount=120,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
                if not ok or inliers is None or len(inliers)<6:continue
                poses[f]=(cv2.Rodrigues(rv)[0],tv.ravel())
            # Triangulate tracks missed by initial frame pair, using widest registered separation.
            for id in allowed-set(cloud):
                seen=[f for f in sorted(poses) if id in maps[f]]
                if len(seen)<3:continue
                l,u=seen[0],seen[-1]
                pt,g,ang=_triangulate(_xy(maps[l],[id]),_xy(maps[u],[id]),poses[l],poses[u],K)
                if g[0]:cloud[id]=pt[0]
        diag['registered_frames']=len(poses)
        if len(poses)<max(3,int(n*.6)):diag['rejected_at']='frame_registration';continue
        ids=sorted(cloud);points=np.array([cloud[id] for id in ids])
        poses,points=_bundle(frames,maps,poses,points,ids,K,i,j)
        errors={id:[] for id in ids};observations={id:0 for id in ids};negative=set()
        for f in poses:
            for k,id in enumerate(ids):
                if id not in maps[f]:continue
                node=maps[f][id];uv,depth=_project(points[k:k+1],*poses[f],K)
                if depth[0]<=0:negative.add(id)
                errors[id].append(float(np.linalg.norm(uv[0]-[node['x'],node['y']])/node['radius']));observations[id]+=1
        keep=[k for k,id in enumerate(ids) if id not in negative and observations[id]>=max(3,int(np.ceil(.30*n))) and np.median(errors[id])<.20 and np.quantile(errors[id],.9)<.6]
        diag['filtered_points']=len(keep)
        if len(keep)<8:diag['rejected_at']='landmark_quality';continue
        ids=[ids[k] for k in keep];points=points[keep]
        frame_errors=[]
        for f in poses:
            es=[]
            for k,id in enumerate(ids):
                if id in maps[f]:
                    uv,depth=_project(points[k:k+1],*poses[f],K);node=maps[f][id]
                    es.append(np.linalg.norm(uv[0]-[node['x'],node['y']]))
            if es:frame_errors.append(float(np.mean(es)/np.mean([maps[f][id]['radius'] for id in ids if id in maps[f]])))
        parallax=[]
        for id,pt in zip(ids,points):
            seen=[f for f in poses if id in maps[f]]
            rays=np.array([pt+poses[f][0].T@poses[f][1] for f in seen]);rays/=np.linalg.norm(rays,axis=1)[:,None]
            parallax.append(float(np.degrees(np.arccos(np.clip(np.min(rays@rays.T),-1,1)))))
        value=dict(status='reconstructed',points=points.tolist(),track_ids=ids,mean_error_radius=float(np.mean(frame_errors)),
                   median_parallax_degrees=float(np.median(parallax)),registered_frames=len(poses),frame_count=n,initial_frames=[i,j],
                   cameras=[dict(frame=f,rotation=R.tolist(),translation=t.tolist()) for f,(R,t) in sorted(poses.items())],
                   K=K.tolist(),intrinsics_assumption='provided K; calibration must be verified separately',
                   minimum_track_frame_fraction=.30,identity_confirmed=False)
        if value['median_parallax_degrees']>.45:candidates.append(value)
    if not candidates:fail['candidates']=diagnostics;return fail
    candidates.sort(key=lambda v:v['mean_error_radius']);best=candidates[0]
    best['candidates']=[dict(initial_frames=v['initial_frames'],points=len(v['points']),error=v['mean_error_radius'],parallax=v['median_parallax_degrees']) for v in candidates]
    best['candidate_diagnostics']=diagnostics
    return best
