"""Adjacent 2D similarity alignment, gated unique association, explicit broken tracks."""
import copy,numpy as np
from scipy.spatial import cKDTree
from registration import similarity,unique_pairs

def track_frames(frames):
    out=copy.deepcopy(frames);next_id=0;previous=None;memory={}
    for index,f in enumerate(out):
        nodes=f['nodes'];q=np.array([[n['x'],n['y']] for n in nodes],float).reshape(-1,2)
        for n in nodes:n['track_id']=None
        if previous is not None and min(len(previous['nodes']),len(nodes))>=4:
            old=previous['nodes'];p=np.array([[n['x'],n['y']] for n in old]);radius=np.median([n['radius'] for n in old])
            gate=max(2.,.65*radius)
            # Start with displacement votes, covering abrupt translation before refinement.
            offsets=(q[None]-p[:,None]).reshape(-1,2)
            grid=np.round(offsets/max(radius*.25,1)).astype(int)
            keys,counts=np.unique(grid,axis=0,return_counts=True)
            shifts=[np.median(q,axis=0)-np.median(p,axis=0),np.zeros(2)]
            shifts += [k*max(radius*.25,1) for k in keys[np.argsort(counts)[-6:]]]
            best=None
            for shift in shifts:
                s=1.;R=np.eye(2);t=shift
                for _ in range(4):
                    ii,jj,d=unique_pairs(s*p@R.T+t,q,gate)
                    if len(ii)<4:break
                    sn,Rn,tn=similarity(p[ii],q[jj])
                    if not .80<sn<1.25 or abs(np.arctan2(Rn[1,0],Rn[0,0]))>.35:break
                    s,R,t=sn,Rn,tn
                ii,jj,d=unique_pairs(s*p@R.T+t,q,gate)
                # Full-circle radii should not jump strongly in adjacent frames.
                good=np.array([.65<nodes[j]['radius']/(old[i]['radius']*s)<1.55 for i,j in zip(ii,jj)],bool)
                ii,jj,d=ii[good],jj[good],d[good]
                score=(len(ii),-float(np.median(d)) if len(d) else -1e9)
                if best is None or score>best[0]:best=(score,ii,jj,d,s,R,t)
            _,ii,jj,d,s,R,t=best
            for i,j,error in zip(ii,jj,d):
                nodes[j]['track_id']=old[i]['track_id'];nodes[j]['association_error_radius']=float(error/radius)
            f['association']=dict(previous=index-1,matched=len(ii),gate_pixels=float(gate))
            # A brief detector miss should not create another anatomical landmark.
            # Propagate stored positions using the measured common frame transform.
            if len(ii)>=6:
                for id,entry in memory.items():
                    entry['position']=s*(R@entry['position'])+t;entry['radius']*=s
                used={n['track_id'] for n in nodes if n['track_id'] is not None}
                missing=[id for id,e in memory.items() if id not in used and 1<index-e['frame']<=3]
                fresh=[j for j,n in enumerate(nodes) if n['track_id'] is None]
                if missing and fresh:
                    pp=np.array([memory[id]['position'] for id in missing]);qq=q[fresh]
                    mi,mj,md=unique_pairs(pp,qq,max(2,.40*radius))
                    recovered=0
                    for u,v,error in zip(mi,mj,md):
                        id=missing[u];j=fresh[v]
                        if .75<nodes[j]['radius']/memory[id]['radius']<1.35:
                            nodes[j]['track_id']=id;nodes[j]['association_error_radius']=float(error/radius)
                            nodes[j]['bridged_missing_frames']=index-memory[id]['frame']-1;recovered+=1
                    f['association']['bridged']=recovered
        for n in nodes:
            if n['track_id'] is None:n['track_id']=next_id;next_id+=1
            memory[n['track_id']]=dict(position=np.array([n['x'],n['y']]),radius=n['radius'],frame=index)
        previous=f
    return out
