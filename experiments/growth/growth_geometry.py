"""Partial temporal correspondence and independently checked sphere proposals."""
import copy
import itertools
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist


def arrays(models):
    x=np.asarray([m['center'] for m in models],float)
    r=np.asarray([m['radius'] for m in models],float)
    if len(x)<4 or x.shape!=(len(models),3) or not np.isfinite(x).all():
        raise ValueError('At least four finite XYZ centers are required')
    if not np.isfinite(r).all() or (r<=0).any():raise ValueError('Positive finite radii required')
    if len({str(m['id']) for m in models})!=len(models):raise ValueError('Duplicate berry IDs')
    if len(np.unique(x,axis=0))<4:raise ValueError('Insufficient distinct centers')
    return x,r


def similarity(source,target):
    """Return row-vector transform target = scale * source @ R + t."""
    a=source.mean(axis=0);b=target.mean(axis=0)
    x=source-a;y=target-b
    u,s,v=np.linalg.svd(x.T@y)
    fix=np.ones(3);fix[-1]=np.sign(np.linalg.det(u@v))
    R=u@np.diag(fix)@v
    scale=float(np.sum(s*fix)/max(np.sum(x*x),1e-12))
    return R,scale,b-scale*a@R


def align_models(early,late,anchors=None):
    x,er=arrays(early);y,_=arrays(late)
    sx=float(np.median(pdist(x)));sy=float(np.median(pdist(y)))
    _,ax=np.linalg.eigh(np.cov(x.T));_,ay=np.linalg.eigh(np.cov(y.T))
    initial_scale=sx/sy;starts=[];anchor_solution=None
    if anchors:
        ei={str(b['id']):i for i,b in enumerate(early)};li={str(b['id']):i for i,b in enumerate(late)}
        if len(anchors)<3 or len({p[0] for p in anchors})!=len(anchors) or len({p[1] for p in anchors})!=len(anchors):
            raise ValueError('At least three unique one-to-one anchors are required')
        xx=x[[ei[str(p[0])] for p in anchors]];yy=y[[li[str(p[1])] for p in anchors]]
        if np.linalg.matrix_rank(xx-xx.mean(0),tol=1e-8)<2 or np.linalg.matrix_rank(yy-yy.mean(0),tol=1e-8)<2:
            raise ValueError('Anchor centers must not be collinear')
        anchor_solution=similarity(yy,xx)
    for p in itertools.permutations(range(3)):
        for signs in itertools.product([-1,1],repeat=3):
            M=np.eye(3)[:,p]@np.diag(signs)
            R=ay@M@ax.T
            if np.linalg.det(R)>.99:starts.append(R)
    solutions=[]
    for R0 in starts if anchor_solution is None else []:
        R=R0;scale=initial_scale;t=x.mean(0)-scale*y.mean(0)@R
        for step in range(35):
            z=scale*y@R+t;dist=cdist(x,z);ii,jj=linear_sum_assignment(dist)
            errors=dist[ii,jj];keep=np.argsort(errors)[:max(4,int(len(ii)*.85))]
            rr,ss,tt=similarity(y[jj[keep]],x[ii[keep]])
            ss=float(np.clip(ss,initial_scale*.65,initial_scale*1.5))
            tt=x[ii[keep]].mean(0)-ss*y[jj[keep]].mean(0)@rr
            delta=np.linalg.norm(rr-R)+abs(ss-scale)+np.linalg.norm(tt-t)
            R,scale,t=rr,ss,tt
            if delta<1e-8:break
        z=scale*y@R+t;dist=cdist(x,z);ii,jj=linear_sum_assignment(dist)
        score=float(np.mean(np.minimum(dist[ii,jj]/np.median(er),3)))
        solutions.append((score,R,scale,t,dist,ii,jj))
    if anchor_solution is not None:
        R,scale,t=anchor_solution;dist=cdist(x,scale*y@R+t);ii,jj=linear_sum_assignment(dist)
        solutions=[(float(np.mean(np.minimum(dist[ii,jj]/np.median(er),3))),R,scale,t,dist,ii,jj)]
    solutions.sort(key=lambda s:s[0]);best=solutions[0]
    score,R,scale,t,dist,ii,jj=best;radius=float(np.median(er))
    matches=[]
    for i,j in zip(ii,jj):
        residual=float(dist[i,j]/radius)
        if residual>1.4:continue
        alternatives=np.delete(dist[i],j)
        margin=float((np.min(alternatives)-dist[i,j])/radius) if len(alternatives) else 0
        reverse=np.delete(dist[:,j],i)
        margin=min(margin,float((np.min(reverse)-dist[i,j])/radius))
        matches.append(dict(early_id=str(early[i]['id']),late_id=str(late[j]['id']),
            early_index=int(i),late_index=int(j),residual_over_radius=residual,
            local_margin_over_radius=margin,geometric_confidence='strong_candidate' if residual<.6 and margin>.3 else 'ambiguous',
            identity_verified=False))
    signature=lambda s:tuple(int(j) for j in s[6])
    second=next((s for s in solutions[1:] if signature(s)!=signature(best)),None)
    gap=float(second[0]-score) if second else None
    return dict(rotation=R.tolist(),scale=scale,translation=t.tolist(),
        transformed_late_centers=(scale*y@R+t).tolist(),matches=matches,
        median_residual_over_radius=float(np.median([m['residual_over_radius'] for m in matches])) if matches else None,
        coverage_early=len(matches)/len(early),coverage_late=len(matches)/len(late),objective=score,
        alternative_objective_gap=gap,ambiguous_global_alignment=gap is not None and gap<.08,
        physical_growth_scale=False,all_correspondences_verified=False,
        alignment_source='manual_anchor_pairs' if anchors else 'automatic_geometric_candidates',anchors=anchors or [])


def surface_metrics(points,center,radius,validation=None):
    x=np.asarray(points,float);c=np.asarray(center,float)
    if len(x)==0:return dict(points=0,median_r=None,p90_r=None,direction_bins=0,validation_median_r=None)
    d=x-c;length=np.linalg.norm(d,axis=1);res=np.abs(length/radius-1)
    good=res<.12;directions=d[good]/np.maximum(length[good,None],1e-12)
    if len(directions):
        az=((np.arctan2(directions[:,1],directions[:,0])+np.pi)/(2*np.pi)*12).astype(int).clip(0,11)
        el=((directions[:,2]+1)*3).astype(int).clip(0,5)
        occupied=len(np.unique(az+12*el))
    else:occupied=0
    v=x if validation is None else np.asarray(validation,float)
    ev=np.abs(np.linalg.norm(v-c,axis=1)/radius-1)
    return dict(points=len(x),surface_inliers=int(good.sum()),median_r=float(np.median(res)),p90_r=float(np.quantile(res,.9)),
        direction_bins=occupied,validation_median_r=float(np.median(ev)),validation_p90_r=float(np.quantile(ev,.9)))


def refit_sphere(points,model,prior_center=None,tree=None):
    x=np.asarray(points,float);c=np.asarray(model['center'],float);r=float(model['radius'])
    tree=tree or cKDTree(x)
    ids=np.asarray(tree.query_ball_point(c,1.45*r),int)
    if len(ids):
        d=np.linalg.norm(x[ids]-c,axis=1)
        ids=ids[(d>.55*r)&(np.abs(d-r)<.4*r)]
    if len(ids)<40:return dict(accepted=False,reason='insufficient_early_surface',model=copy.deepcopy(model))
    if len(ids)>4000:ids=ids[np.linspace(0,len(ids)-1,4000,dtype=int)]
    p=x[ids];singular=np.linalg.svd(p-p.mean(0),compute_uv=False)
    if singular[1]<r*.05 or singular[2]/singular[1]<.015:
        return dict(accepted=False,reason='near_planar_or_linear_patch',model=copy.deepcopy(model))
    # Every fifth source index is held out and never used to optimize parameters.
    val=np.arange(len(ids))%5==0;train=p[~val];heldout=p[val]
    before=surface_metrics(p,c,r,heldout)
    hint=np.asarray(prior_center,float) if prior_center is not None else c
    hint=c+np.clip(hint-c,-.2*r,.2*r)
    def residual(v):
        data=(np.linalg.norm(train-v[:3],axis=1)-v[3])/r
        # Weak regularization: observations dominate the temporal suggestion.
        return np.r_[data,(v[:3]-c)/r*.6,(v[3]-r)/r*.8,(v[:3]-hint)/r*.25 if prior_center is not None else []]
    sol=least_squares(residual,np.r_[c,r],bounds=(np.r_[c-.35*r,.7*r],np.r_[c+.35*r,1.3*r]),
        loss='soft_l1',f_scale=.035,max_nfev=100)
    cc,rr=sol.x[:3],float(sol.x[3]);after=surface_metrics(p,cc,rr,heldout)
    movement=float(np.linalg.norm(cc-c)/r)
    accepted=bool(sol.success and after['direction_bins']>=8 and after['validation_median_r']<before['validation_median_r']*.94
        and after['validation_p90_r']<=before['validation_p90_r']*1.03 and after['median_r']<.14 and movement<.45)
    q=copy.deepcopy(model)
    if accepted:
        for key in ['axes','rotation','source','fit','surface_support_points','surface_refinement','support_indices_zero_based']:
            q.pop(key,None)
        q.update(center=cc.tolist(),radius=rr,shape='sphere',evidence='early_surface_refit_candidate',
            uses_future_reference=prior_center is not None,individually_verified_around_full_orbit=False,
            observations=[],surface_support_indices_require_recompute=True)
    return dict(accepted=accepted,reason='heldout_surface_improvement' if accepted else 'no_reliable_heldout_improvement',
        model=q,before=before,after=after,center_movement_r=movement,
        support_source_indices=ids.tolist(),heldout_source_indices=ids[val].tolist())


def duplicate_candidates(models):
    x,r=arrays(models);d=cdist(x,x)
    return [dict(a=str(models[i]['id']),b=str(models[j]['id']),separation_over_smaller_radius=float(d[i,j]/min(r[i],r[j])))
        for i in range(len(x)) for j in range(i+1,len(x)) if d[i,j]<.7*min(r[i],r[j])]


def repair_early(points,early,late,alignment):
    x=np.asarray(points,float);tree=cKDTree(x)
    hints={m['early_id']:alignment['transformed_late_centers'][m['late_index']] for m in alignment['matches']
        if m['geometric_confidence']=='strong_candidate' and not alignment['ambiguous_global_alignment']}
    updated=[];checks=[]
    for b in early:
        check=refit_sphere(x,b,prior_center=hints.get(str(b['id'])),tree=tree)
        updated.append(check['model']);checks.append(dict(id=str(b['id']),**{k:v for k,v in check.items() if k!='model'}))
    additions=[];pending=[];matched={m['late_index'] for m in alignment['matches']}
    c,r=arrays(updated);typical=float(np.median(r));new_ids={str(b['id']) for b in updated}
    for j,center in enumerate(alignment['transformed_late_centers']):
        if j in matched:continue
        distances=np.linalg.norm(c-center,axis=1)
        if np.min(distances/(r+typical))<.65:continue
        key=f'TG_{j:03d}'
        while key in new_ids:key+='x'
        seed=dict(id=key,center=center,radius=typical,shape='sphere',uses_future_reference=True,late_reference_id=str(late[j]['id']))
        check=refit_sphere(x,seed,tree=tree)
        # Missing berries remain review proposals even if their local sphere is plausible:
        # a matched pose alone does not prove the patch is this berry rather than a neighbor.
        pending.append(dict(seed=seed,proposal=check['model'],surface_fit_accepted=check['accepted'],
            reason='manual_video_identity_confirmation_required' if check['accepted'] else check['reason'],
            diagnostics={k:v for k,v in check.items() if k!='model'}))
    return dict(models=updated,checks=checks,added_models=additions,pending_additions=pending,
        modified_ids=[q['id'] for q in checks if q['accepted']],duplicates=duplicate_candidates(updated),
        uses_future_reference=True,all_berries_verified=False,true_total_berry_count=None)
