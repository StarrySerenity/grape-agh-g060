"""Relative endpoint forecasting using only unchanged early inputs."""
import copy
import numpy as np
from scipy.spatial.distance import pdist,cdist
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.special import expit,logit
from growth_geometry import arrays

TARGETS=['radius_over_center_spacing','middle_major_ratio','minor_major_ratio','relative_nearest_spacing']


def endpoint_prediction(x,y,query,alpha=1.0):
    """Learn positive lengths and ordered shape ratios rather than clip failures."""
    y=np.asarray(y,float)
    if (y<=0).any() or (y[:,1:3]>=1).any() or (y[:,2]>y[:,1]).any():
        raise ValueError('Training target morphology is invalid')
    transformed=np.c_[np.log(y[:,0]),logit(np.clip(y[:,1],1e-8,1-1e-8)),
        logit(np.clip(y[:,2]/y[:,1],1e-8,1-1e-8)),np.log(y[:,3])]
    model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(x,transformed);p=model.predict(query)
    result=np.c_[np.exp(p[:,0]),expit(p[:,1]),expit(p[:,1])*expit(p[:,2]),np.exp(p[:,3])]
    if not np.isfinite(result).all():raise ValueError('Forecast overflow')
    return result


def adaptive_prediction(x,y,query,train_ids):
    """Select mean or regularized predictor using exclusively inner training folds."""
    options=[('mean',None)]+[(f'ridge_{a}',a) for a in [1.,3.,10.,30.]]
    predictions=np.zeros((len(options),len(y),4));inner=[]
    for i,identity in enumerate(train_ids):
        mask=np.arange(len(y))!=i
        inner.append(dict(held_out=identity,train_ids=[k for j,k in enumerate(train_ids) if j!=i]))
        for j,(name,alpha) in enumerate(options):
            predictions[j,i]=y[mask].mean(0) if alpha is None else endpoint_prediction(x[mask],y[mask],x[i:i+1],alpha)[0]
    mae=np.mean(abs(predictions-y[None]),axis=1)
    groups=[[0],[1,2],[3]];result=np.zeros((len(query),4));choices=[]
    for fields in groups:
        scores=mae[:,fields].mean(1);choice=int(np.argmin(scores));name,alpha=options[choice]
        pred=np.tile(y.mean(0),(len(query),1)) if alpha is None else endpoint_prediction(x,y,query,alpha)
        result[:,fields]=pred[:,fields]
        choices.append(dict(target_indices=fields,selected=name,inner_mae=float(scores[choice]),candidate_inner_mae={n:float(s) for (n,_),s in zip(options,scores)}))
    return result,dict(selection_scope='inner_training_leave_one_identity_out',training_ids=train_ids,inner_folds=inner,choices=choices)


def metrics(models):
    x,r=arrays(models);d=pdist(x);scale=np.median(d[d>0]);p=(x-x.mean(0))/scale
    e=np.maximum(np.linalg.eigvalsh(p.T@p/len(p))[::-1],1e-12)
    nn=cdist(x,x);np.fill_diagonal(nn,np.inf)
    return dict(berry_count=len(x),radius_over_center_spacing=float(np.median(r)/scale),
        middle_major_ratio=float(e[1]/e[0]),minor_major_ratio=float(e[2]/e[0]),
        relative_nearest_spacing=float(np.median(nn.min(1))/scale),radius_cv=float(np.std(r)/np.median(r)),
        fingerprint=np.r_[np.quantile(d/scale,[.1,.25,.5,.75,.9]),np.quantile(r/np.median(r),[.1,.25,.5,.75,.9]),e/e.sum()].tolist())


def early_features(sample):
    if sample.get('uses_future_reference') or any(m.get('uses_future_reference') for m in sample['models']):
        raise ValueError('future-reference corrected input is forbidden in forecasting')
    m=metrics(sample['models'])
    return np.array([np.log1p(m['berry_count']),m['radius_over_center_spacing'],m['middle_major_ratio'],
        m['minor_major_ratio'],m['relative_nearest_spacing'],m['radius_cv']])


def evaluate(pairs):
    results={}
    for group in sorted({p['group'] for p in pairs}):
        rows=[p for p in pairs if p['group']==group and p.get('status','ready')=='ready']
        if len(rows)<4:
            results[group]=dict(pairs=len(rows),status='insufficient_independent_pairs',folds=[]);continue
        x=np.array([early_features(p['early']) for p in rows]);ms=[metrics(p['late']['models']) for p in rows]
        y=np.array([[m[k] for k in TARGETS] for m in ms]);folds=[]
        for i,p in enumerate(rows):
            train=np.arange(len(rows))!=i
            train_ids=[q['identity'] for j,q in enumerate(rows) if j!=i]
            predicted,selection=adaptive_prediction(x[train],y[train],x[i:i+1],train_ids)
            pred=predicted[0];baseline=y[train].mean(0)
            folds.append(dict(identity=p['identity'],train_ids=[q['identity'] for j,q in enumerate(rows) if j!=i],
                early_sample=p['early']['sample_id'] if 'sample_id' in p['early'] else None,
                target_days=p['days'],reference=y[i].tolist(),prediction=pred.tolist(),baseline=baseline.tolist(),
                out_of_range=bool((pred<0).any() or (pred[1:3]>1).any()),uses_future_reference=False,model_selection=selection))
        pred=np.array([f['prediction'] for f in folds]);base=np.array([f['baseline'] for f in folds])
        mae=np.mean(abs(pred-y),0);baseline_mae=np.mean(abs(base-y),0)
        results[group]=dict(pairs=len(rows),targets=TARGETS,folds=folds,model_mae=mae.tolist(),baseline_mae=baseline_mae.tolist(),
            better_than_mean=(mae<baseline_mae).tolist(),status='exploratory_relative_endpoint_forecast',
            model_parameterization='nested train-only choice: mean or constrained Ridge alpha in [1,3,10,30]',
            limitations=['no physical size calibration','not a continuous biological growth curve',
                'estimated late spheres are targets, not agricultural ground truth','future-guided corrections are excluded'])
    return results


def predict_layout(sample,target):
    """Illustrate cohort-level predicted metrics without consulting this sample's future."""
    early_features(sample)
    target=np.asarray(target,float)
    if target.shape!=(4,) or not np.isfinite(target).all() or (target<=0).any() or (target[1:3]>1).any() or target[2]>target[1]:
        raise ValueError('Invalid predicted morphology; do not fabricate a plausible result')
    x,r=arrays(sample['models']);mean=x.mean(0);v=x-mean
    eig,U=np.linalg.eigh(v.T@v/len(v));eig=eig[::-1];U=U[:,::-1]
    p=v@U
    p[:,1]*=np.sqrt(target[1]*eig[0]/max(eig[1],1e-12));p[:,2]*=np.sqrt(target[2]*eig[0]/max(eig[2],1e-12))
    shape=p@U.T;shape*=np.median(pdist(x))/np.median(pdist(shape))
    shape+=mean;rr=r/np.median(r)*target[0]*np.median(pdist(shape));models=copy.deepcopy(sample['models'])
    for m,c,radius in zip(models,shape,rr):m.update(center=c.tolist(),radius=float(radius),shape='sphere',evidence='predicted_layout_illustration')
    return dict(models=models,uses_future_reference=False,illustrative_not_individual_growth_truth=True,
        constraints='PCA shape ratios and relative radius only; no per-berry biological motion learned',
        predicted_targets=target.tolist(),actual_illustration_metrics=metrics(models),
        relative_nearest_spacing_not_enforced=True)


def growth_display(sample,layout,progress=1.0,center_expansion=1.35):
    """Interpolate an explicit, uncalibrated display scale separately from predicted shape."""
    if not 0<=progress<=1 or not 1<=center_expansion<=4:raise ValueError('Invalid display range')
    early_features(sample);x,r=arrays(sample['models']);y,s=arrays(layout['models'])
    if len(x)!=len(y):raise ValueError('Different berry topology')
    if progress==0:return dict(models=copy.deepcopy(sample['models']),median_radius_ratio=1.,center_spacing_ratio=1.,absolute_scale_is_assumed=True)
    center=x.mean(0);target_center=center_expansion*(y-center)+center;target_radius=center_expansion*s
    models=copy.deepcopy(sample['models'])
    for m,c,rr in zip(models,(1-progress)*x+progress*target_center,(1-progress)*r+progress*target_radius):
        m.update(center=c.tolist(),radius=float(rr),shape='sphere',evidence='growth_prediction_display')
    return dict(models=models,median_radius_ratio=float(np.median([(1-progress)*a+progress*b for a,b in zip(r,target_radius)])/np.median(r)),
        center_spacing_ratio=float(np.median(pdist(np.array([m['center'] for m in models])))/np.median(pdist(x))),
        absolute_scale_is_assumed=True,center_expansion=center_expansion,progress=progress,
        uses_future_reference=False,per_berry_dropout_is_not_predicted=True)


def relax_contacts(models,minimum_separation=.72,max_movement=.8,iterations=60):
    """Bounded approximate non-interpenetration, not learned biological motion."""
    x,r=arrays(models);origin=x.copy();ii,jj=np.triu_indices(len(x),1);sumr=r[ii]+r[jj]
    before=np.linalg.norm(x[ii]-x[jj],axis=1)/sumr
    for step in range(iterations):
        vec=x[ii]-x[jj];d=np.linalg.norm(vec,axis=1);gap=minimum_separation*sumr-d;active=gap>0
        if not active.any():break
        a=ii[active];b=jj[active];length=d[active];direction=vec[active]/np.maximum(length[:,None],1e-12)
        for k in np.flatnonzero(length<1e-10):
            direction[k]=np.array([np.sin(a[k]+1),np.cos(b[k]+1),.4]);direction[k]/=np.linalg.norm(direction[k])
        force=direction*gap[active,None]*.35;delta=np.zeros_like(x);np.add.at(delta,a,force);np.add.at(delta,b,-force)
        neighbors=np.bincount(np.r_[a,b],minlength=len(x));x+=delta/np.maximum(neighbors[:,None],1)
        movement=x-origin;length=np.linalg.norm(movement,axis=1)
        x=origin+movement*np.minimum(1,max_movement*r/np.maximum(length,1e-12))[:,None]
    after=np.linalg.norm(x[ii]-x[jj],axis=1)/sumr;result=copy.deepcopy(models)
    for model,c in zip(result,x):model['center']=c.tolist()
    return dict(models=result,before_severe_overlap_pairs=int(np.sum(before<.5)),after_severe_overlap_pairs=int(np.sum(after<.5)),
        minimum_separation_ratio=minimum_separation,maximum_movement_over_radius=float(np.max(np.linalg.norm(x-origin,axis=1)/r)),
        is_geometric_display_constraint=True,per_berry_biological_motion_learned=False)


def predict_new(sample,pairs,group):
    rows=[p for p in pairs if p['group']==group and p.get('status','ready')=='ready']
    if len(rows)<4:raise ValueError('Insufficient independent training pairs for this cohort')
    if sample.get('bunch_id'):
        rows=[p for p in rows if p['identity']!=sample['bunch_id']]
    if len(rows)<4:raise ValueError('Insufficient other identities after excluding this bunch')
    x=np.array([early_features(p['early']) for p in rows]);y=np.array([[metrics(p['late']['models'])[k] for k in TARGETS] for p in rows])
    predicted,selection=adaptive_prediction(x,y,early_features(sample)[None],[p['identity'] for p in rows]);prediction=predicted[0]
    layout=predict_layout(sample,prediction)
    return dict(prediction=prediction.tolist(),targets=TARGETS,layout=layout,group=group,target_days=rows[0]['days'],
        training_identities=[p['identity'] for p in rows],uses_future_reference=False,
        absolute_sizes_available=False,model_selection=selection)
