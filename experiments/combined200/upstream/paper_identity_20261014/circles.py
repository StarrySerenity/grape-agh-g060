"""Robust circle arcs. Visible-mask centroids are never called sphere centers."""
import numpy as np
from scipy.optimize import least_squares

def fit_circle(contour,seed=71):
    p=np.asarray(contour,float).reshape(-1,2)
    if not np.isfinite(p).all():raise ValueError('Nonfinite contour')
    fail=dict(accepted=False,reason='insufficient or degenerate arc',center=None,radius=None)
    if len(p)<6:return fail
    p=p[np.linspace(0,len(p)-1,min(len(p),160),dtype=int)]
    span=np.linalg.norm(np.ptp(p,axis=0))
    if span<2:return fail
    rng=np.random.default_rng(seed);candidates=[]
    for _ in range(200):
        a,b,c=p[rng.choice(len(p),3,replace=False)]
        mat=2*np.vstack([b-a,c-a])
        if abs(np.linalg.det(mat))<.003*span**2:continue
        center=np.linalg.solve(mat,np.array([b@b-a@a,c@c-a@a]))
        radius=np.linalg.norm(a-center)
        if not .2*span<radius<2*span:continue
        residual=abs(np.linalg.norm(p-center,axis=1)-radius)
        mask=residual<max(.8,.045*radius)
        if mask.sum()<6:continue
        angles=np.sort(np.arctan2(*(p[mask]-center).T[::-1]))
        extent=2*np.pi-np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))
        score=mask.sum()/len(p)+.1*min(extent,np.pi)/np.pi-.05*np.median(residual[mask])/radius
        candidates.append((score,center,radius,mask))
    if not candidates:return fail
    _,center,radius,mask=max(candidates,key=lambda v:v[0])
    fit=least_squares(lambda v:np.linalg.norm(p[mask]-v[:2],axis=1)-v[2],np.r_[center,radius],loss='soft_l1',f_scale=.5,max_nfev=60)
    center=fit.x[:2];radius=fit.x[2]
    residual=abs(np.linalg.norm(p-center,axis=1)-radius)
    mask=residual<max(.8,.045*radius)
    angles=np.sort(np.arctan2(*(p[mask]-center).T[::-1]))
    extent=2*np.pi-np.max(np.diff(np.r_[angles,angles[0]+2*np.pi])) if len(angles)>1 else 0
    support=float(mask.mean());error=float(np.median(residual[mask])/max(radius,1)) if mask.any() else 1.
    accepted=bool(support>=.48 and extent>=1.6 and radius>2 and radius<1.4*span and error<.05)
    return dict(accepted=accepted,reason='supported circle arc' if accepted else 'short arc or unstable circle',
                center=center.tolist(),radius=float(radius),arc_radians=float(extent),support_fraction=support,
                median_radial_error_over_radius=error,confidence=float(support*min(extent/(2*np.pi),1)),
                center_type='robust circle inferred from visible contour; not verified anatomical center')
