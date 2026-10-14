"""Small, versioned young-to-grown illustration; biological predictor unchanged."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,copy
import numpy as np
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;sys.path.insert(0,str(BASE))
from forecast import early_features,predict_new,growth_display,relax_contacts,metrics
from growth_geometry import arrays

def pack_spheres(models):
    result=relax_contacts(models,minimum_separation=1.005,max_movement=.8,iterations=2400)
    p,r=arrays(result['models']);d=np.linalg.norm(p[:,None]-p[None],axis=2)/(r[:,None]+r[None]);np.fill_diagonal(d,np.inf)
    result['minimum_contact_ratio']=float(d.min());result['overlapping_pairs']=int(np.sum(np.triu(d<.999,k=1)))
    result['iterations_limit']=2400;result['status']='clear' if result['overlapping_pairs']==0 else 'unresolved_overlap'
    return result

def predict_young(sample,pairs,group):
    early_features(sample)
    forecast=predict_new(sample,pairs,group);display=growth_display(sample,forecast['layout'],center_expansion=1.35)
    contact=pack_spheres(display['models']);grown=dict(display,models=contact['models'],contact_constraint={k:v for k,v in contact.items() if k!='models'})
    return dict(forecast=forecast,grown=grown)
