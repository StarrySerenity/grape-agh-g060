"""Export the predicted sphere surface, explicitly separate from measured points."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,json,numpy as np,open3d as o3d
from growth_data import read,save
from forecast import growth_display,relax_contacts
ROOT=Path(__file__).resolve().parent


def sphere_cloud(models,path):
    n=500;k=np.arange(n);z=1-2*(k+.5)/n;angle=k*np.pi*(3-np.sqrt(5))
    unit=np.c_[np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z]
    points=np.vstack([np.asarray(m['center'])+m['radius']*unit for m in models])
    cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(points)
    cloud.colors=o3d.utility.Vector3dVector(np.tile([.67,.43,.82],(len(points),1)))
    if not o3d.io.write_point_cloud(str(path),cloud):raise OSError('Cannot write prediction PLY')
    verify=o3d.io.read_point_cloud(str(path))
    if len(verify.points)!=len(points) or not np.allclose(verify.points,points):raise ValueError('Prediction PLY readback failed')
    return len(points)


def main():
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);args=p.parse_args()
    summary=read(ROOT/'results/summary.json');report=next(r for r in summary['pairs'] if r['identity']==args.identity)
    if report['status']!='ready' and not report.get('has_demo_prediction'):raise ValueError('No explicit conditional stage display available')
    out=ROOT/'pairs'/args.identity;early=read(out/'early_original.json');layout=read(out/'predicted_layout.json')
    grown=growth_display(early,layout);contact=relax_contacts(grown['models']);grown['models']=contact['models']
    save(out/'contact_constraint.json',{k:v for k,v in contact.items() if k!='models'})
    n=sphere_cloud(grown['models'],out/'predicted_grown_spheres.ply')
    save(out/'predicted_grown_models.json',dict(berries=grown['models'],uses_future_reference=False,
        absolute_scale_is_assumed=True,center_expansion=1.35,model_illustration_not_measured_cloud=True,
        training_identities=layout['train_ids'],prediction_method=layout['method'],contact_constraint={k:v for k,v in contact.items() if k!='models'}))
    print(args.identity,n,'surface points; not measured geometry',flush=True)


if __name__=='__main__':main()
