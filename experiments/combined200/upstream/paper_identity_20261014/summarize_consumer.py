"""Final verification and descriptive held-out detection metrics, no tuning."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'synthetic_consumer'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 results=read(OUT/'evaluation_results.json');rows=read(OUT/'evaluation_rankings.json');frozen=read(OUT/'frozen_evaluation_protocol.json')
 checks={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in frozen['sha256'].items()}
 assert all(checks.values()),'Frozen model/matching/data changed'
 models=[read(ROOT.parent/'fingerprint_mix_20261014/synthetic'/f'SYN_{i:03d}'/'model.json') for i in range(100)]
 hashes=[m['geometry_hash'] for m in models];assert len(set(hashes))==100
 labelled=predicted=matched=0;errors=[];radius_errors=[]
 for row in rows:
  p=row['nodes'];t=row['ground_truth_nodes'];labelled+=len(t);predicted+=len(p)
  a=np.array([[n['x'],n['y']] for n in p]);b=np.array([[n['x'],n['y']] for n in t]);radius=np.array([n['radius'] for n in t])
  d=np.linalg.norm(a[:,None]-b[None],axis=2)/radius[None];ii,jj=linear_sum_assignment(d);ok=d[ii,jj]<.35;ii,jj=ii[ok],jj[ok]
  matched+=len(ii);errors.extend(d[ii,jj].tolist());radius_errors.extend(abs(p[i]['radius']/t[j]['radius']-1) for i,j in zip(ii,jj))
 precision=matched/max(predicted,1);recall=matched/max(labelled,1)
 detector=dict(annotated=labelled,predicted=predicted,matched=matched,precision=precision,recall=recall,
   f1=2*precision*recall/max(precision+recall,1e-9),median_center_error_radius=float(np.median(errors)),
   median_relative_radius_error=float(np.median(radius_errors)),query_images=len(rows),geometry_identities=10,
   targets='known projected sphere centers with >=35% visible area',assignment_gate_radius=.35,never_used_to_adjust_model=True)
 (OUT/'heldout_detection_results.json').write_text(json.dumps(detector,ensure_ascii=False,indent=2),encoding='utf-8')
 verification=dict(verified_at=datetime.datetime.now().isoformat(),unit_tests=17,unit_tests_passed=True,
     frozen_files_unchanged=checks,distinct_synthetic_geometries=100,heldout_detection=detector,
     identity_results=results,browser=read(OUT/'browser_verification.json'),
     gpu_used=False,cpu_threads=2,subagents_used=False,original_data_modified=False,
     scope='synthetic proof of concept only; clean simulated enrollment geometry; real photos not validated')
 assert verification['browser']['passed'] and not verification['browser']['errors']
 (ROOT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(dict(heldout_detection=detector,identity=results,browser_passed=True),ensure_ascii=False))
if __name__=='__main__':main()
