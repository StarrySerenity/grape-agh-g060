"""Calibrate on development, freeze, then compare both weights on fresh identities."""
from data import *
import argparse,time
from scipy.optimize import linear_sum_assignment
from engine200 import rank_nodes,accepted,gallery,gallery_image
import evaluate_synthetic_consumer as legacy_review
legacy_review.ROOT=ROOT;legacy_review.OUT=OUT;legacy_review.gallery=lambda:(gallery(),None);legacy_review.gallery_image=gallery_image
summary=legacy_review.summary

def ranks(split,versions):
 rows=[r for r in read(OUT/'query_manifest.json') if r['split']==split];results={v:[] for v in versions}
 for query in rows:
  for version in versions:
   nodes=query['nodes'] if version=='new200' else query['old_nodes'];ranking=rank_nodes(nodes);record=dict(query,nodes=nodes,ranking=ranking,model_version=version);results[version].append(record)
   save(OUT/f'{split}_rankings_{version}.json',results[version])
   top=ranking['candidates'][0];print(split,version,query['key'],'truth',query['identity'] if query['in_database'] else 'OUT','top',top['identity'],'score',round(top['score'],3),'seconds',round(ranking['elapsed_seconds'],2),flush=True)
 return results

def calibrate(records,version):
 choices=[]
 for score in np.arange(.36,.811,.005):
  threshold=dict(minimum_score=float(score),minimum_margin=.03,minimum_pairs=11,minimum_query_coverage=.65)
  metrics=summary(records,threshold);rank=(-metrics['unknown_wrong_accepted']-metrics['known_wrong_accepted'],metrics['known_correct_accepted']);choices.append((rank,threshold,metrics))
 best=max(r[0] for r in choices);optimal=[r for r in choices if r[0]==best];middle=(min(r[1]['minimum_score'] for r in optimal)+max(r[1]['minimum_score'] for r in optimal))/2
 _,threshold,metrics=min(optimal,key=lambda r:abs(r[1]['minimum_score']-middle))
 threshold.update(selected_on='development only',calibration_rule='midpoint of best-error/best-recall plateau; fixed margin .03, 11 unique matches',
     model_version=version,calibration_known_ids='SYN180..185',calibration_unknown_ids='SYN186..189',evaluation_not_opened_at_selection=True,
     scope='orthographic synthetic only; clean enrollment geometry; not calibrated for real photos',development_summary=metrics)
 return threshold

def detection_metrics(records):
 annotated=predicted=matched=0;errors=[];radius_errors=[]
 for row in records:
  nodes=row['nodes'];truth=row['ground_truth_nodes'];annotated+=len(truth);predicted+=len(nodes)
  if not nodes or not truth:continue
  p=np.array([[n['x'],n['y']] for n in nodes]);q=np.array([[n['x'],n['y']] for n in truth]);r=np.array([n['radius'] for n in truth]);d=np.linalg.norm(p[:,None]-q[None],axis=2)/r[None]
  cost=np.full((len(p),len(q)+len(p)),2.);cost[:,:len(q)]=np.where(d<.35,d,1e6);ii,jj=linear_sum_assignment(cost)
  keep=jj<len(q);ii,jj=ii[keep],jj[keep];keep=d[ii,jj]<.35;ii,jj=ii[keep],jj[keep];matched+=len(ii);errors.extend(d[ii,jj].tolist());radius_errors.extend(abs(nodes[i]['radius']/truth[j]['radius']-1) for i,j in zip(ii,jj))
 precision=matched/max(predicted,1);recall=matched/max(annotated,1)
 return dict(annotated=annotated,predicted=predicted,matched=matched,precision=precision,recall=recall,f1=2*precision*recall/max(precision+recall,1e-9),
    median_center_error_radius=float(np.median(errors)) if errors else None,median_relative_radius_error=float(np.median(radius_errors)) if radius_errors else None,
    label_definition='full sphere centers >=35% visible, assignment gate .35 radius; exact foreground supplied')

def freeze():
 path=OUT/'frozen_evaluation_protocol.json'
 if path.exists() or (OUT/'evaluation_rankings_new200.json').exists():raise RuntimeError('Freeze once, before opening fresh rankings')
 files=[ROOT/f for f in ['data.py','engine200.py','matching_fast.py','prepare_benchmark.py','evaluate.py','train.py']]
 files+=[OUT/f for f in ['detector_best.pt','detector_protocol.json','query_manifest.json','gallery.json','decision_threshold.json','baseline_decision_threshold.json']]
 files+=[LEGACY/f for f in ['circle_network.py','consumer_geometry.py','consumer_engine.py','train_circles.py','prepare_circle_data.py']]
 save(path,dict(created_at=time.strftime('%Y-%m-%dT%H:%M:%S'),fresh_rankings_not_opened=True,sha256={str(p):sha(p) for p in files}))
 print('FROZEN_BEFORE_FRESH_EVALUATION',flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['development','freeze','evaluation'],required=True);args=parser.parse_args()
 if args.phase=='freeze':freeze();return
 if args.phase=='development':
  if (OUT/'frozen_evaluation_protocol.json').exists():raise RuntimeError('Development already frozen')
  results=ranks('development',['new200','old100'])
  for version,records in results.items():
   threshold=calibrate(records,version);save(OUT/('decision_threshold.json' if version=='new200' else 'baseline_decision_threshold.json'),threshold)
   print('THRESHOLD',version,json.dumps(threshold,ensure_ascii=False),flush=True)
 else:
  frozen=read(OUT/'frozen_evaluation_protocol.json');assert all(sha(p)==h for p,h in frozen['sha256'].items()),'Frozen input changed'
  results=ranks('evaluation',['new200','old100']);comparison={}
  for version,records in results.items():
   threshold=read(OUT/('decision_threshold.json' if version=='new200' else 'baseline_decision_threshold.json'))
   metrics=summary(records,threshold);audit=legacy_review.audit_correspondences(records);detection=detection_metrics(records)
   report=dict(**metrics,registered_identities=41,views_per_identity=72,query_geometry_ids='SYN190..199',known_registered_geometry_ids='SYN190..194',unknown_geometry_ids='SYN195..199',
     detector_training_geometry_disjoint=True,query_circles_are_model_predictions=True,single_photo_input=True,no_threshold_changes_after_evaluation=True,
     threshold=threshold,model_version=version,physical_correspondence_audit=audit,heldout_detection=detection,
     scope='new shapes, orthographic synthetic, clean enrollment geometry; not real-world accuracy',same_gallery_and_query_pixels=True)
   comparison[version]=report;save(OUT/f'evaluation_rankings_{version}.json',records);save(OUT/f'evaluation_results_{version}.json',report)
   if version=='new200':save(OUT/'evaluation_results.json',report)
   # Render all query/candidate correspondences. Move per version to retain both.
   legacy_review.review(records,threshold);target=OUT/'qa'/version;target.mkdir(exist_ok=True)
   for p in (OUT/'qa').glob('identity_review_*.jpg'):p.replace(target/p.name)
  comparison['interpretation']='Continued training and added data both changed; not a controlled data-only ablation. Both models use identical new pixels, gallery and matcher, with separately development-calibrated thresholds.'
  save(OUT/'comparison.json',comparison);print('FRESH_EVALUATION_FINISHED',json.dumps({k:{s:v[s] for s in ['known_correct_accepted','known_wrong_accepted','unknown_wrong_accepted','unknown_correct_rejected']} for k,v in comparison.items() if isinstance(v,dict)}),flush=True)
if __name__=='__main__':main()
