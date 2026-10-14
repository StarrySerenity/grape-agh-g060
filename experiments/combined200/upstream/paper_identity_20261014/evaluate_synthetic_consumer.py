"""Lock decision thresholds on development, then open held-out geometry identities."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,time,hashlib,cv2,numpy as np
from pathlib import Path
from consumer_engine import rank_nodes,accepted,gallery_image,gallery
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'synthetic_consumer'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def summary(records,threshold):
 known=[r for r in records if r['in_database']];unknown=[r for r in records if not r['in_database']]
 top1=sum(r['ranking']['candidates'][0]['identity']==r['identity'] for r in known)
 top3=sum(r['identity'] in [c['identity'] for c in r['ranking']['candidates'][:3]] for r in known)
 correct_accept=sum(accepted(r['ranking'],threshold) and r['ranking']['candidates'][0]['identity']==r['identity'] for r in known)
 wrong_accept=sum(accepted(r['ranking'],threshold) and r['ranking']['candidates'][0]['identity']!=r['identity'] for r in known)
 unknown_accept=sum(accepted(r['ranking'],threshold) for r in unknown)
 return dict(known_queries=len(known),unknown_queries=len(unknown),known_top1=top1,known_top3=top3,
             known_correct_accepted=correct_accept,known_wrong_accepted=wrong_accept,known_abstained=len(known)-correct_accept-wrong_accept,
             unknown_wrong_accepted=unknown_accept,unknown_correct_rejected=len(unknown)-unknown_accept,
             known_top1_rate=top1/max(len(known),1),known_correct_accept_rate=correct_accept/max(len(known),1),
             unknown_false_accept_rate=unknown_accept/max(len(unknown),1),
             mean_query_seconds=float(np.mean([r['ranking']['elapsed_seconds'] for r in records])))

def evaluate(split):
 queries=[r for r in read(OUT/'query_manifest.json') if r['split']==split];records=[]
 for query in queries:
  ranking=rank_nodes(query['nodes']);rec=dict(query,ranking=ranking);records.append(rec)
  print(split,query['key'],'truth',query['identity'] if query['in_database'] else 'OUT',
        'top',ranking['candidates'][0]['identity'],'score',round(ranking['candidates'][0]['score'],3),'margin',round(ranking['margin'],3),
        'seconds',round(ranking['elapsed_seconds'],2),flush=True)
  save(OUT/(split+'_rankings.json'),records)
 return records

def audit_correspondences(records):
 """Ground-truth berry IDs enter only AFTER ranking, for physical-match auditing."""
 lookup={v['key']:v for v in gallery()[0]['records']};total=correct=0
 for r in records:
  candidate=r['ranking']['candidates'][0]
  if not r['in_database'] or candidate['identity']!=r['identity']:continue
  predicted=r['nodes'];truth=r['ground_truth_nodes'];template=lookup[candidate['key']]['nodes']
  p=np.array([[n['x'],n['y']] for n in predicted]);q=np.array([[n['x'],n['y']] for n in truth]);rad=np.array([n['radius'] for n in truth])
  d=np.linalg.norm(p[:,None]-q[None],axis=2)/rad[None];ii,jj=linear_sum_assignment(d)
  mapping={int(i):truth[j]['berry_id'] for i,j in zip(ii,jj) if d[i,j]<.35}
  checked=[(mapping[a['query']],template[a['gallery']]['berry_id']) for a in candidate['pairs'] if a['query'] in mapping]
  count=sum(a==b for a,b in checked);total+=len(checked);correct+=count
  r['post_ranking_physical_audit']=dict(labelled_pairs=len(checked),same_physical_berry=count,
       labels_used_for_ranking=False,query_label_assignment_gate_radius=.35)
 return dict(labelled_pairs=total,same_physical_berry=correct,correct_pair_fraction=correct/max(total,1),
             applies_to_correct_top1_only=True,labels_used_for_ranking=False)

def review(records,threshold):
 tiles=[]
 for r in records:
  query=cv2.imdecode(np.fromfile(str(ROOT/r['image']),np.uint8),1);candidate=r['ranking']['candidates'][0];im,view=gallery_image(candidate['key'])
  for i,pair in enumerate(candidate['pairs']):
   color=tuple(map(int,cv2.cvtColor(np.array([[[i*47%180,220,245]]],np.uint8),cv2.COLOR_HSV2BGR)[0,0]))
   a=r['nodes'][pair['query']];b=view['nodes'][pair['gallery']]
   for picture,node in [(query,a),(im,b)]:
    xy=(round(node['x']),round(node['y']));cv2.circle(picture,xy,round(node['radius']),color,1)
    cv2.putText(picture,str(i),xy,cv2.FONT_HERSHEY_SIMPLEX,.28,color,1)
  title=np.full((44,512,3),245,np.uint8);decision='ACCEPT' if accepted(r['ranking'],threshold) else 'UNSURE'
  cv2.putText(title,r['key']+' / truth '+(r['identity'] if r['in_database'] else 'OUT'),(3,16),cv2.FONT_HERSHEY_SIMPLEX,.37,(20,40,30),1)
  cv2.putText(title,'top '+candidate['identity']+f' / {candidate["score"]:.3f} / '+decision,(3,35),cv2.FONT_HERSHEY_SIMPLEX,.37,(20,40,30),1)
  tiles.append(np.vstack([title,np.hstack([query,im])]))
 qa=OUT/'qa';qa.mkdir(exist_ok=True)
 for start in range(0,len(tiles),8):
  group=tiles[start:start+8]
  while len(group)%2:group.append(np.full_like(group[0],245))
  cv2.imencode('.jpg',np.vstack([np.hstack(group[i:i+2]) for i in range(0,len(group),2)]),[cv2.IMWRITE_JPEG_QUALITY,93])[1].tofile(str(qa/f'identity_review_{start:02d}.jpg'))

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['development','evaluation'],required=True);args=parser.parse_args()
 if args.split=='development':
  records=evaluate('development');choices=[]
  for score in np.arange(.36,.811,.005):
   for margin in [.03]:
    for pairs in [11]:
     threshold=dict(minimum_score=float(score),minimum_margin=margin,minimum_pairs=pairs,minimum_query_coverage=.65)
     m=summary(records,threshold)
     # Prefer no false accepts, then recover as many known queries as possible.
     rank=(-m['unknown_wrong_accepted']-m['known_wrong_accepted'],m['known_correct_accepted'])
     choices.append((rank,threshold,m))
  best_rank=max(c[0] for c in choices);optimal=[c for c in choices if c[0]==best_rank]
  # Center the plateau of equally good development thresholds instead of
  # selecting its most permissive endpoint. Never inspect held-out rankings.
  middle=(min(c[1]['minimum_score'] for c in optimal)+max(c[1]['minimum_score'] for c in optimal))/2
  _,threshold,m=min(optimal,key=lambda c:abs(c[1]['minimum_score']-middle));threshold.update(selected_on='development only',calibration_rule='midpoint of best-error, best-recall score plateau; fixed margin .03 and 11 unique matches',calibration_known_geometry_ids='SYN080..085',calibration_unknown_geometry_ids='SYN086..089',
      evaluation_not_opened_at_selection=True,scope='illustrative synthetic renderer; not calibrated for real photos',development_summary=m,
      detector_checkpoint_sha256=hashlib.sha256((OUT/'detector_best.pt').read_bytes()).hexdigest())
  save(OUT/'decision_threshold.json',threshold);print('THRESHOLD LOCKED',json.dumps(threshold,ensure_ascii=False),flush=True)
 else:
  frozen=read(OUT/'frozen_evaluation_protocol.json')
  for file,expected in frozen['sha256'].items():
   actual=hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
   if actual!=expected:raise ValueError('Evaluation input changed after freeze: '+file)
  threshold=read(OUT/'decision_threshold.json');records=evaluate('evaluation');m=summary(records,threshold)
  audit=audit_correspondences(records);save(OUT/'evaluation_rankings.json',records)
  save(OUT/'evaluation_results.json',dict(**m,threshold=threshold,registered_identities=41,views_per_identity=72,
      detector_training_geometry_disjoint=True,query_geometry_ids='SYN090..099',known_registered_geometry_ids='SYN090..094',unknown_geometry_ids='SYN095..099',
      renderer='orthographic synthetic, approximately spherical berries; not a real-world accuracy claim',
      no_threshold_changes_after_evaluation=True,single_photo_input=True,query_circles_are_model_predictions=True,
      physical_correspondence_audit=audit,frozen_protocol=frozen))
  review(records,threshold);print('EVALUATION FINISHED',json.dumps(m),flush=True)

if __name__=='__main__':main()
