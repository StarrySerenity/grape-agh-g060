"""Prepare coordinate grids for manual circle annotation and finalize stable windows."""
import json,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'circle_training'
def main():
 rows=json.loads((OUT/'real_manifest.json').read_text(encoding='utf-8'))
 # Reject sampled windows with large between-frame changes, not just low blur.
 for row in rows:
  if row['role']=='train_weak' and (row['window_max_normalized_center_motion']>.13 or row['window_max_log_extent_change']>.35):
   row['role']='excluded_motion';row['exclusion_reason']='sampled window motion/scale change too large for this stable-frame training pilot'
 (OUT/'real_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 picks=[]
 for identity in ['FP2_09','FP3_02','GROW_03']:
  row=next(v for v in rows if v['identity']==identity);picks.append(row)
 for key,frame in [('grow_B',0),('fp203_A',20)]:
  info=json.loads((OUT/f'manual_{key}_{frame}.json').read_text(encoding='utf-8'));picks.append(info)
 for row in picks:
  im=cv2.imdecode(np.fromfile(row['image'],np.uint8),1);im=cv2.resize(im,(768,768))
  for coordinate in range(0,256,16):
   c=coordinate*3;cv2.line(im,(c,0),(c,768),(210,210,210),1);cv2.line(im,(0,c),(768,c),(210,210,210),1)
   cv2.putText(im,str(coordinate),(c+2,14),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
   cv2.putText(im,str(coordinate),(1,c+12),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
  name=Path(row['image']).stem;cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(OUT/'qa'/f'grid_{name}.jpg'))
 print('Stable weak-label frames',sum(r['role']=='train_weak' for r in rows),'excluded',sum(r['role']=='excluded_motion' for r in rows))
 print('manual sources',[(r.get('identity',r.get('key')),r['image']) for r in picks])

if __name__=='__main__':main()
