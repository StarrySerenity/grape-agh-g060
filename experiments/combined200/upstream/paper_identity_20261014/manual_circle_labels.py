"""Visual, manually entered circle estimates; explicitly not instrument truth."""
import json,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'circle_training'
LABELS={
 'FP2_09_193':[(147,38,18),(88,61,18),(122,66,18),(96,84,18),(152,86,20),(126,100,17),(98,124,19),(126,139,20),(159,124,21),(107,164,21),(140,171,21),(169,186,18),(122,194,18),(113,218,17),(150,211,17)],
 'FP3_02_271':[(59,47,26),(109,52,26),(157,48,27),(82,91,26),(131,88,25),(173,89,23),(194,107,22),(104,126,25),(154,128,27),(202,147,22),(85,157,23),(126,175,26),(175,170,26),(113,208,23),(149,215,24)],
 'GROW_03_745':[(82,52,18),(108,53,19),(129,32,21),(163,51,20),(180,72,16),(139,78,19),(105,84,18),(77,102,18),(116,103,19),(158,111,21),(168,135,17),(130,136,20),(103,128,21),(122,173,19),(162,165,19),(145,188,19),(142,215,20)],
 'manual_grow_B_0':[(90,47,17),(93,70,17),(137,57,22),(149,85,17),(112,91,22),(83,92,15),(90,119,20),(126,121,18),(154,117,20),(108,149,22),(144,154,21),(86,151,17),(106,182,24),(146,189,24),(91,214,21),(131,225,19),(167,100,18)],
 'manual_fp203_A_20':[(88,23,14),(119,27,15),(135,19,13),(93,37,14),(111,47,15),(133,51,14),(81,56,16),(146,42,13),(126,77,17),(99,86,16),(80,101,16),(122,109,17),(140,101,13),(82,122,15),(70,141,14),(88,153,17),(108,133,17),(135,129,13),(114,158,15),(82,188,15),(108,177,17)]}

def main():
 records=[];qa=OUT/'qa'
 for name,coords in LABELS.items():
  role='validation_manual_visible_subset' if not name.startswith('manual_') else 'train_manual_visible_subset'
  identity=name.rsplit('_',1)[0] if not name.startswith('manual_') else 'GROW_02' if 'grow_B' in name else 'FP2_03'
  nodes=[dict(x=float(x),y=float(y),radius=float(r),visibility=1.,id=f'M{i:02d}') for i,(x,y,r) in enumerate(coords)]
  path=OUT/(name+'.jpg');im=cv2.imdecode(np.fromfile(str(path),np.uint8),1)
  for n in nodes:
   xy=(round(n['x']),round(n['y']));cv2.circle(im,xy,round(n['radius']),(30,220,70),1);cv2.circle(im,xy,1,(0,0,230),-1)
   cv2.putText(im,n['id'],xy,cv2.FONT_HERSHEY_SIMPLEX,.25,(255,255,255),1)
  cv2.imencode('.jpg',cv2.resize(im,(768,768)),[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(qa/(name+'_manual.jpg')))
  records.append(dict(image=str(path),identity=identity,domain='real_manual',role=role,nodes=nodes,
      label_source='assistant visually entered approximate circle centers/radii on coordinate grid, not instrument-measured truth',
      annotation_complete=False,visibility_target_is_proxy=True,
      evaluation='Recall and center error on explicitly annotated reasonably visible subset; unmatched predictions are unverified, not automatically false positives'))
 (OUT/'manual_labels.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Manual estimates',sum(len(v['nodes']) for v in records),'validation',sum(len(v['nodes']) for v in records if v['role'].startswith('validation')))

if __name__=='__main__':main()
