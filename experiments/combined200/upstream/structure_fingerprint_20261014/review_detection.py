"""Conservative visible-fragment exclusion after the 40 raw overlays were viewed."""
import json
from pathlib import Path
import cv2,numpy as np
from detection import ROOT,overlay
from prepare_pilot import write_image,save

catalog=json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'));audit=[]
for record in catalog['records']:
    raw=record.get('raw_nodes',record['nodes']);median=float(np.median([n['radius'] for n in raw]))
    kept=[dict(n) for n in raw if n['radius']>=.55*median and n['confidence']>=.30]
    excluded=[dict(n,reason='small visible fragment or low segmentation confidence') for n in raw if n['radius']<.55*median or n['confidence']<.30]
    for i,n in enumerate(kept):n['id']=f'B{i+1:02d}'
    record.update(raw_nodes=raw,nodes=kept,excluded_nodes=excluded,checked=True,
                  source='GBISC CPU masks, raw 40 overlays visually reviewed; uncertain small fragments excluded, not claimed complete',
                  all_visible_berries_verified=False)
    image=cv2.imread(str(ROOT.parent/record['image']));write_image(ROOT/'qa'/(record['key']+'_nodes.jpg'),overlay(image,kept))
    save(ROOT/'data'/(record['key']+'.json'),record)
    audit.append(dict(key=record['key'],raw=len(raw),kept=len(kept),excluded=len(excluded)))
for identity in catalog['identities']:
    tiles=[]
    for record in [r for r in catalog['records'] if r['identity']==identity]:
        im=cv2.imread(str(ROOT/'qa'/(record['key']+'_nodes.jpg')));label=np.full((32,180,3),245,np.uint8)
        cv2.putText(label,record['key']+' / '+str(len(record['nodes'])),(3,20),cv2.FONT_HERSHEY_SIMPLEX,.38,(30,50,30),1)
        tiles.append(np.vstack([label,cv2.resize(im,(180,250))]))
    while len(tiles)%5:tiles.append(np.full_like(tiles[0],245))
    write_image(ROOT/'qa'/(identity+'_detections.jpg'),np.vstack([np.hstack(tiles[k:k+5]) for k in range(0,len(tiles),5)]))
save(ROOT/'catalog.json',catalog);save(ROOT/'qa/detection_review.json',dict(records=audit,all_raw_views_inspected=True,
    complete_instance_count_claimed=False,filter_rule='radius >= 0.55 median, confidence >=0.30; visible geometry only'))
print(json.dumps(audit),flush=True)
