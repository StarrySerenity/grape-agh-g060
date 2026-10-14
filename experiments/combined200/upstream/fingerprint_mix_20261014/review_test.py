"""Visual audit of every fixed test query and actual top retrieved views; no tuning."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent))
from fingerprint_2d import imread,masked_crop
rows=json.loads((ROOT/'real_rows.json').read_text(encoding='utf-8'))
split=json.loads((ROOT/'real_protocol.json').read_text(encoding='utf-8'))
result=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
arrays={k:np.load(ROOT/'gallery'/k/'all_embeddings.npy') for k in ['mixed','real_only']}

def tile(index,label,correct=True):
    row=rows[index];crop,_=masked_crop(imread(row['image']),imread(row['mask'],True))
    ratio=min(110/crop.shape[1],140/crop.shape[0]);crop=cv2.resize(crop,(max(1,int(crop.shape[1]*ratio)),max(1,int(crop.shape[0]*ratio))))
    canvas=np.full((185,120,3),245,np.uint8);y=(140-crop.shape[0])//2;x=(120-crop.shape[1])//2;canvas[y:y+crop.shape[0],x:x+crop.shape[1]]=crop
    cv2.putText(canvas,label,(3,157),cv2.FONT_HERSHEY_SIMPLEX,.38,(30,60,30),1)
    cv2.putText(canvas,row['identity'],(3,175),cv2.FONT_HERSHEY_SIMPLEX,.4,(30,110,30) if correct else (20,20,200),1)
    return canvas

groups=[];audit=[]
for j,i in enumerate(split['test_query_indices']):
    trio=[tile(i,'QUERY')];record=dict(query=rows[i]['name'],identity=rows[i]['identity'])
    for key in ['mixed','real_only']:
        prediction=result['test'][key]['identity']['records'][j]['candidates'][0]['identity']
        candidates=[g for g in split['test_gallery_indices'] if rows[g]['identity']==prediction]
        best=max(candidates,key=lambda g:float(arrays[key][i]@arrays[key][g]))
        trio.append(tile(best,'MIXED' if key=='mixed' else 'REAL ONLY',prediction==rows[i]['identity']))
        record[key]=dict(prediction=prediction,best_view=best)
    groups.append(np.hstack(trio));audit.append(record)
sheet=np.vstack([np.hstack(groups[j:j+4]) for j in range(0,len(groups),4)])
cv2.imencode('.jpg',sheet,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(ROOT/'qa/all_test_queries.jpg'))
(ROOT/'qa/test_visual_records.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('ALL_TEST_QUERIES_RENDERED',len(audit))
