import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,cv2,numpy as np
from pathlib import Path
cv2.setNumThreads(2)
ROOT=Path(__file__).resolve().parent
snapshot=json.loads((ROOT.parent/'data/snapshot.json').read_text(encoding='utf-8'))
selected=[s for s in snapshot['samples'] if s['date']=='20260707' and s['bunch_id'] in ['GROW_02','FP2_03']]
out=ROOT/'qa';out.mkdir(exist_ok=True);records=[]
for s in selected:
    for j,path in enumerate(s['source_videos']):
        cap=cv2.VideoCapture(path);fps=cap.get(cv2.CAP_PROP_FPS);count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tiles=[]
        for t in np.linspace(.4,max(.4,count/fps-.5),12):
            cap.set(cv2.CAP_PROP_POS_MSEC,float(t*1000));ok,im=cap.read()
            if not ok:continue
            h,w=im.shape[:2];a=cv2.resize(im,(180,round(h*180/w)))
            text=np.full((26,a.shape[1],3),245,np.uint8)
            cv2.putText(text,f'{s["bunch_id"]}/{j+1} {t:.1f}s',(3,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(30,40,30),1)
            tiles.append(np.vstack([text,a]))
        cap.release()
        dest=out/f'{s["bunch_id"]}_video{j+1}_overview.jpg'
        cv2.imencode('.jpg',np.vstack([np.hstack(tiles[k:k+4]) for k in range(0,12,4)]))[1].tofile(str(dest))
        records.append(dict(identity=s['bunch_id'],video=j+1,path=path,fps=fps,duration=count/fps,width=w,height=h,overview=str(dest)))
        print(s['bunch_id'],j+1,count/fps,w,h,flush=True)
(ROOT/'video_inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
