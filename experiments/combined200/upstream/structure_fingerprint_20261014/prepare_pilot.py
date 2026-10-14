import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np
from detection import prepare_crop,detect_crop,overlay,ROOT,WEIGHTROOT
sys.path.insert(0,str(ROOT.parent))
from fingerprint_2d import imread

def write_image(path,image):cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(str(path))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def prepare(compare=False):
    rows=json.loads((ROOT.parent/'fingerprint_2d/all_records.json').read_text(encoding='utf-8'));identities=['FP2_01','FP2_03','FP3_03','GROW_02']
    out=ROOT/'data';out.mkdir(exist_ok=True);qa=ROOT/'qa';qa.mkdir(exist_ok=True);records=[];pieces=[];start=time.time()
    for identity in identities:
        pool=[(i,r) for i,r in enumerate(rows) if r['identity']==identity]
        photos=[p for p in pool if p[1]['capture_kind']=='photo'][:2];videos=[p for p in pool if p[1]['capture_kind']=='video_frame']
        chosen=photos+[videos[k] for k in np.linspace(0,len(videos)-1,8,dtype=int)]
        for j,(index,row) in enumerate(chosen):
            crop,target=prepare_crop(imread(row['image']),imread(row['mask'],True))
            key=f'{identity}_{j:02d}';nodes=detect_crop(crop,target)
            write_image(out/(key+'.jpg'),crop);cv2.imencode('.png',target)[1].tofile(str(out/(key+'_target.png')))
            write_image(qa/(key+'_nodes.jpg'),overlay(crop,nodes))
            record=dict(key=key,identity=identity,name=row['name'],capture_kind=row['capture_kind'],video_group=row['video_group'],
                image='structure_fingerprint_20261014/data/'+key+'.jpg',target='structure_fingerprint_20261014/data/'+key+'_target.png',
                nodes=nodes,width=crop.shape[1],height=crop.shape[0],source_index=index,
                source_image_sha256=row['image_sha256'],source_mask_sha256=row['mask_sha256'],
                checked=False,source='CPU pretrained berry masks; visible centroid, no manual correction yet')
            records.append(record);save(out/(key+'.json'),record)
            tile=cv2.resize(overlay(crop,nodes),(180,250));label=np.full((32,180,3),245,np.uint8)
            cv2.putText(label,key+' / '+str(len(nodes)),(3,20),cv2.FONT_HERSHEY_SIMPLEX,.38,(30,50,30),1)
            pieces.append(np.vstack([label,tile]));print(key,len(nodes),'nodes',round(time.time()-start),flush=True)
            if compare and j==0:
                alternative=detect_crop(crop,target,'wine_berries_yolov8nseg_v13')
                write_image(qa/(key+'_nseg.jpg'),overlay(crop,alternative));print('nseg',identity,len(alternative),flush=True)
        sheet=np.vstack([np.hstack(pieces[-10:-5]),np.hstack(pieces[-5:])])
        write_image(qa/(identity+'_detections.jpg'),sheet)
    save(ROOT/'catalog.json',dict(records=records,identities=identities,pilot_only=True,independent_holdout=False,
             matcher_inputs='geometry only',weights=str(WEIGHTROOT/'gbisc_yolov8m/weights/best.pt')))
    print('PILOT_READY',len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--compare',action='store_true');a=p.parse_args();prepare(a.compare)
