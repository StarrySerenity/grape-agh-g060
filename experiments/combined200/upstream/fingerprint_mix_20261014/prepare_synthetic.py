"""Generate one foreground review group; no unattended queue or training launch."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'synthetic_bunch'))
from paper_grammar import generate_paper_bunch
from mature_geometry import mature_bunch
from render_synthetic import render

def prepare_identity(index):
    identity=f'SYN_{index:03d}';out=ROOT/'synthetic'/identity;out.mkdir(parents=True,exist_ok=True)
    if (out/'records.json').exists():
        print(identity,'already prepared',flush=True);return json.loads((out/'records.json').read_text(encoding='utf-8'))
    seed=900001+index*977;rng=np.random.default_rng(seed);profile=['cone','shoulder','curved'][index%3];count=int(rng.integers(36,61));start=time.time();failures=[]
    for trial in range(4):
        try:
            source=generate_paper_bunch(seed=seed+trial*200003,count=count,profile=profile,density='compact')
            model=mature_bunch(source)
            if model['maturity']['after_exposure']['mean']>.23:raise RuntimeError('Too much exposed internal stem')
            break
        except RuntimeError as e:failures.append(dict(trial=trial,error=str(e)));print(identity,'retry',trial,flush=True)
    else:
        (out/'failure.json').write_text(json.dumps(failures,indent=2),encoding='utf-8');raise RuntimeError(identity+' failed morphology QA')
    geometry_hash=model['geometry_hash'];model['training_identity']=identity
    (out/'model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
    rows=[];thumbnails=[]
    for view in range(24):
        yaw=(view%8)*math.pi/4+float(rng.uniform(-.06,.06));pitch=[-.20,0,.20][view//8]
        im,mask=render(model,yaw,pitch,appearance_seed=300019+index*1009+view*71)
        if (mask>0).sum()<4000:raise RuntimeError('Insufficient visible target pixels')
        image=out/f'view_{view:02d}.jpg';maskfile=out/f'view_{view:02d}_mask.png'
        cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(86,96))])[1].tofile(str(image));cv2.imencode('.png',mask)[1].tofile(str(maskfile))
        rows.append(dict(identity=identity,name=image.name,image=str(image),mask=str(maskfile),yaw=yaw,pitch=pitch,
                         geometry_hash=geometry_hash,view=view,capture_kind='synthetic',image_sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
        if view<8:thumbnails.append(cv2.resize(im,(128,128)))
    record=dict(identity=identity,geometry_hash=geometry_hash,rows=rows,models=count,source_seed=seed,failures=failures,
                geometry_qa=model['qa'],exposure=model['maturity']['after_exposure'],renderer='CPU depth-buffer spheres/capsules; illustrative material')
    (out/'records.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    cv2.imencode('.jpg',np.hstack(thumbnails),[cv2.IMWRITE_JPEG_QUALITY,92])[1].tofile(str(out/'review.jpg'))
    print(identity,count,'berries',24,'views',round(time.time()-start,1),'seconds',flush=True)
    return record

def review_group(start,count):
    records=[]
    for i in range(start,start+count):records.append(prepare_identity(i))
    tiles=[]
    for rec in records:
        im=cv2.imdecode(np.fromfile(ROOT/'synthetic'/rec['identity']/'review.jpg',np.uint8),cv2.IMREAD_COLOR)
        label=np.full((24,im.shape[1],3),245,np.uint8);cv2.putText(label,rec['identity']+' / '+str(rec['models'])+' berries',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(25,50,30),1)
        tiles.append(np.vstack([label,im]))
    qa=ROOT/'qa';qa.mkdir(exist_ok=True);path=qa/f'group_{start:03d}.jpg'
    cv2.imencode('.jpg',np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,93])[1].tofile(str(path))
    print('REVIEW',path,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',type=int,required=True);parser.add_argument('--count',type=int,default=5);args=parser.parse_args()
    if args.count>10:raise ValueError('Generate at most ten identities before the next explicit review')
    review_group(args.start,args.count)
