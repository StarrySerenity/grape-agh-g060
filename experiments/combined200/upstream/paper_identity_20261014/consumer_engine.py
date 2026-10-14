"""Single-photo interface: circle detection, multi-view ranking, calibrated abstention."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import base64,json,time,uuid,sys
from pathlib import Path
from functools import lru_cache
import cv2,numpy as np,torch
from circle_network import CircleDetector,decode
from consumer_geometry import descriptor,match_view,normalized
from train_circles import normalize
from prepare_circle_data import letterbox
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'synthetic_consumer'
cv2.setNumThreads(2);torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
@lru_cache(maxsize=1)
def gallery():
    data=read(OUT/'gallery.json');rows=data['records'];vectors=np.array([r['descriptor'] for r in rows])
    return data,vectors
@lru_cache(maxsize=1)
def network():
    net=CircleDetector(pretrained=False);state=torch.load(OUT/'detector_best.pt',map_location='cpu',weights_only=False);net.load_state_dict(state['model']);net.eval();return net

def detect_image(image,mask,crop_target=False):
    if image is None or image.ndim!=3 or max(image.shape[:2])>6000:raise ValueError('Invalid image')
    if mask.shape!=image.shape[:2] or not np.any(mask):raise ValueError('Empty or mismatched target mask')
    x0=y0=0;x1=image.shape[1];y1=image.shape[0]
    if crop_target:
        yy,xx=np.where(mask>0);padx=max(2,round(np.ptp(xx)*.12));pady=max(2,round(np.ptp(yy)*.12))
        x0=max(0,int(xx.min()-padx));x1=min(image.shape[1],int(xx.max()+padx+1));y0=max(0,int(yy.min()-pady));y1=min(image.shape[0],int(yy.max()+pady+1))
    crop=image[y0:y1,x0:x1];target=mask[y0:y1,x0:x1];isolated=np.where(target[...,None]>0,crop,128)
    prepared,_,fg,transform=letterbox(isolated,[],target)
    with torch.no_grad():pred=network()(normalize(prepared)[None]);nodes=decode(pred,threshold=.25)
    nodes=[dict(n,id=f'D{i:02d}') for i,n in enumerate(nodes) if 0<=n['x']<256 and 0<=n['y']<256 and fg[int(n['y']),int(n['x'])]>0]
    return dict(nodes=nodes,width=256,height=256,image_base64=base64.b64encode(cv2.imencode('.jpg',prepared)[1]).decode('ascii'),
                source='synthetic-only trained circle center/radius heads; target mask supplied, not automatic whole-bunch segmentation',
                transform=dict(**transform,crop_offset=[x0,y0]),input_circle_threshold=.25)

def query_payload(payload):
    binary=base64.b64decode(payload.get('image_base64',''),validate=True)
    if len(binary)>10*1024*1024:raise ValueError('Image exceeds 10 MiB')
    im=cv2.imdecode(np.frombuffer(binary,np.uint8),1)
    if im is None or max(im.shape[:2])>6000:raise ValueError('Invalid image')
    polygon=np.asarray(payload.get('polygon'),float)
    if polygon.ndim!=2 or polygon.shape[1]!=2 or not 3<=len(polygon)<=500 or not np.isfinite(polygon).all():raise ValueError('Circle the target bunch first')
    if (polygon<0).any() or (polygon[:,0]>=im.shape[1]).any() or (polygon[:,1]>=im.shape[0]).any():raise ValueError('Polygon outside image')
    mask=np.zeros(im.shape[:2],np.uint8);cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255)
    if (mask>0).sum()<100:raise ValueError('Target polygon too small')
    return detect_image(im,mask,crop_target=bool(payload.get('crop_target',False)))

def rank_nodes(nodes,views_per_identity=None):
    normalized(nodes)
    start=time.time();data,vectors=gallery();rows=data['records']
    if len(nodes)<6:return dict(status='insufficient',candidates=[],identity_confirmed=False,message='可见可靠果粒不足，无法判断。',elapsed_seconds=0.)
    vector=descriptor(nodes);dist=np.linalg.norm(vectors-vector[None],axis=1)
    results=[]
    for identity in data['identities']:
        indices=np.array([i for i,r in enumerate(rows) if r['identity']==identity])
        # Global shape statistics are not safe for pruning a partial visible side.
        # The consumer path checks every registered direction; optional pruning
        # remains only for timing/diagnostics, never for the reported experiment.
        choice=indices if views_per_identity is None else indices[np.argsort(dist[indices])[:views_per_identity]]
        options=[]
        for i in choice:
            rec=rows[int(i)];matched=match_view(nodes,rec['nodes']);options.append(dict(key=rec['key'],identity=identity,**matched))
        options.sort(key=lambda r:-r['score']);best=options[0]
        best['view_score']=best['score'];best['supporting_view_score']=options[1]['score'] if len(options)>1 else 0
        best['score']=.85*best['view_score']+.15*best['supporting_view_score']
        results.append(best)
    results.sort(key=lambda r:-r['score']);margin=results[0]['score']-results[1]['score'] if len(results)>1 else results[0]['score']
    return dict(status='ranked',candidates=results,margin=float(margin),elapsed_seconds=time.time()-start,
                query_node_count=len(nodes),identity_confirmed=False,geometry_only=True,uses_rgb_for_matching=False,
                compared_views_per_identity=72 if views_per_identity is None else views_per_identity,registered_identities=len(data['identities']),
                fingerprint=dict(type='visible berry center/radius geometry with partial correspondences',auxiliary_vector=vector.tolist(),dimension=len(vector),
                                 vector_alone_not_identity=True,nodes=nodes))

def accepted(result,threshold):
    if not result.get('candidates'):return False
    b=result['candidates'][0]
    return bool(b['score']>=threshold['minimum_score'] and result['margin']>=threshold['minimum_margin'] and len(b['pairs'])>=threshold['minimum_pairs'] and b['query_coverage']>=threshold['minimum_query_coverage'])

def match_payload(payload):
    result=rank_nodes(payload.get('nodes',[]))
    threshold=read(OUT/'decision_threshold.json') if (OUT/'decision_threshold.json').exists() else None
    if threshold and accepted(result,threshold):
        result.update(status='synthetic_match',matched_identity=result['candidates'][0]['identity'],decision='仿真库内候选通过验证',
                      identity_confirmed=False,message='在本仿真试验约定下通过几何验证；实拍身份不自动确认。')
    else:result.update(status='uncertain',matched_identity=None,decision='未找到足够证据',identity_confirmed=False,
                       message='可能不在库中，也可能角度、遮挡或定位误差过大；不能强行认作第一候选。')
    result['threshold']=threshold
    out=OUT/'uploads'/uuid.uuid4().hex;out.mkdir(parents=True,exist_ok=True)
    (out/'fingerprint.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    result['download']=(out/'fingerprint.json').relative_to(ROOT).as_posix()
    return result

def gallery_image(key):
    data,_=gallery();row=next((r for r in data['records'] if r['key']==key),None)
    if row is None:raise ValueError('Unknown gallery view')
    sys.path.insert(0,str(ROOT.parent/'fingerprint_mix_20261014'))
    from render_synthetic import render
    model=read(ROOT.parent/'fingerprint_mix_20261014/synthetic'/row['identity']/'model.json')
    im,mask=render(model,row['yaw'],row['pitch'],appearance_seed=130019)
    return cv2.resize(im,(256,256)),row
