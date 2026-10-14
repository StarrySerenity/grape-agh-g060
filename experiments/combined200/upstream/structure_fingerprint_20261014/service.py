"""Local geometry API; all source catalogs stay immutable during manual edits."""
import base64,json,uuid
import cv2,numpy as np
from detection import ROOT,prepare_crop,detect_crop
from geometry import match_graphs,fingerprint,arrays
from prepare_pilot import save

def catalog():return json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'))

def match(payload):
    nodes=payload.get('nodes');arrays(nodes)
    if len(nodes)<6:return dict(status='insufficient',candidates=[],fingerprint=None,identity_confirmed=False,message='至少需要六个可见果粒，且应分布在多个区域。')
    records=catalog()['records'];overrides=payload.get('overrides',{})
    if not isinstance(overrides,dict) or len(overrides)>100:raise ValueError('Invalid edited gallery')
    allowed={r['key'] for r in records if r['capture_kind']=='video_frame'}
    for key,value in overrides.items():
        if key not in allowed:raise ValueError('Unknown gallery view')
        arrays(value)
    candidates=[]
    for entry in records:
        if entry['capture_kind']!='video_frame':continue
        m=match_graphs(nodes,overrides.get(entry['key'],entry['nodes']))
        candidates.append(dict(key=entry['key'],identity=entry['identity'],**m))
    best=[max([r for r in candidates if r['identity']==identity],key=lambda r:r['score']) for identity in catalog()['identities']]
    best.sort(key=lambda r:-r['score']);margin=best[0]['score']-best[1]['score']
    chosen=payload.get('gallery_key');pair=next((c for c in candidates if c['key']==chosen),None)
    result=dict(status='candidate_only',fingerprint=fingerprint(nodes),candidates=best,selected_pair=pair,
        margin=margin,identity_confirmed=False,uses_rgb_for_matching=False,geometry_only=True,
        message='结构候选，需要核对对应果粒；分数不是身份概率，未校准自动确认阈值。',
        edited_gallery_views=list(overrides),
        gallery_overrides={key:[dict(x=n['x'],y=n['y'],radius=n['radius'],id=n.get('id')) for n in value] for key,value in overrides.items()},
        query_nodes=[dict(x=n['x'],y=n['y'],radius=n['radius'],id=n.get('id')) for n in nodes])
    out=ROOT/'queries'/uuid.uuid4().hex;out.mkdir(parents=True,exist_ok=True)
    save(out/'structure_fingerprint.json',result);result['download']=(out/'structure_fingerprint.json').relative_to(ROOT.parent).as_posix()
    return result

def detect(payload):
    binary=base64.b64decode(payload.get('image_base64',''),validate=True)
    if len(binary)>10*1024*1024:raise ValueError('Image exceeds 10 MiB')
    image=cv2.imdecode(np.frombuffer(binary,np.uint8),cv2.IMREAD_COLOR)
    if image is None or max(image.shape[:2])>6000:raise ValueError('Invalid image')
    polygon=np.asarray(payload.get('polygon'),float)
    if polygon.ndim!=2 or polygon.shape[1]!=2 or not 3<=len(polygon)<=500 or not np.isfinite(polygon).all():raise ValueError('Circle the target bunch with a polygon')
    if (polygon<0).any() or (polygon[:,0]>=image.shape[1]).any() or (polygon[:,1]>=image.shape[0]).any():raise ValueError('Polygon outside image')
    mask=np.zeros(image.shape[:2],np.uint8);cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255)
    crop,target=prepare_crop(image,mask);raw=detect_crop(crop,target);median=np.median([n['radius'] for n in raw]) if raw else 1
    nodes=[n for n in raw if n['radius']>=.55*median and n['confidence']>=.30]
    for i,n in enumerate(nodes):n['id']=f'B{i+1:02d}'
    return dict(nodes=nodes,width=crop.shape[1],height=crop.shape[0],image_base64=base64.b64encode(cv2.imencode('.jpg',crop)[1]).decode(),
                source='CPU GBISC on normalized value channel; automatic visible estimates, manually correct if needed')
