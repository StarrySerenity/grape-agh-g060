"""Version-two single-photo engine, isolated weights/gallery/query outputs."""
from data import *
import base64,time,uuid,torch
from functools import lru_cache
from circle_network import CircleDetector,decode
from train_circles import normalize
from prepare_circle_data import letterbox
from matching_fast import match_view,normalized,descriptor
from consumer_engine import accepted
torch.set_num_threads(2)

@lru_cache(maxsize=2)
def network(version='new200'):
 path=OUT/'detector_best.pt' if version=='new200' else LEGACY/'synthetic_consumer/detector_best.pt'
 if version not in ['new200','old100']:raise ValueError('Unknown model version')
 net=CircleDetector(pretrained=False);net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);net.eval();return net

def detect_image(image,mask,crop_target=False,version='new200'):
 if image is None or image.ndim!=3 or max(image.shape[:2])>6000:raise ValueError('Invalid image')
 if mask.shape!=image.shape[:2] or not np.any(mask):raise ValueError('Empty or mismatched target mask')
 x0=y0=0;x1=image.shape[1];y1=image.shape[0]
 if crop_target:
  yy,xx=np.where(mask>0);px=max(2,round(np.ptp(xx)*.12));py=max(2,round(np.ptp(yy)*.12));x0=max(0,int(xx.min()-px));x1=min(image.shape[1],int(xx.max()+px+1));y0=max(0,int(yy.min()-py));y1=min(image.shape[0],int(yy.max()+py+1))
 crop=image[y0:y1,x0:x1];target=mask[y0:y1,x0:x1];isolated=np.where(target[...,None]>0,crop,128);prepared,_,fg,transform=letterbox(isolated,[],target)
 with torch.no_grad():pred=network(version)(normalize(prepared)[None]);nodes=decode(pred,threshold=.25)
 nodes=[dict(n,id=f'D{i:02d}') for i,n in enumerate(nodes) if 0<=n['x']<256 and 0<=n['y']<256 and fg[int(n['y']),int(n['x'])]>0]
 return dict(nodes=nodes,width=256,height=256,image_base64=base64.b64encode(cv2.imencode('.jpg',prepared)[1]).decode('ascii'),
     source='synthetic circle-center detector; supplied target mask; geometry matching',model_version=version,
     transform=dict(**transform,crop_offset=[x0,y0]),input_circle_threshold=.25)

def query_payload(payload):
 binary=base64.b64decode(payload.get('image_base64',''),validate=True)
 if len(binary)>10*1024*1024:raise ValueError('Image exceeds 10 MiB')
 im=cv2.imdecode(np.frombuffer(binary,np.uint8),1)
 if im is None or max(im.shape[:2])>6000:raise ValueError('Invalid image')
 polygon=np.asarray(payload.get('polygon'),float)
 if polygon.ndim!=2 or polygon.shape[1]!=2 or not 3<=len(polygon)<=500 or not np.isfinite(polygon).all():raise ValueError('Circle target bunch first')
 if (polygon<0).any() or (polygon[:,0]>=im.shape[1]).any() or (polygon[:,1]>=im.shape[0]).any():raise ValueError('Polygon outside image')
 mask=np.zeros(im.shape[:2],np.uint8);cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255)
 if (mask>0).sum()<100:raise ValueError('Target polygon too small')
 return detect_image(im,mask,crop_target=bool(payload.get('crop_target',False)))

@lru_cache(maxsize=1)
def gallery():return read(OUT/'gallery.json')

def rank_nodes(nodes):
 normalized(nodes);start=time.time();data=gallery();results=[]
 if len(nodes)<6:return dict(status='insufficient',candidates=[],identity_confirmed=False,message='可靠果粒不足。',elapsed_seconds=0)
 for identity in data['identities']:
  options=[]
  for rec in data['records']:
   if rec['identity']==identity:options.append(dict(key=rec['key'],identity=identity,**match_view(nodes,rec['nodes'])))
  options.sort(key=lambda r:-r['score']);best=options[0];best['view_score']=best['score'];best['supporting_view_score']=options[1]['score'];best['score']=.85*best['view_score']+.15*best['supporting_view_score'];results.append(best)
 results.sort(key=lambda r:-r['score']);margin=results[0]['score']-results[1]['score']
 return dict(status='ranked',candidates=results,margin=float(margin),elapsed_seconds=time.time()-start,query_node_count=len(nodes),identity_confirmed=False,
   geometry_only=True,uses_rgb_for_matching=False,compared_views_per_identity=72,registered_identities=len(data['identities']),
   fingerprint=dict(type='visible berry center/radius geometry with partial correspondences',nodes=nodes,auxiliary_vector=descriptor(nodes).tolist(),vector_alone_not_identity=True))

def match_payload(payload):
 result=rank_nodes(payload.get('nodes',[]));threshold=read(OUT/'decision_threshold.json');passed=accepted(result,threshold)
 result.update(status='synthetic_match' if passed else 'uncertain',matched_identity=result['candidates'][0]['identity'] if passed else None,
   decision='仿真库内候选通过验证' if passed else '未找到足够证据',model_version='new200',identity_confirmed=False,threshold=threshold)
 folder=OUT/'uploads'/uuid.uuid4().hex;folder.mkdir(parents=True,exist_ok=True);save(folder/'fingerprint.json',result);result['download']=(folder/'fingerprint.json').relative_to(ROOT).as_posix();return result

def gallery_image(key):
 row=next((r for r in gallery()['records'] if r['key']==key),None)
 if row is None:raise ValueError('Unknown gallery view')
 from render_synthetic import render
 im,_=render(model(row['identity']),row['yaw'],row['pitch'],appearance_seed=130019)
 return cv2.resize(im,(256,256)),row
