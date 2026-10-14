"""Loopback-only synthetic single-photo demo. Never modifies original grape data."""
from consumer_engine import ROOT,OUT,read,detect_image,query_payload,match_payload,gallery_image
import argparse,base64,json,threading,sys,math
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs,unquote
import cv2,numpy as np

LOCK=threading.Lock()

def encoded(im):return base64.b64encode(cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,94])[1]).decode('ascii')

def sample_payload(key):
 row=next((r for r in read(OUT/'query_manifest.json') if r['key']==key),None)
 if row is None:raise ValueError('Unknown sample')
 im=cv2.imdecode(np.fromfile(str(ROOT/row['image']),np.uint8),1)
 mask=cv2.imdecode(np.fromfile(str(ROOT/row['mask']),np.uint8),0)
 result=detect_image(im,mask)
 result.update(expected_identity=row['identity'] if row['in_database'] else None,expected_in_database=row['in_database'],sample_key=key,
               message='只把图像和目标区域送入检测器；身份匹配只接收检测出的圆心与半径。')
 return result

def random_payload(payload):
 seed=payload.get('seed',int(np.random.default_rng().integers(0,2**31)))
 if not isinstance(seed,(int,float)) or not np.isfinite(seed):raise ValueError('Invalid seed')
 seed=int(seed)%2**31;rng=np.random.default_rng(seed)
 known=bool(payload.get('in_database',True));index=int(rng.integers(90,95) if known else rng.integers(95,100))
 identity=f'SYN_{index:03d}';model=read(ROOT.parent/'fingerprint_mix_20261014/synthetic'/identity/'model.json')
 sys.path.insert(0,str(ROOT.parent/'fingerprint_mix_20261014'))
 from render_synthetic import render
 yaw=float(rng.uniform(0,2*np.pi));pitch=float(rng.uniform(-.25,.25))
 image,mask=render(model,yaw,pitch,appearance_seed=seed+300001)
 M=cv2.getRotationMatrix2D((144,144),float(rng.uniform(-14,14)),float(rng.uniform(.92,1.05)))
 image=cv2.warpAffine(image,M,(288,288),borderValue=(128,128,128));mask=cv2.warpAffine(mask,M,(288,288),flags=cv2.INTER_NEAREST)
 hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);hsv[...,0]=(hsv[...,0].astype(int)+int(rng.integers(-25,26)))%180
 image=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);image=np.clip(image.astype(float)*rng.uniform(.70,1.25),0,255).astype(np.uint8);image[mask==0]=128
 image=cv2.resize(image,(256,256));mask=cv2.resize(mask,(256,256),interpolation=cv2.INTER_NEAREST)
 # Round-trip the displayed JPEG, so the image the model sees is inspectable.
 image=cv2.imdecode(np.frombuffer(base64.b64decode(encoded(image)),np.uint8),1)
 result=detect_image(image,mask)
 result.update(expected_identity=identity if known else None,expected_in_database=known,sample_key=f'random_{seed}',seed=seed,
               random_query=True,matching_does_not_receive_identity=True)
 return result

class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(OUT),**kwargs)
 def log_message(self,*args):pass
 def allowed(self):
  host=self.headers.get('Host');origin=self.headers.get('Origin')
  return host in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}') and (not origin or origin=='http://'+host)
 def end_headers(self):
  self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');super().end_headers()
 def js(self,status,value):
  body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode();self.send_response(status)
  self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def do_GET(self):
  if not self.allowed():return self.js(403,dict(error='Local access only'))
  url=urlsplit(self.path);path=unquote(url.path)
  if path=='/health':return self.js(200,dict(app='synthetic-grape-single-photo',threads=2,device='cpu'))
  if path=='/api/info':
   return self.js(200,dict(registered_identities=41,views_per_identity=72,
     detector=read(OUT/'detector_summary.json')['validation'],
     evaluation=read(OUT/'evaluation_results.json') if (OUT/'evaluation_results.json').exists() else None,
     samples=[{k:r[k] for k in ['key','identity','in_database','split','image']} for r in read(OUT/'query_manifest.json')]))
  if path=='/api/template':
   try:
    with LOCK:im,row=gallery_image(parse_qs(url.query).get('key',[''])[0])
    return self.js(200,dict(image_base64=encoded(im),**row))
   except (ValueError,KeyError) as e:return self.js(400,dict(error=str(e)))
  resolved=(OUT/path.lstrip('/')).resolve()
  if resolved!=OUT.resolve() and OUT.resolve() not in resolved.parents:return self.js(403,dict(error='Invalid path'))
  return super().do_GET()
 def do_POST(self):
  if not self.allowed():return self.js(403,dict(error='Local access only'))
  try:
   length=int(self.headers.get('Content-Length',0))
   if not 0<length<=14*1024*1024:raise ValueError('Request too large or empty')
   payload=json.loads(self.rfile.read(length))
   if not isinstance(payload,dict):raise ValueError('Expected a JSON object')
   path=urlsplit(self.path).path
   with LOCK:
    if path=='/api/sample':result=sample_payload(payload.get('key'))
    elif path=='/api/random':result=random_payload(payload)
    elif path=='/api/detect':result=query_payload(payload)
    elif path=='/api/match':
     # Whitelist geometry; expected labels/sample IDs are never passed to ranking.
     result=match_payload(dict(nodes=payload.get('nodes',[])))
     if result.get('download'):result['download']='/'+result['download'].split('synthetic_consumer/',1)[1]
    else:return self.js(404,dict(error='Unknown operation'))
   return self.js(200,result)
  except (ValueError,TypeError,KeyError,OverflowError) as e:return self.js(400,dict(error=str(e)))
  except Exception as e:
   print(type(e).__name__,str(e),file=sys.stderr,flush=True);return self.js(500,dict(error='Processing failed; see local server log'))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8792);args=parser.parse_args()
 print(f'http://127.0.0.1:{args.port}/',flush=True)
 ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
