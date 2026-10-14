"""Use saved JPEG pixels through the same detection path exposed by the page."""
from consumer_engine import OUT,ROOT,read,detect_image
import cv2,numpy as np,json,hashlib

def main():
 rows=read(OUT/'query_manifest.json')
 archive=OUT/'history/pre_jpeg_detection';archive.mkdir(parents=True,exist_ok=True)
 if not (archive/'query_manifest.json').exists():
  (archive/'query_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 for r in rows:
  im=cv2.imdecode(np.fromfile(str(ROOT/r['image']),np.uint8),1)
  mask=cv2.imdecode(np.fromfile(str(ROOT/r['mask']),np.uint8),0)
  detected=detect_image(im,mask)
  r['nodes']=detected['nodes'];r['detection_pipeline']='saved JPEG -> supplied foreground mask -> no target recrop -> shared consumer_engine.detect_image'
  r['detector_checkpoint_sha256']=hashlib.sha256((OUT/'detector_best.pt').read_bytes()).hexdigest()
  print(r['key'],len(r['nodes']),flush=True)
 (OUT/'query_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

if __name__=='__main__':main()
