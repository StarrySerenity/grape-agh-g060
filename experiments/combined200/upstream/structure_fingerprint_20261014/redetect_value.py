import json,shutil
import cv2,numpy as np
from detection import ROOT,detect_crop
from prepare_pilot import save

for name in ['catalog.json','pilot_results.json','stress_results.json']:
    target=ROOT/'qa'/('rgb_'+name)
    if not target.exists():shutil.copyfile(ROOT/name,target)
catalog=json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'))
for record in catalog['records']:
    im=cv2.imdecode(np.fromfile(ROOT.parent/record['image'],np.uint8),1)
    target=cv2.imdecode(np.fromfile(ROOT.parent/record['target'],np.uint8),0)
    raw=detect_crop(im,target);record['nodes']=raw;record['raw_nodes']=raw;record['checked']=False
    record['detector_input']='HSV value-equivalent max(R,G,B), normalize target 85th percentile; no hue/saturation'
    print(record['key'],len(raw),flush=True)
save(ROOT/'catalog.json',catalog)
