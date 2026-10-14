import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'synthetic_bunch'))
def geometry_hash(model):
    geometry={k:model[k] for k in ['models','stems','nodes']}
    return hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def audit():
    records=[];hashes=set();minimum=1e9;seen=set();total_berries=0
    for i in range(100):
        folder=ROOT/'synthetic'/f'SYN_{i:03d}'
        model=json.loads((folder/'model.json').read_text(encoding='utf-8'))
        rec=json.loads((folder/'records.json').read_text(encoding='utf-8'))
        assert model['geometry_hash']==rec['geometry_hash']==geometry_hash(model)
        assert rec['geometry_hash'] not in hashes;hashes.add(rec['geometry_hash'])
        xyz=np.array([b['center'] for b in model['models']]);radius=np.array([b['radius'] for b in model['models']])
        distance=np.linalg.norm(xyz[:,None]-xyz[None],axis=2);ratio=distance/(radius[:,None]+radius[None]);np.fill_diagonal(ratio,100)
        assert ratio.min()>.995 and rec['geometry_qa']['unrelated_stem_intersections']==0
        assert len(rec['rows'])==24 and rec['geometry_qa']['connected_berries']==len(xyz)
        total_berries+=len(xyz)
        for row in rec['rows']:
            binary=Path(row['image']).read_bytes();h=hashlib.sha256(binary).hexdigest()
            assert h==row['image_sha256'] and h not in seen;seen.add(h)
            im=cv2.imdecode(np.frombuffer(binary,np.uint8),cv2.IMREAD_COLOR)
            mask=cv2.imdecode(np.fromfile(row['mask'],np.uint8),cv2.IMREAD_GRAYSCALE)
            assert im.shape==(288,288,3) and mask.shape==(288,288) and set(np.unique(mask))<={0,255}
            pixels=int((mask>0).sum());assert pixels>=4000;minimum=min(minimum,pixels)
        records.append(dict(identity=rec['identity'],berries=len(xyz),geometry_hash=rec['geometry_hash'],min_contact_ratio=float(ratio.min()),
                            stem_exposure=rec['exposure']['mean'],retries=len(rec['failures'])))
    real=json.loads((ROOT/'real_rows.json').read_text(encoding='utf-8'));split=json.loads((ROOT/'real_protocol.json').read_text(encoding='utf-8'))
    for a,b in [('train_indices','val_query_indices'),('train_indices','test_query_indices'),('val_query_indices','val_gallery_indices'),('test_query_indices','test_gallery_indices')]:
        assert not ({real[i]['image_sha256'] for i in split[a]}&{real[i]['image_sha256'] for i in split[b]})
    result=dict(identities=100,images=2400,distinct_geometries=len(hashes),distinct_rgb_images=len(seen),berries=total_berries,
                minimum_foreground_pixels=minimum,real_query_gallery_byte_overlap=0,records=records,
                visual_review='All explicit group sheets viewed by the implementer; eight azimuth views per identity')
    (ROOT/'data_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':audit()
