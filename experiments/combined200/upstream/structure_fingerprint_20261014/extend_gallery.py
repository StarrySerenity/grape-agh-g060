import argparse,json,sys
import cv2,numpy as np
from detection import ROOT,prepare_crop,detect_crop,overlay
from prepare_pilot import save,write_image
sys.path.insert(0,str(ROOT.parent))
from fingerprint_2d import imread

def extend(identity):
    catalog=json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'))
    if identity not in catalog['identities']:raise ValueError('Choose one of the four existing pilot identities')
    rows=json.loads((ROOT.parent/'fingerprint_2d/all_records.json').read_text(encoding='utf-8'));existing={r['source_index'] for r in catalog['records']};tiles=[]
    added=[];j=sum(r['identity']==identity for r in catalog['records'])
    for index,row in enumerate(rows):
        if index in existing or row['identity']!=identity or row['capture_kind']!='video_frame':continue
        crop,target=prepare_crop(imread(row['image']),imread(row['mask'],True));raw=detect_crop(crop,target);median=np.median([n['radius'] for n in raw]) if raw else 1
        nodes=[dict(n) for n in raw if n['radius']>=.55*median and n['confidence']>=.30]
        for k,n in enumerate(nodes):n['id']=f'B{k+1:02d}'
        key=f'{identity}_{j:02d}';j+=1
        write_image(ROOT/'data'/(key+'.jpg'),crop);cv2.imencode('.png',target)[1].tofile(str(ROOT/'data'/(key+'_target.png')))
        record=dict(key=key,identity=identity,name=row['name'],capture_kind=row['capture_kind'],video_group=row['video_group'],
            image='structure_fingerprint_20261014/data/'+key+'.jpg',target='structure_fingerprint_20261014/data/'+key+'_target.png',
            nodes=nodes,raw_nodes=raw,width=crop.shape[1],height=crop.shape[0],source_index=index,
            source_image_sha256=row['image_sha256'],source_mask_sha256=row['mask_sha256'],checked=False,all_visible_berries_verified=False,
            source='CPU GBISC, additional registered view; same fragment rule; not selected by matching score')
        added.append(record);save(ROOT/'data'/(key+'.json'),record);write_image(ROOT/'qa'/(key+'_nodes.jpg'),overlay(crop,nodes))
        label=np.full((25,180,3),245,np.uint8);cv2.putText(label,key+' / '+str(len(nodes)),(3,18),cv2.FONT_HERSHEY_SIMPLEX,.38,(30,50,30),1)
        tiles.append(np.vstack([label,cv2.resize(overlay(crop,nodes),(180,250))]));print(key,len(nodes),flush=True)
    if tiles:
        while len(tiles)%5:tiles.append(np.full_like(tiles[0],245))
        write_image(ROOT/'qa'/(identity+'_extra.jpg'),np.vstack([np.hstack(tiles[k:k+5]) for k in range(0,len(tiles),5)]))
    catalog['records'].extend(added);save(ROOT/'catalog.json',catalog);print('ADDED',identity,len(added),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);extend(p.parse_args().identity)
