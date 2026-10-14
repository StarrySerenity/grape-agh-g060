import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import sys,json
from pathlib import Path
import cv2,numpy as np
from detection import ROOT,overlay
from projection import project_instances
from geometry import match_graphs,fingerprint
from prepare_pilot import save,write_image
sys.path.insert(0,str(ROOT.parent/'fingerprint_mix_20261014'))
from render_synthetic import render

samples=[];results=[]
for index in [0,15,35,95]:
    model=json.loads((ROOT.parent/'fingerprint_mix_20261014/synthetic'/f'SYN_{index:03d}'/'model.json').read_text(encoding='utf-8'))
    views=[]
    for k,(yaw,seed) in enumerate([(0.,70001),(0.,71019),(.12,71019)]):
        im,_=render(model,yaw,0,seed);p=project_instances(model,yaw,0);nodes=p['nodes'];median=np.median([n['radius'] for n in nodes])
        nodes=[n for n in nodes if n['radius']>=.55*median]
        key=f'SYN_{index:03d}_{k}';image='structure_fingerprint_20261014/data/'+key+'.jpg'
        write_image(ROOT.parent/image,im);np.save(ROOT/'data'/(key+'_labels.npy'),p['labels'])
        rec=dict(key=key,identity=f'SYN_{index:03d}',nodes=nodes,image=image,width=288,height=288,synthetic=True,
                 source='known berry-ID depth buffer; visible centroid separate from true projected sphere center',yaw=yaw)
        save(ROOT/'data'/(key+'.json'),rec);write_image(ROOT/'qa'/(key+'_nodes.jpg'),overlay(im,nodes));views.append(rec);samples.append(rec)
    for j in [1,2]:
        m=match_graphs(views[0]['nodes'],views[j]['nodes']);correct=sum(views[0]['nodes'][p['query']]['berry_id']==views[j]['nodes'][p['gallery']]['berry_id'] for p in m['pairs'])
        results.append(dict(identity=views[0]['identity'],type='same view different appearance' if j==1 else 'slightly changed viewpoint',
                     score=m['score'],pairs=len(m['pairs']),known_id_correct=correct,
                     fixed_geometry_fingerprint_equal=fingerprint(views[0]['nodes'])==fingerprint(views[j]['nodes'])))
    tiles=[cv2.resize(overlay(cv2.imread(str(ROOT.parent/v['image'])),v['nodes']),(288,288)) for v in views]
    write_image(ROOT/'qa'/(f'SYN_{index:03d}_projections.jpg'),np.hstack(tiles))
negatives=[]
for a in samples[::3]:
    for b in samples[::3]:
        if a['identity']>=b['identity']:continue
        m=match_graphs(a['nodes'],b['nodes']);negatives.append(dict(a=a['identity'],b=b['identity'],score=m['score'],pairs=len(m['pairs'])))
save(ROOT/'synthetic_catalog.json',dict(records=samples));save(ROOT/'synthetic_results.json',dict(positives=results,different_geometry=negatives,
    graph_training_performed=False,ground_truth_source='deterministic berry surface ID projection'))
print(json.dumps(dict(positives=results,negatives=negatives)),flush=True)
