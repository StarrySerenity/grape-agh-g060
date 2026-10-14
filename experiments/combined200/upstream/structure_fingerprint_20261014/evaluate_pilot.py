import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import json,time
import cv2,numpy as np
from detection import ROOT,detect_crop,overlay
from geometry import match_graphs,fingerprint
from prepare_pilot import write_image,save

def matches_image(query,gallery,result):
    a=cv2.imread(str(ROOT.parent/query['image']));b=cv2.imread(str(ROOT.parent/gallery['image']))
    width=370;height=520
    def panel(im,record):
        im=overlay(im,record['nodes']);scale=min(width/im.shape[1],height/im.shape[0]);iw=round(im.shape[1]*scale);ih=round(im.shape[0]*scale)
        canvas=np.full((height,width,3),128,np.uint8);x=(width-iw)//2;y=(height-ih)//2;canvas[y:y+ih,x:x+iw]=cv2.resize(im,(iw,ih))
        return canvas,scale,x,y
    a,sa,xa,ya=panel(a,query);b,sb,xb,yb=panel(b,gallery);canvas=np.hstack([a,b]);rng=np.random.default_rng(7)
    for pair in result['pairs']:
        u=query['nodes'][pair['query']];v=gallery['nodes'][pair['gallery']];color=tuple(int(c) for c in rng.integers(80,255,3))
        p=(round(xa+u['x']*sa),round(ya+u['y']*sa));q=(round(width+xb+v['x']*sb),round(yb+v['y']*sb))
        cv2.line(canvas,p,q,color,1,cv2.LINE_AA)
    label=np.full((50,width*2,3),245,np.uint8)
    cv2.putText(label,query['key']+' -> '+gallery['key'],(8,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,50,30),1)
    cv2.putText(label,f"score {result['score']:.3f} / pairs {len(result['pairs'])} / geometry candidate only",(8,40),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,50,30),1)
    return np.vstack([label,canvas])

def evaluate():
    catalog=json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'));records=catalog['records'];queries=[r for r in records if r['capture_kind']=='photo'];gallery=[r for r in records if r['capture_kind']=='video_frame'];start=time.time();results=[]
    for query in queries:
        candidates=[]
        for entry in gallery:
            m=match_graphs(query['nodes'],entry['nodes']);candidates.append(dict(key=entry['key'],identity=entry['identity'],**m))
        best=[]
        for identity in catalog['identities']:best.append(max([c for c in candidates if c['identity']==identity],key=lambda c:c['score']))
        best.sort(key=lambda c:-c['score']);rank=next(i+1 for i,c in enumerate(best) if c['identity']==query['identity'])
        result=dict(query=query['key'],identity=query['identity'],rank=rank,candidates=best,margin=best[0]['score']-best[1]['score'],
                    fingerprint=fingerprint(query['nodes']),independent_accuracy_claimed=False)
        results.append(result);top=next(r for r in gallery if r['key']==best[0]['key'])
        write_image(ROOT/'qa'/(query['key']+'_match.jpg'),matches_image(query,top,best[0]))
        print(query['key'],'top',top['identity'],'rank',rank,'score',round(best[0]['score'],3),'elapsed',round(time.time()-start),flush=True)
    save(ROOT/'pilot_results.json',dict(queries=len(results),identities=4,gallery_views=len(gallery),top1=float(np.mean([r['rank']==1 for r in results])),
         independent_accuracy_claimed=False,scope='development pilot on old training identities, four identities only',results=results))
    stress=[]
    for query in [queries[0],queries[2],queries[4],queries[6]]:
        image=cv2.imread(str(ROOT.parent/query['image']));target=cv2.imdecode(np.fromfile(ROOT.parent/query['target'],np.uint8),cv2.IMREAD_GRAYSCALE)
        for kind in ['dark','bright','hue']:
            if kind=='hue':
                hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);hsv[...,0]=(hsv[...,0].astype(int)+55)%180;changed=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            else:changed=np.clip(image.astype(float)*(.55 if kind=='dark' else 1.35),0,255).astype(np.uint8)
            changed[target==0]=128;raw=detect_crop(changed,target);median=np.median([n['radius'] for n in raw]) if raw else 1
            nodes=[n for n in raw if n['radius']>=.55*median and n['confidence']>=.30]
            m=match_graphs(query['nodes'],nodes);same=fingerprint(query['nodes'])==fingerprint([dict(n,color=kind) for n in query['nodes']])
            stress.append(dict(query=query['key'],change=kind,original_nodes=len(query['nodes']),detected_nodes=len(nodes),
                               fixed_nodes_fingerprint_equal=same,redetected_match=m))
            write_image(ROOT/'qa'/(query['key']+'_'+kind+'.jpg'),overlay(changed,nodes))
            save(ROOT/'data'/(query['key']+'_'+kind+'.json'),dict(nodes=nodes))
    save(ROOT/'stress_results.json',dict(records=stress,fixed_node_invariance=all(s['fixed_nodes_fingerprint_equal'] for s in stress),
        end_to_end_invariance_guaranteed=False))
    print('PILOT_EVALUATED',flush=True)

if __name__=='__main__':evaluate()
