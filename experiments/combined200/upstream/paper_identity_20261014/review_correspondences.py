import json,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def review(a,b):
 pair=read(ROOT/'comparisons'/f'{a}__{b}.json');panels=[];source_frames=[]
 for side,key in [('source',a),('target',b)]:
  rec=read(ROOT/'real'/key/'reconstruction.json');tracks=read(ROOT/'real'/key/'tracks.json');ids=[rec['track_ids'][p[side]] for p in pair.get('pairs',[])]
  if not ids:continue
  frame=max(tracks['frames'],key=lambda f:sum(n['track_id'] in ids for n in f['nodes']))
  source_frames.append(dict(clip=key,frame=frame['frame'],time=frame['time_seconds']))
  im=cv2.imdecode(np.fromfile(frame['image'],np.uint8),1);nodes={n['track_id']:n for n in frame['nodes']}
  for j,p in enumerate(pair['pairs']):
   id=rec['track_ids'][p[side]]
   if id not in nodes:continue
   n=nodes[id];hsv=np.array([[[round(j*179/max(len(pair['pairs']),1)),220,245]]],np.uint8);color=tuple(map(int,cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)[0,0]))
   xy=(round(n['x']),round(n['y']));cv2.circle(im,xy,round(n['radius']),color,3)
   cv2.putText(im,'P'+str(j),xy,cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,20),4,cv2.LINE_AA);cv2.putText(im,'P'+str(j),xy,cv2.FONT_HERSHEY_SIMPLEX,.7,color,2,cv2.LINE_AA)
  title=np.full((40,im.shape[1],3),245,np.uint8);cv2.putText(title,key+f' / f{frame["frame"]} / {frame["time_seconds"]:.2f}s',(5,27),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,40,30),1);panels.append(np.vstack([title,im]))
 if panels:
  out=ROOT/'comparisons'/f'{a}__{b}_physical_review.jpg';cv2.imencode('.jpg',np.hstack(panels),[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(str(out))
  print('REVIEW',out,source_frames,flush=True)
  pair['review_image']=str(out);pair['review_frames']=source_frames
  (ROOT/'comparisons'/f'{a}__{b}.json').write_text(json.dumps(pair,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('a');p.add_argument('b');v=p.parse_args();review(v.a,v.b)
