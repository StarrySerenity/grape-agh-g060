"""Explicit consecutive stable interval, not a cherry-picked correspondence subset."""
import json,cv2,numpy as np
from pathlib import Path
from prepare_real import render_review
ROOT=Path(__file__).resolve().parent
def main():
 for source in ['grow_B','grow_C','grow_B_mixed','grow_C_mixed']:
  key=source.replace('_mixed','')+'_stable'+('_mixed' if source.endswith('_mixed') else '')
  out=ROOT/'real'/key;out.mkdir(exist_ok=True)
  data=json.loads((ROOT/'real'/source/'detections.json').read_text(encoding='utf-8'))
  # B's later frames contain pronounced blur, C's later frames stronger viewpoint
  # and target motion. Keep the same initial 1.3-second interval in both methods.
  data['frames']=data['frames'][:14];data['clip']=key
  data['stable_subset']=dict(source_clip=source,frame_start=0,frame_end=13,
       reason='continuous early section before later blur and stronger movement; selected before comparing stable reconstructions',
       sampled_interval_seconds=[data['frames'][0]['time_seconds'],data['frames'][-1]['time_seconds']])
  for name in ['frames.json','detections.json']:(out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
  render_review(key,data)
  print('STABLE',key,len(data['frames']),data['stable_subset']['sampled_interval_seconds'])

if __name__=='__main__':main()
