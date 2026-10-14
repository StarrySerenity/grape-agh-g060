"""Versioned combined data; physical identities, not image views, define splits."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys,json,hashlib,ctypes
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('E:/grape_growth_workspace_20261013');LEGACY=BASE/'paper_identity_20261014';OUT=ROOT/'synthetic_consumer'
OUT.mkdir(exist_ok=True)
for p in [LEGACY,BASE/'structure_fingerprint_20261014',BASE/'fingerprint_mix_20261014']:sys.path.insert(0,str(p))
import cv2,numpy as np
from projection import project_instances
cv2.setNumThreads(2)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def split_ids():
 raw=read(BASE/'synthetic_expansion_200_20261014/proposed_training_split.json')
 return {k:raw[k] for k in ['training','development','previous_test_regression','fresh_evaluation']}
def folder(identity):
 index=int(identity.split('_')[1])
 if not 0<=index<200:raise ValueError('Identity outside 200-bunch catalog')
 return (BASE/('fingerprint_mix_20261014' if index<100 else 'synthetic_expansion_200_20261014'))/'synthetic'/identity
def model(identity):return read(folder(identity)/'model.json')
def load_image_records(identities,views):
 records=[]
 for identity in identities:
  src=folder(identity);rows=read(src/'records.json')['rows'];bunch=model(identity)
  for view in views:
   row=rows[view];mask=cv2.imdecode(np.fromfile(row['mask'],np.uint8),0);proj=project_instances(bunch,row['yaw'],row['pitch'],visible_mask=mask)
   nodes=[dict(x=n['true_projected_center'][0],y=n['true_projected_center'][1],radius=n['true_projected_radius'],visibility=min(1,n['visible_fraction']),berry_id=n['berry_id']) for n in proj['nodes']]
   records.append(dict(domain='synthetic',identity=identity,image=row['image'],mask=row['mask'],nodes=nodes,view=view,
    geometry_hash=bunch['geometry_hash'],label_type='known full projected sphere center'))
 return records

def memory_available_gib():
 class Status(ctypes.Structure):
  _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ['total','available','total_page','available_page','total_virtual','available_virtual','extended']]
 status=Status();status.length=ctypes.sizeof(status);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status));return status.available/2**30

def protected_snapshot():
 p=ROOT/'protected_originals.json'
 if p.exists():return
 original=read(BASE/'synthetic_expansion_200_20261014/original_hashes.json')
 original.update({str(LEGACY/f):h for f,h in read(LEGACY/'synthetic_consumer/frozen_evaluation_protocol.json')['sha256'].items()})
 save(p,original)
