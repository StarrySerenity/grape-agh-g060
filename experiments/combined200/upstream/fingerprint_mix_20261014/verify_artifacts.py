import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent))
from fingerprint_2d import query_image,imread,model_resources

rows=json.loads((ROOT/'real_rows.json').read_text(encoding='utf-8'))
split=json.loads((ROOT/'real_protocol.json').read_text(encoding='utf-8'))
registry=json.loads((ROOT/'model_registry.json').read_text(encoding='utf-8'))
checks=[]
for key in ['real_only','mixed']:
    checkpoint,folder=model_resources(key)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==registry['models'][key]['checkpoint_sha256']
    assert json.loads((folder/'all_records.json').read_text(encoding='utf-8'))==rows
    vectors=np.load(folder/'all_embeddings.npy');syn=np.load(folder/'all_synthetic_embeddings.npy')
    assert vectors.shape==(1104,256) and syn.shape==(2400,256)
    assert np.allclose(np.linalg.norm(syn,axis=1),1,atol=1e-5)
    i=split['test_query_indices'][0];row=rows[i]
    result=query_image(imread(row['image']),imread(row['mask'],True),model_key=key)
    error=float(np.max(np.abs(vectors[i]-result['fingerprint'])))
    assert error<1e-5,(key,error)
    assert all(c['identity'] not in split['excluded_ids'] and rows[c['best_view_index']]['capture_kind']=='video_frame' for c in result['candidates'])
    checks.append(dict(model=key,max_inference_gallery_difference=error,real_shape=list(vectors.shape),synthetic_shape=list(syn.shape)))
for row in rows:
    assert hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()==row['image_sha256']
    assert hashlib.sha256(Path(row['mask']).read_bytes()).hexdigest()==row['mask_sha256']
total=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
out=dict(models=checks,source_images_verified=len(rows),source_masks_verified=len(rows),source_files_changed=False,
         gpu_used=False,experiment_size_mib=round(total/2**20,1))
(ROOT/'qa/artifact_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out),flush=True)
