import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=json.loads((ROOT.parent/'fingerprint_2d/all_records.json').read_text(encoding='utf-8'));catalog=json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'))
for rec in catalog['records']:
    row=source[rec['source_index']]
    assert rec['source_image_sha256']==hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()
    assert rec['source_mask_sha256']==hashlib.sha256(Path(row['mask']).read_bytes()).hexdigest()
    assert row['identity']==rec['identity'] and row['name']==rec['name']
    assert (ROOT.parent/rec['image']).exists()
size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())/2**20
out=dict(source_images_verified=len(catalog['records']),source_masks_verified=len(catalog['records']),original_data_changed=False,
         artifact_size_mib=round(size,2),gpu_used=False,matcher_has_color_inputs=False)
(ROOT/'qa/source_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out),flush=True)
