"""Current, read-only editor snapshot and explicitly staged longitudinal pairing."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def after_thinning_allowed(row):return row.get('stage') in ('after','young')


def freeze(source_root,identity_csv,stages_path):
    root=Path(source_root);rows=read(root/'inventory.json')['jobs'];stages=read(stages_path)['jobs']
    with Path(identity_csv).open(encoding='utf-8-sig',newline='') as f:mapping={str(Path(r['source_dir']).resolve()).casefold():r for r in csv.DictReader(f)}
    samples=[];hashes={str(identity_csv):digest(identity_csv),str(root/'inventory.json'):digest(root/'inventory.json'),str(stages_path):digest(stages_path)}
    for row in rows:
        key=row['key'];selection=root/key/'selection.json';selected=read(selection) if selection.exists() else {}
        state=root/'manual_edits'/key/'state.json';source=state if state.exists() else Path(selected.get('model_path',str(root/key/'berry_centers.json')))
        h=digest(source);raw=read(source)
        if digest(source)!=h:raise RuntimeError('Source changed during snapshot: '+str(source))
        hashes[str(source)]=h
        if selection.exists():hashes[str(selection)]=digest(selection)
        info=mapping[str(Path(row['sources'][0]).parent.resolve()).casefold()]
        models=[]
        for b in raw.get('models',raw.get('berries',[])):
            # Match the live sphere-only editor. Its radius is authoritative; axes are not used.
            q={k:b[k] for k in ('id','center','radius','manual','manual_edited','evidence') if k in b}
            q.update(id=str(q['id']),shape='sphere');models.append(q)
        stage=stages.get(row['job'],{}).get('stage',info['stage']) if row['period']=='May' else info['stage']
        samples.append(dict(sample_id=key,bunch_id=info['bunch_id'],group=info['identity_group'],date=info['source_relpath'].split('/')[0],
            stage=stage,models=models,model_source=str(source),model_sha256=h,revision=raw.get('revision',0),
            measured_cloud=row['measured_cloud'],source_videos=row['sources'],uses_future_reference=False,
            input_manifest=row.get('input_manifest'),viewer_path=selected.get('viewer_path',str(root/key/'viewer.html'))))
    return dict(samples=samples,source_hashes=hashes,created=datetime.now().isoformat(),coordinate_units='independent arbitrary COLMAP units')


def longitudinal_pairs(samples):
    grouped={}
    for s in samples:
        if s['group'] in ('GROW','YG'):grouped.setdefault(s['bunch_id'],[]).append(s)
    pairs=[]
    for identity,rows in sorted(grouped.items()):
        group=rows[0]['group'];dates=sorted(set(s['date'] for s in rows))
        if len(dates)!=2:continue
        candidates=[s for s in rows if s['date']==dates[0] and after_thinning_allowed(s)]
        late=[s for s in rows if s['date']==dates[-1]]
        if not late:continue
        early=candidates[0] if len(candidates)==1 else None
        fallback=next((s for s in rows if s['date']==dates[0]),None)
        pairs.append(dict(identity=identity,group=group,early=early or fallback,late=late[0],
            status='ready' if early is not None and len(late)==1 else 'early_after_thinning_stage_unconfirmed',
            days=(datetime.strptime(dates[-1],'%Y%m%d')-datetime.strptime(dates[0],'%Y%m%d')).days))
    return pairs
