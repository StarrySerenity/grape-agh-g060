"""Copy only finished upstream artifacts into a dated competition snapshot."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path('E:/grape_growth_workspace_20261013')
DEST=ROOT/'data/update-20261014'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 DEST.mkdir(parents=True,exist_ok=True);copied={}
 def take(relative,name=None):
  source=(SOURCE/relative).resolve()
  if not source.is_relative_to(SOURCE.resolve()):raise ValueError('Source outside project')
  target=DEST/(name or relative);target.parent.mkdir(parents=True,exist_ok=True)
  before=sha(source);shutil.copyfile(source,target)
  if sha(source)!=before or sha(target)!=before:raise ValueError('Input changed during snapshot')
  copied[relative]={'sha256':before,'snapshot':target.relative_to(ROOT).as_posix()};return target
 for name in ['README.md','results/forecast.json','structure_training_20261014/results.json','structure_training_20261014/README.md','structure_training_20261014/registry.json','structure_training_20261014/protocol.json','structure_training_20261014/selection.json','structure_training_20261014/graph_model.py']:
  take(name)
 training=read(SOURCE/'structure_training_20261014/results.json')
 catalog_path=SOURCE/'structure_training_20261014/catalog.json';catalog_hash=sha(catalog_path);catalog=read(catalog_path)
 keep=set()
 for model in training['models'].values():
  for r in model['regression']['records']:
   keep.add(r['index']);keep.update(c['best_index'] for c in r['candidates'][:1])
 for r in catalog['records']:
  if r['index'] in keep:
   expected='/api/structure-trained/thumb/'+str(r['index'])
   if r['image']!=expected:raise ValueError('Unexpected image endpoint')
   target=DEST/'images'/f"{r['index']}.jpg";target.parent.mkdir(parents=True,exist_ok=True)
   with urllib.request.urlopen('http://127.0.0.1:8790'+expected,timeout=15) as response:target.write_bytes(response.read())
   r['portable_image']=target.relative_to(ROOT).as_posix()
   copied[expected]={'sha256':sha(target),'snapshot':r['portable_image'],'copy_kind':'derived masked crop from local upstream API'}
 if sha(catalog_path)!=catalog_hash:raise ValueError('Catalog changed')
 (DEST/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
 copied['structure_training_20261014/catalog.json']={'sha256':catalog_hash,'snapshot':'data/update-20261014/catalog.json','copy_kind':'normalized JSON; selected image paths added'}
 registry=read(SOURCE/'structure_training_20261014/registry.json')
 for model in registry['models'].values():
  take(model['checkpoint']);take(model['gallery']+'/embeddings.npy')
 growth=[]
 for folder in sorted((SOURCE/'pairs').iterdir()):
  if not (folder/'predicted_grown_models.json').exists():continue
  for name in ['early_original.json','late_original.json','predicted_grown_models.json','grown_display.json','report.json']:
   if (folder/name).exists():take((folder/name).relative_to(SOURCE).as_posix())
  early=read(folder/'early_original.json');late=read(folder/'late_original.json');grown=read(folder/'predicted_grown_models.json')
  growth.append({'identity':folder.name,'early':early,'late':late,'grown':grown})
 (DEST/'growth-cases.json').write_text(json.dumps(growth,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
 manifest={'source_root':str(SOURCE),'snapshot_date':'2026-10-14','source_version':'structure training finalized 16:28 local; subsequent research not included','files':copied,'growth_cases':len(growth),'portable_images':len(keep)}
 (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'files':len(copied),'growth_cases':len(growth),'portable_images':len(keep)}))
if __name__=='__main__':main()
