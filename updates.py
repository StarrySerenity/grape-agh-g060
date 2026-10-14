"""Auditable adapters for the October 14 geometry-only fingerprint and growth work."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/update-20261014'
def read(relative):return json.loads((DATA/relative).read_text(encoding='utf-8'))
def structure_scores(result):
 rows=result['records']
 if not rows:raise ValueError('No query records')
 n=len(rows);ranks=[r.get('rank') for r in rows]
 return {'queries':n,'top1':sum(r==1 for r in ranks)/n,'top3':sum(r is not None and r<=3 for r in ranks)/n,
         'hits_top1':sum(r==1 for r in ranks),'hits_top3':sum(r is not None and r<=3 for r in ranks),
         'frame_map':sum(float(r.get('ap',0)) for r in rows)/n,'insufficient_queries':sum(r is None for r in ranks)}
def growth_scores(result):
 rows=result['folds']
 if not rows:raise ValueError('No prediction folds')
 for r in rows:
  if r['identity'] in r['train_ids']:raise ValueError('Test identity leaked into training')
  if len(r['prediction'])!=len(r['reference']) or len(r['baseline'])!=len(r['reference']):raise ValueError('Target mismatch')
 dims=len(rows[0]['reference'])
 def mae(field):return [sum(abs(float(r[field][i])-float(r['reference'][i])) for r in rows)/len(rows) for i in range(dims)]
 model,baseline=mae('prediction'),mae('baseline')
 if not all(math.isfinite(x) for x in model+baseline):raise ValueError('Nonfinite prediction')
 return {'pairs':len(rows),'targets':result.get('targets',[]),'model_mae':model,'baseline_mae':baseline,'better_than_mean':[a<b for a,b in zip(model,baseline)],'all_test_identities_held_out':True}
def structure_evaluate():
 source=read('structure_training_20261014/results.json');models={}
 for key,model in source['models'].items():
  models[key]={scope:structure_scores(model[scope]) for scope in ['validation','regression']}
  for scope in ['validation','regression']:
   for name in ['top1','top3']:
    if not math.isclose(models[key][scope][name],model[scope][name],abs_tol=1e-12):raise ValueError('Stored metric mismatch')
 return {'ok':True,'models':models,'dimension':128,'geometry_only':True,'training_real_identities':source['training_real_identities'],'training_real_graphs':source['training_real_graphs'],
         'synthetic_identities':source['data']['synthetic_identities'],'synthetic_graphs':source['data']['synthetic_graphs'],
         'independent_test':False,'interpretation':'Previously viewed photos are regression checks. Development gains did not improve regression Top-1. Detection can still depend on lighting.'}
def growth_evaluate():
 source=read('results/forecast.json');groups={key:growth_scores(source[key]) for key in ['GROW','YG']}
 for key,value in groups.items():
  for field in ['model_mae','baseline_mae']:
   if not all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) for a,b in zip(value[field],source[key][field])):raise ValueError('Stored growth MAE mismatch')
 return {'ok':True,'groups':groups,'excluded_stage_uncertain':['YG_07','YG_08'],'center_expansion_default':1.35,'center_expansion_measured':False,
         'interpretation':'Endpoint morphology only. Animation is interpolation, not a measured daily curve. Future-assisted repairs excluded from independent forecasts.'}
def overview():
 return {'ok':True,'snapshot':read('manifest.json')['snapshot_date'],'workbench':'web/updates.html','source_pages':{'growth':'http://127.0.0.1:8790/','fingerprint':'http://127.0.0.1:8790/structure_trained.html'},
         'new_features':['geometry-only 128-dimensional trained image fingerprint','real vs synthetic regression comparison','20 growth shape demonstrations, 18 evaluation pairs'],
         'older_workbench':'web/index.html','scope':'New results are frozen separately; October 13 3D baseline results remain separately labeled.'}
def structure_case(record_index=0,model_key='real_only'):
 if model_key not in ('real_only','mixed'):raise ValueError('Unknown model')
 import hashlib,importlib.util
 import numpy as np
 module_path=DATA/'structure_training_20261014/graph_model.py'
 spec=importlib.util.spec_from_file_location('g060_graph_encoder',module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 torch=module.torch
 catalog=read('catalog.json')['records'];by_index={r['index']:r for r in catalog}
 source=read('structure_training_20261014/results.json');queries=source['models'][model_key]['regression']['records']
 if not 0<=record_index<len(queries):raise ValueError('Query index must be 0..19')
 query=by_index[queries[record_index]['index']]
 config=read('structure_training_20261014/registry.json')['models'][model_key];checkpoint=DATA/config['checkpoint']
 if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=config['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
 state=torch.load(checkpoint,map_location='cpu',weights_only=True)
 net=module.GraphNet(len(state['classes']));net.load_state_dict(state['model']);net.eval()
 x,mask=module.pack(query['nodes'])
 with torch.inference_mode():vector=net(torch.tensor(x[None]),torch.tensor(mask[None])).numpy()[0]
 embeddings=np.load(DATA/config['gallery']/'embeddings.npy',allow_pickle=False)
 identity_scores={}
 for row in catalog:
  if row['capture_kind']=='video_frame':identity_scores.setdefault(row['identity'],[]).append((float(embeddings[row['index']]@vector),row))
 candidates=[]
 for identity,rows in identity_scores.items():
  rows.sort(key=lambda pair:(-pair[0],pair[1]['index']));best=rows[0][1]
  candidates.append({'identity':identity,'cosine':float(np.mean([s for s,r in rows[:3]])),'best_index':best['index']})
 candidates.sort(key=lambda c:(-c['cosine'],c['identity']))
 return {'ok':True,'query_index':record_index,'record_index':query['index'],'reference_identity':query['identity'],'model_key':model_key,'dimensions':len(vector),
         'rank':next(i+1 for i,c in enumerate(candidates) if c['identity']==query['identity']),'candidates':candidates[:5],'node_count':len(query['nodes']),
         'encoding_recomputed':True,'uses_rgb_for_matching':False,'identity_confirmed':False,'checkpoint_sha256':config['checkpoint_sha256']}
def dashboard():
 import numpy as np
 source=read('structure_training_20261014/results.json');catalog={r['index']:r for r in read('catalog.json')['records']}
 registry=read('structure_training_20261014/registry.json')['models'];vectors={key:np.load(DATA/value['gallery']/'embeddings.npy',allow_pickle=False) for key,value in registry.items()}
 cases=[]
 for i,row in enumerate(source['models']['real_only']['regression']['records']):
  q=catalog[row['index']];models={}
  for key,value in source['models'].items():
   record=next(r for r in value['regression']['records'] if r['index']==row['index']);candidate=catalog[record['candidates'][0]['best_index']]
   models[key]={'rank':record['rank'],'candidate':candidate,'cosine':record['candidates'][0]['cosine'],'fingerprint':vectors[key][row['index']].tolist(),'checkpoint_sha256':registry[key]['checkpoint_sha256']}
  cases.append({'query':q,'models':models,'index':i})
 return {'overview':overview(),'structure':structure_evaluate(),'growth':growth_evaluate(),'cases':cases,'growth_cases':read('growth-cases.json')}
def generate():
 payload=dashboard();(ROOT/'results/update-20261014.js').write_text('window.GRAPE_UPDATE='+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';',encoding='utf-8')
 return {'ok':True,'report':'web/updates.html','queries':len(payload['cases']),'growth_cases':len(payload['growth_cases'])}
if __name__=='__main__':print(json.dumps(generate(),ensure_ascii=True))
