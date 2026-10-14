"""Save real normal, boundary and failure numerical results."""
import copy,json
from pathlib import Path
from geometry import describe
from experiments import evaluate_forecast
from cli import execute

def main():
 results=[{'case':'normal','result':execute('describe','May_g2_07')}]
 data=[{'center':p,'radius':.1} for p in [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]]
 for label,models in [('four_distinct_centers',data),('tiny_positive_radius',[{**m,'radius':1e-12} for m in data]),('partial_duplicate_centers',data+[data[0]])]:
  results.append({'case':label,'result':describe(models)})
 for label in ['too_few','nan_center','negative_radius','identical_centers']:
  models=copy.deepcopy(data)
  if label=='too_few':models=models[:3]
  elif label=='nan_center':models[0]['center'][0]=float('nan')
  elif label=='negative_radius':models[0]['radius']=-1
  else:
   for m in models:m['center']=[0,0,0]
  try:describe(models)
  except ValueError as e:results.append({'case':label,'rejected':True,'error':str(e)})
  else:raise AssertionError('Failure case accepted: '+label)
 try:evaluate_forecast([])
 except ValueError as e:results.append({'case':'missing_longitudinal_pairs','rejected':True,'error':str(e)})
 else:raise AssertionError('Missing pairs accepted')
 target=Path(__file__).resolve().parent/'evidence/numerical-cases.json'
 target.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps({'cases':len(results),'expected_failures':sum(r.get('rejected',False) for r in results),'saved':target.name}))
if __name__=='__main__':main()
