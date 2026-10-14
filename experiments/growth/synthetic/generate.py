from demo_core import *
import hashlib,json,time,math
from growth_data import read,save,longitudinal_pairs,digest
sys.path.insert(0,str(BASE/'synthetic_bunch'))
from paper_grammar import generate_paper_bunch
from export_growth import sphere_cloud

CONFIGS=[dict(id='YOUNG_01',group='GROW',seed=330711,count=36,profile='cone',relative_radius=.115),
         dict(id='YOUNG_02',group='GROW',seed=330733,count=42,profile='shoulder',relative_radius=.125),
         dict(id='YOUNG_03',group='GROW',seed=330757,count=48,profile='curved',relative_radius=.135),
         dict(id='YOUNG_04',group='YG',seed=330779,count=52,profile='cone',relative_radius=.074),
         dict(id='YOUNG_05',group='YG',seed=330799,count=60,profile='shoulder',relative_radius=.080),
         dict(id='YOUNG_06',group='YG',seed=330823,count=68,profile='curved',relative_radius=.086)]

def make_young(config):
    b=generate_paper_bunch(seed=config['seed'],count=config['count'],profile=config['profile'],density='loose')
    p,r=arrays(b['models']);dist=__import__('scipy').spatial.distance.pdist(p)
    radius_scale=min(config['relative_radius']*np.median(dist)/np.median(r),.88 if config['group']=='GROW' else .72)
    nodes={n['id']:n for n in b['nodes']}
    for m in b['models']:
        node=nodes[m['node_id']];direction=np.asarray(node['position'])-m['center'];direction/=np.linalg.norm(direction)
        m.update(radius=float(m['radius']*radius_scale),evidence='synthetic_young_spherical_fruit')
        node['position']=(np.asarray(m['center'])+direction*m['radius']).tolist()
    for stem in b['stems']:
        node=nodes[f'N{stem["node"]:04d}'];stem['points']=[nodes[node['parent']]['position'],node['position']]
    b.update(identity_id=config['id'],stage='synthetic_young',bunch_id=config['id'],group=config['group'],uses_future_reference=False,
        source_seed=config['seed'],generation_parameters=config,radius_scale=radius_scale,
        generation_note='New independent branching grammar; reduced fruit radii make a young morphology. No corresponding mature model supplied.',
        future_growth_ground_truth_available=False)
    return b

def main():
    (ROOT/'cases').mkdir(exist_ok=True);(ROOT/'qa').mkdir(exist_ok=True)
    protected=[BASE/'data/snapshot.json',BASE/'forecast.py',BASE/'growth_geometry.py',BASE/'growth_data.py',BASE/'results/forecast.json',
        BASE/'synthetic_bunch/paper_grammar.py']
    save(ROOT/'protected_sources.json',{str(p):digest(p) for p in protected})
    snapshot=read(BASE/'data/snapshot.json');pairs=longitudinal_pairs(snapshot['samples']);records=[]
    save(ROOT/'plan.json',dict(samples=CONFIGS,count=6,uses_existing_growth_predictor=True,no_new_training=True,
        source_pairs={g:[p['identity'] for p in pairs if p['group']==g and p['status']=='ready'] for g in ['GROW','YG']},
        input_is_independent_synthetic_young=True,cohort_specific_days={'GROW':40,'YG':14},center_expansion=1.35,
        expansion_is_uncalibrated_display_assumption=True,young_input_generation='branching grammar with small spherical berries',
        output_no_dropout_model=True,relative_geometry_only=True,validation='sphere IDs/counts, growth ratios, overlaps, multi-angle browser inspection'))
    for config in CONFIGS:
        start=time.time();folder=ROOT/'cases'/config['id'];folder.mkdir(exist_ok=True)
        if (folder/'record.json').exists():raise RuntimeError('Existing versioned result; do not overwrite '+config['id'])
        young=make_young(config);save(folder/'young.json',young)
        result=predict_young(young,pairs,config['group']);forecast=result['forecast'];grown=result['grown']
        save(folder/'raw_forecast.json',forecast)
        save(folder/'grown.json',dict(identity_id=config['id']+'_GROWN',models=grown['models'],stage='predicted_grown',synthetic=True,uses_future_reference=False,
            target_days=forecast['target_days'],training_identities=forecast['training_identities'],predicted_shape_metrics=metrics(grown['models']),
            growth_display=grown,generation='existing predictor followed by limited collision relaxation; not a known synthetic future',
            per_berry_motion_learned=False,biological_future_ground_truth=False))
        if grown['contact_constraint']['status']!='clear':
            save(folder/'failure.json',grown['contact_constraint']);raise RuntimeError('Inspect unresolved sphere overlaps: '+config['id'])
        for phase,models in [('young',young['models']),('grown',grown['models'])]:sphere_cloud(models,folder/f'{phase}.ply')
        p,r=arrays(young['models']);q,s=arrays(grown['models']);bb=lambda a,rad:np.ptp(a,axis=0)+2*np.median(rad)
        record=dict(**config,target_days=forecast['target_days'],models=len(r),young=f'cases/{config["id"]}/young.json',grown=f'cases/{config["id"]}/grown.json',
            young_ply=f'cases/{config["id"]}/young.ply',grown_ply=f'cases/{config["id"]}/grown.ply',
            young_metrics=metrics(young['models']),grown_metrics=metrics(grown['models']),
            berry_radius_ratio=float(np.median(s/r)),center_distance_ratio=float(np.median(__import__('scipy').spatial.distance.pdist(q))/np.median(__import__('scipy').spatial.distance.pdist(p))),
            extent_ratio=(bb(q,s)/bb(p,r)).tolist(),same_id_order=[m['id'] for m in young['models']]==[m['id'] for m in grown['models']],
            contact_constraint=grown['contact_constraint'],selection=forecast['model_selection']['choices'],seconds=round(time.time()-start,2))
        save(folder/'record.json',record);records.append(record);save(ROOT/'catalog.json',dict(records=records,complete=len(records)==6))
        print(config['id'],config['group'],len(r),'radii x',round(record['berry_radius_ratio'],3),'overlaps',record['contact_constraint']['overlapping_pairs'],'seconds',record['seconds'],flush=True)
    save(ROOT/'results.json',dict(samples=6,berries=sum(r['models'] for r in records),records=records,
        ready=True,mode='approximate synthetic young-to-grown test',synthetic_future_gt_available=False,
        validation_can_show_geometry_and_plausibility_not_forecast_accuracy=True))

if __name__=='__main__':main()
