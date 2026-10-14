"""Adapted restricted stochastic grammar from Xin et al. (2020), Algorithm 1.

Skeleton is interpreted before each terminal berry. No post-hoc berry layout,
RRT stem completion, or free sphere displacement is used here.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

PAPER='https://doi.org/10.1111/ajgw.12444'
# Pinot Noir internode coefficients, Table 1. The 6-unit radius conversion is
# an explicit project normalization, not a reported or calibrated berry radius.
PARAMS={'rachis':(12.58/6,.89,1),'secondary':(8.20/6,.84,.74),
        'tertiary':(2.45/6,.75,.84),'pedicel':(5.88/6,1,.98)}

def internode_length(kind,longitudinal,transverse,noise=0):
    base,al,at=PARAMS[kind]
    return max(.06,base*al**(longitudinal-1)*at**(transverse-1)+noise)

def rotate_child(parent,alpha,beta):
    parent=np.asarray(parent,float);parent=parent/np.linalg.norm(parent)
    normal=np.array([parent[1],-parent[0],0.]);normal/=max(np.linalg.norm(normal),1e-12)
    if np.linalg.norm(normal)<.1:normal=np.array([1.,0.,0.])
    def rotate(v,axis,angle):
        c=math.cos(angle);s=math.sin(angle)
        return v*c+np.cross(axis,v)*s+axis*np.dot(axis,v)*(1-c)
    tilted=rotate(parent,normal,math.radians(alpha))
    child=rotate(tilted,parent,math.radians(beta));return child/np.linalg.norm(child)

def segment_dist(points,a,b):
    v=b-a;t=np.clip((points-a)@v/max(float(v@v),1e-14),0,1)
    return np.linalg.norm(points-a-t[:,None]*v,axis=1)

class Grammar:
    def __init__(self,seed,count,profile,density,scale):
        self.rng=np.random.default_rng(seed);self.count=count;self.profile=profile;self.density=density;self.scale=scale
        self.nodes=[];self.edges=[];self.models=[];self.derivation=[]
        self._edge_cache=None
        self.rejections=dict(boundary=0,berry_collision=0,stem_collision=0);self.attempts=0
        self.length=14.8*scale;self.width=(4.6 if profile=='shoulder' else 4.05)*scale
        self.slices=[dict(t=float(t),radius=self.hull_radius(float(t))) for t in np.linspace(0,1,50)]
        self.hull_t=np.array([s['t'] for s in self.slices]);self.hull_r=np.array([s['radius'] for s in self.slices])

    def axis(self,y):
        t=min(1.,max(0.,-y/self.length))
        return np.array([.5*math.sin(math.pi*t) if self.profile=='curved' else 0.,y,
                         .24*math.sin(math.pi*t*1.4) if self.profile=='curved' else 0.])

    def hull_radius(self,t):
        shape=[.76,1,.91,.65,.3] if self.profile!='shoulder' else [.78,1,1,.65,.3]
        return self.width*float(np.interp(t,[0,.17,.43,.76,1],shape))

    def inside(self,p,r=0):
        if p[1]+r>1.6*self.scale or p[1]-r<-self.length-.9*self.scale:return False
        t=min(1.,max(0.,-float(p[1])/self.length))
        ax=.5*math.sin(math.pi*t) if self.profile=='curved' else 0.
        az=.24*math.sin(math.pi*t*1.4) if self.profile=='curved' else 0.
        radius=float(np.interp(t,self.hull_t,self.hull_r))
        return math.hypot(float(p[0])-ax,(float(p[2])-az)/.9)+r<=radius

    def node(self,parent,position,kind,level,transverse,alpha=0,beta=0):
        idx=len(self.nodes);node=dict(id=f'N{idx:04d}',parent=parent,position=np.asarray(position).tolist(),kind=kind,
                                    longitudinal=level,transverse=transverse,alpha=float(alpha),beta=float(beta))
        self.nodes.append(node)
        if parent is not None:
            width={'peduncle':.12,'rachis':.105,'secondary':.065,'tertiary':.045,'pedicel':.03}[kind]
            edge=dict(id=f'E{idx:04d}',kind=kind,radius=width,points=[self.nodes[parent]['position'],node['position']],
                      parent=None if parent==0 else f'E{parent:04d}',node=idx)
            self.edges.append(edge)
            self._edge_cache=None
        self.derivation.append(dict(symbol='C',kind=kind,node=idx,parent=parent))
        return idx

    def segment_clear(self,a,b,width):
        if not self.models:return True
        p=np.array([m['center'] for m in self.models]);r=np.array([m['radius'] for m in self.models])
        return bool(np.all(segment_dist(p,a,b)>=r+width+.003))

    def try_internode(self,parent,heading,kind,nl,nt,alpha,beta):
        start=np.array(self.nodes[parent]['position']);width={'rachis':.105,'secondary':.065,'tertiary':.045,'pedicel':.03}[kind]
        for attempt in range(100):
            self.attempts+=1
            aa=alpha+self.rng.normal(0,5);bb=beta+self.rng.normal(0,10)
            direction=rotate_child(heading,aa,bb)
            length=internode_length(kind,nl,nt,noise=self.rng.normal(0,.035))*self.scale
            end=start+direction*length
            if not self.inside(end,width):self.rejections['boundary']+=1;continue
            if not self.segment_clear(start,end,width):self.rejections['stem_collision']+=1;continue
            idx=self.node(parent,end,kind,nl,nt,aa,bb)
            return idx,direction
        return None

    def pedicel(self,parent,heading,nl,nt,phase):
        start=np.array(self.nodes[parent]['position']);r=float(np.clip(self.rng.normal(1,.045),.85,1.15))
        for attempt in range(100):
            self.attempts+=1;aa=45+self.rng.normal(0,5);bb=phase+attempt*137.507+self.rng.normal(0,10)
            direction=rotate_child(heading,aa,bb)
            length=internode_length('pedicel',nl,nt,self.rng.normal(0,.025))*self.scale
            tip=start+direction*length;center=tip+direction*r
            if not self.inside(center,r):self.rejections['boundary']+=1;continue
            if self.models:
                p=np.array([m['center'] for m in self.models]);rr=np.array([m['radius'] for m in self.models])
                spacing=1.005 if self.density=='compact' else 1.07
                if np.any(np.linalg.norm(p-center,axis=1)<spacing*(rr+r)):
                    self.rejections['berry_collision']+=1;continue
            if not self.segment_clear(start,tip,.03):self.rejections['stem_collision']+=1;continue
            # New berry cannot engulf any pre-existing branch, including its ancestors.
            if self.edges:
                if self._edge_cache is None:
                    a=np.array([e['points'][0] for e in self.edges]);v=np.array([e['points'][1] for e in self.edges])-a
                    self._edge_cache=(a,v,np.sum(v*v,axis=1),np.array([e['radius'] for e in self.edges]))
                a,v,v2,widths=self._edge_cache
                t=np.clip(np.sum((center-a)*v,axis=1)/np.maximum(v2,1e-14),0,1)
                distances=np.linalg.norm(center-a-t[:,None]*v,axis=1)
                if np.any(distances<r+widths+.003):self.rejections['stem_collision']+=1;continue
            idx=self.node(parent,tip,'pedicel',nl,nt,aa,bb);bid=f'B{len(self.models)+1:03d}'
            self.edges[-1]['berry']=bid
            self.models.append(dict(id=bid,node_id=f'N{idx:04d}',center=center.tolist(),radius=r,shape='sphere'))
            self.derivation.append(dict(symbol='S',node=idx,berry=bid,radius=r))
            return True
        return False

    def terminal_cluster(self,parent,heading,nl,nt,budget):
        before=len(self.models)
        # Paper terminal multiplicity 1..8; quotas are an explicit project cap.
        desired=min(int(self.rng.integers(1,9)),budget)
        self.derivation.append(dict(symbol='TERMINAL_CLUSTER',node=parent,requested=desired))
        for i in range(desired):self.pedicel(parent,heading,nl,nt,i*137.507)
        return len(self.models)-before

    def secondary(self,parent,heading,nl,quota,phase):
        before=len(self.models);node=parent;direction=heading
        self.derivation.append(dict(symbol='[',node=parent,production='SECONDARY'))
        for nt in range(1,7):
            remaining=quota-(len(self.models)-before)
            if remaining<=0:break
            grown=self.try_internode(node,direction,'secondary',nl,nt,75 if nt==1 else 0,phase if nt==1 else 0)
            if grown is None:break
            node,direction=grown
            # Probability is a documented approximation; measured heatmap is unavailable.
            probability=max(.2,.8-.065*nl-.085*nt)
            if self.rng.random()<probability:
                third=self.try_internode(node,direction,'tertiary',nl,nt,75,self.rng.uniform(0,360))
                if third is not None:
                    self.derivation.append(dict(symbol='[',node=third[0],production='TERTIARY'))
                    self.terminal_cluster(third[0],third[1],nl,nt,min(remaining,3))
                    self.derivation.append(dict(symbol=']',node=third[0]))
            else:
                self.pedicel(node,direction,nl,nt,self.rng.uniform(0,360))
        # At the branch termination, invoke the paper's terminal-cluster production.
        remaining=quota-(len(self.models)-before)
        if remaining>0 and node!=parent:self.terminal_cluster(node,direction,nl,nt,remaining)
        self.derivation.append(dict(symbol=']',node=parent))
        return len(self.models)-before

    def run(self):
        root=self.node(None,[0,1.3*self.scale,0],'root',0,0);rachis=[];pos=np.array([0.,0.,0.])
        parent=self.node(root,pos,'peduncle',0,0)
        # Build the main skeleton first using the paper's small-angle rotations.
        direction=np.array([0.,-1.,0.])
        for nl in range(1,10):
            if self.profile=='curved':
                target=self.axis(float(self.nodes[parent]['position'][1])-internode_length('rachis',nl,1)*self.scale)
                direction=target-np.array(self.nodes[parent]['position']);direction/=np.linalg.norm(direction)
            grown=self.try_internode(parent,direction,'rachis',nl,1,0,self.rng.uniform(0,360))
            if grown is None:break
            parent,direction=grown;rachis.append(parent)
        tip_budget=2
        weights=np.array([.9**i for i in range(len(rachis))]);raw=weights/weights.sum()*(self.count-tip_budget)
        quotas=np.floor(raw).astype(int)
        for i in np.argsort(-(raw-quotas))[:self.count-tip_budget-int(quotas.sum())]:quotas[i]+=1
        phases=[0.,180.]
        for i,node in enumerate(rachis):
            n=self.nodes[node];a=np.array(self.nodes[n['parent']]['position']);b=np.array(n['position']);v=(b-a)/np.linalg.norm(b-a)
            # Alternating sides and quarter-turn between same-side nodes, Table 2.
            side=i%2
            if i>=2:phases[side]+=float(self.rng.normal(90,10))
            phase=phases[side]
            self.secondary(node,v,i+1,int(quotas[i]),phase)
        n=self.nodes[rachis[-1]];a=np.array(self.nodes[n['parent']]['position']);b=np.array(n['position']);v=(b-a)/np.linalg.norm(b-a)
        self.terminal_cluster(rachis[-1],v,len(rachis),1,min(tip_budget,self.count-len(self.models)))
        remaining=self.count-len(self.models)
        # Grammar-driven replenishment at rachis nodes, not random sphere insertion.
        for cycle in range(3):
            if remaining<=0:break
            for i,node in enumerate(rachis):
                if remaining<=0:break
                n=self.nodes[node];a=np.array(self.nodes[n['parent']]['position']);b=np.array(n['position']);v=(b-a)/np.linalg.norm(b-a)
                self.secondary(node,v,i+1,min(remaining,4),(i*137.507+cycle*73+97)%360)
                remaining=self.count-len(self.models)
        return self.count==len(self.models)

    def result(self,seed,count,attempt):
        # Prune only empty twigs; ancestral connectivity of every berry is retained.
        ids={int(m['node_id'][1:]) for m in self.models}
        for idx in list(ids):
            while idx is not None:ids.add(idx);idx=self.nodes[idx]['parent']
        nodes=[dict(n, parent=None if n['parent'] is None else f'N{n["parent"]:04d}') for i,n in enumerate(self.nodes) if i in ids]
        stems=[e for e in self.edges if e['node'] in ids]
        p=np.array([m['center'] for m in self.models]);r=np.array([m['radius'] for m in self.models]);d=np.linalg.norm(p[:,None]-p[None,:],axis=2);np.fill_diagonal(d,np.inf)
        minimum=float(np.min(d/(r[:,None]+r[None,:])));intersection=0
        for e in stems:
            dist=segment_dist(p,np.array(e['points'][0]),np.array(e['points'][1]));active=np.ones(len(r),bool)
            if 'berry' in e:active[[m['id'] for m in self.models].index(e['berry'])]=False
            intersection+=int(np.sum(dist[active]<r[active]+e['radius']-1e-7))
        if intersection or minimum<.995:raise RuntimeError('Independent geometry check failed')
        extent=np.ptp(p,axis=0)+2*np.median(r);geometry=dict(models=self.models,stems=stems,nodes=nodes)
        digest=hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return dict(identity_id=f'GRAMMAR_{self.profile}_{self.density}_{seed}',synthetic=True,method='restricted_stochastic_grammar',
                    source_paper=PAPER,source_reproduction='adaptation, not full reproduction',seed=seed,profile=self.profile,density=self.density,
                    geometry_hash=digest,**geometry,derivation=self.derivation,hull_slices=self.slices,
                    qa=dict(connected_berries=len(r),unrelated_stem_intersections=intersection,min_contact_ratio=minimum,
                            radius_cv=float(r.std()/r.mean()),length_width_ratio=float(extent[1]/max(extent[0],extent[2])),
                            secondary_branches=sum(n['kind']=='secondary' for n in nodes),tertiary_branches=sum(n['kind']=='tertiary' for n in nodes),
                            candidate_attempts=self.attempts,rejections=self.rejections,restart_attempt=attempt,scale=self.scale),
                    adaptations=['parametric 50-slice outline instead of image-derived hull','normalized length and uncalibrated radius',
                                 'approximate tertiary probabilities','quantity quotas and bounded restart','additional fruit/stem collision rejection'])

def generate_paper_bunch(seed=17,count=50,profile='cone',density='compact'):
    if not isinstance(count,int) or not 12<=count<=100:raise ValueError('count must be 12..100')
    if profile not in ('cone','shoulder','curved') or density not in ('compact','loose'):raise ValueError('Invalid morphology')
    failures=[]
    for attempt in range(5):
        initial_scale=1.2*(count/40)**(1/3)
        grammar=Grammar(seed+attempt*100003,count,profile,density,initial_scale+attempt*.09)
        success=grammar.run()
        if success:
            result=grammar.result(seed,count,attempt);result['previous_attempts']=failures;return result
        failures.append(dict(attempt=attempt,berries=len(grammar.models),rejections=grammar.rejections))
    raise RuntimeError('Restricted grammar did not reach requested count: '+json.dumps(failures))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=17);parser.add_argument('--count',type=int,default=50)
    parser.add_argument('--profile',default='cone');parser.add_argument('--density',default='compact');args=parser.parse_args()
    bunch=generate_paper_bunch(**vars(args));out=Path(__file__).resolve().parent/'paper_output';out.mkdir(exist_ok=True)
    path=out/(bunch['identity_id']+'.json');path.write_text(json.dumps(bunch,indent=2,ensure_ascii=False),encoding='utf-8')
    print(path);print(json.dumps(bunch['qa']))
