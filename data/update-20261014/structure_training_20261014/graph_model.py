"""Small learned geometric graph encoder. RGB/texture never enters this module."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import numpy as np,torch
from torch import nn
torch.set_num_threads(2)
if torch.get_num_interop_threads()!=2:torch.set_num_interop_threads(2)
MAX_NODES=96;DIMENSION=128;FEATURES=9

def pack(nodes,seed=None):
    try:p=np.asarray([[n['x'],n['y']] for n in nodes],dtype=np.float32);r=np.asarray([n['radius'] for n in nodes],dtype=np.float32)
    except (TypeError,KeyError,ValueError):raise ValueError('Expected x,y,radius nodes')
    if len(nodes)<6 or len(nodes)>160 or not np.isfinite(p).all() or not np.isfinite(r).all() or (r<=0).any():raise ValueError('Need 6–160 finite positive-radius nodes')
    if len(p)>MAX_NODES:
        chosen=np.argsort(-r,kind='stable')[:MAX_NODES];p=p[chosen];r=r[chosen]
    if seed is not None:
        rng=np.random.default_rng(seed);keep=rng.random(len(p))>rng.uniform(0,.12)
        if keep.sum()>=6:p=p[keep];r=r[keep]
        median=float(np.median(r));p=p+rng.normal(0,.045*median,p.shape);r=r*rng.uniform(.94,1.06,len(r))
        angle=rng.uniform(-.18,.18);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        p=(p-p.mean(0))@rot.T
    median=max(float(np.median(r)),1e-6);p=p-p.mean(0);extent=max(float(np.max(np.ptp(p,axis=0))),2*median)
    distance=np.linalg.norm(p[:,None]-p[None,:],axis=2);np.fill_diagonal(distance,np.inf)
    near=np.sort(distance,axis=1)[:,:3]/median
    x=np.c_[p/extent,np.log(np.clip(r/median,.1,5)),np.clip(near/6,0,3),np.full(len(p),len(p)/MAX_NODES),
            np.linalg.norm(p,axis=1)/extent,r/extent].astype(np.float32)
    result=np.zeros((MAX_NODES,FEATURES),np.float32);mask=np.zeros(MAX_NODES,bool);result[:len(x)]=x;mask[:len(x)]=True
    return result,mask

class GraphNet(nn.Module):
    def __init__(self,nclass):
        super().__init__();self.node=nn.Sequential(nn.Linear(FEATURES,48),nn.LayerNorm(48),nn.ReLU())
        self.edges=nn.ModuleList([nn.Sequential(nn.Linear(100,64),nn.ReLU(),nn.Linear(64,48),nn.LayerNorm(48),nn.ReLU()) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(96,DIMENSION),nn.LayerNorm(DIMENSION),nn.ReLU(),nn.Linear(DIMENSION,DIMENSION))
        self.classifier=nn.Linear(DIMENSION,nclass)
    def forward(self,x,mask):
        mask=mask.bool();b,n,_=x.shape
        d=torch.cdist(x[:,:,:2],x[:,:,:2]);d=d.masked_fill(~mask[:,None,:],1e6)
        d=d.masked_fill(torch.eye(n,dtype=torch.bool,device=x.device)[None],1e6);indices=d.topk(3,largest=False).indices
        batch=torch.arange(b,device=x.device)[:,None,None];neighbors=x[batch,indices]
        local=neighbors[:,:,:,:3]-x[:,:,:3,None].transpose(2,3)
        relative=torch.cat([local,torch.linalg.vector_norm(local[:,:,:,:2],dim=-1,keepdim=True)],dim=-1)
        h=self.node(x)
        for layer in self.edges:
            delta=h[batch,indices]-h[:,:,None,:];center=h[:,:,None,:].expand(-1,-1,3,-1)
            h=h+layer(torch.cat([center,delta,relative],dim=-1)).max(dim=2).values
            h=h*mask[:,:,None]
        mean=h.sum(1)/mask.sum(1,keepdim=True).clamp_min(1);maximum=h.masked_fill(~mask[:,:,None],-1e6).max(1).values
        return nn.functional.normalize(self.head(torch.cat([mean,maximum],1)),dim=1)
