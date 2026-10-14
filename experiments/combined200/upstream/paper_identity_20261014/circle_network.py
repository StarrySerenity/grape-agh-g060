"""CircleNet-style center/radius heads with a local frozen ResNet18 backbone.

Engineering adaptation, not the original CircleNet/berry authors' implementation.
"""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL';os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(2)

class CircleDetector(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        from torchvision.models import resnet18
        net=resnet18(weights=None)
        if pretrained:
            from pathlib import Path
            path=Path('C:/Users/zhq/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
            net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False))
        self.stem=nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool)
        self.layer1=net.layer1;self.layer2=net.layer2;self.layer3=net.layer3
        for m in [self.stem,self.layer1,self.layer2,self.layer3]:
            for p in m.parameters():p.requires_grad=False
        self.laterals=nn.ModuleList([nn.Conv2d(c,16,1) for c in [64,128,256]])
        self.decoder=nn.Sequential(nn.Conv2d(48,48,3,padding=1),nn.ReLU(),nn.Conv2d(48,48,3,padding=1,groups=48),nn.ReLU())
        self.output=nn.Conv2d(48,5,1)
        nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)
        self.output.bias.data[0]=-2.19

    def features(self,x):
        # Keep pretrained normalization stable even while heads train.
        self.stem.eval();self.layer1.eval();self.layer2.eval();self.layer3.eval()
        with torch.no_grad():
            a=self.layer1(self.stem(x));b=self.layer2(a);c=self.layer3(b)
        return a,b,c

    def heads(self,features):
        size=features[0].shape[-2:]
        latent=torch.cat([F.interpolate(conv(f.float()),size=size,mode='bilinear',align_corners=False) for conv,f in zip(self.laterals,features)],1)
        out=self.output(self.decoder(latent))
        return {'heatmap':out[:,0:1],'offset':out[:,1:3],
                'radius':F.softplus(out[:,3:4])+.05,'visibility':out[:,4:5]}

    def forward(self,x):return self.heads(self.features(x))

def targets(nodes,size=256):
    h=size//4;hm=np.zeros((1,h,h),np.float32);radius=hm.copy();visibility=hm.copy();mask=hm.copy();offset=np.zeros((2,h,h),np.float32)
    for n in nodes:
        cx,cy=float(n['x'])/4,float(n['y'])/4;x,y=int(cx),int(cy)
        if not 0<=x<h or not 0<=y<h:continue
        rad=float(n['radius'])/4;sigma=max(.65,min(2.5,rad*.22));extent=int(np.ceil(3*sigma))
        x0,x1=max(0,x-extent),min(h,x+extent+1);y0,y1=max(0,y-extent),min(h,y+extent+1)
        yy,xx=np.mgrid[y0:y1,x0:x1];gauss=np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma))
        hm[0,y0:y1,x0:x1]=np.maximum(hm[0,y0:y1,x0:x1],gauss)
        radius[0,y,x]=rad;offset[:,y,x]=[cx-x,cy-y];visibility[0,y,x]=n.get('visibility',1.);mask[0,y,x]=1
    return dict(heatmap=hm,radius=radius,offset=offset,visibility=visibility,mask=mask)

def loss(pred,t,weak=False):
    prob=pred['heatmap'].sigmoid().clamp(1e-5,1-1e-5);gt=t['heatmap']
    positive=(gt==1).float();negative=(gt<1).float();weight=(1-gt)**4
    # Unknown, unannotated real berries must not become confident negative labels.
    negative_valid=t.get('negative_valid',torch.ones_like(gt)) if weak else torch.ones_like(gt)
    hl=-(torch.log(prob)*(1-prob)**2*positive+torch.log(1-prob)*prob**2*weight*negative*negative_valid).sum()/positive.sum().clamp(min=1)
    mask=t['mask'];count=mask.sum().clamp(min=1)
    rl=(F.smooth_l1_loss(pred['radius'],t['radius'],reduction='none')*mask).sum()/count
    ol=(F.smooth_l1_loss(pred['offset'],t['offset'],reduction='none')*mask).sum()/count
    vl=(F.binary_cross_entropy_with_logits(pred['visibility'],t['visibility'],reduction='none')*mask).sum()/count
    return hl+.15*rl+.5*ol+.2*vl,dict(heatmap=float(hl.detach()),radius=float(rl.detach()),offset=float(ol.detach()),visibility=float(vl.detach()))

def decode(pred,threshold=.20,max_nodes=100):
    hm=pred['heatmap'].detach().sigmoid();peaks=(F.max_pool2d(hm,3,stride=1,padding=1)==hm)
    values=(hm*peaks)[0,0].cpu().numpy();yy,xx=np.where(values>=threshold)
    order=np.argsort(values[yy,xx])[::-1][:max_nodes];nodes=[]
    radius=pred['radius'].detach()[0,0].cpu().numpy();offset=pred['offset'].detach()[0].cpu().numpy()
    vis=pred['visibility'].detach().sigmoid()[0,0].cpu().numpy()
    for k in order:
        y,x=yy[k],xx[k];r=float(radius[y,x]*4)
        cx=float((x+offset[0,y,x])*4);cy=float((y+offset[1,y,x])*4)
        if not np.isfinite([cx,cy,r]).all() or not 2<=r<=70:continue
        if any(np.hypot(cx-n['x'],cy-n['y'])<1.0*max(r,n['radius']) for n in nodes):continue
        nodes.append(dict(x=cx,y=cy,radius=r,confidence=float(values[y,x]),visibility=float(vis[y,x]),center_type='trained circle-center regression'))
    return nodes
