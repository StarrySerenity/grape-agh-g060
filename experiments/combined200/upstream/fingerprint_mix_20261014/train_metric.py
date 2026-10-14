"""Frozen ImageNet CPU encoder + trained 256D metric head; no old task weights."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,copy,hashlib,json,math,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torchvision import models
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent))
from fingerprint_2d import imread,masked_crop
from protocol import retrieval
torch.set_num_threads(2)
if torch.get_num_interop_threads()!=2:torch.set_num_interop_threads(2)
cv2.setNumThreads(2)
IMAGENET=Path('C:/Users/zhq/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')

class Net(nn.Module):
    def __init__(self,nclass):
        super().__init__();base=models.resnet18(weights=None)
        base.load_state_dict(torch.load(IMAGENET,map_location='cpu',weights_only=True))
        self.backbone=nn.Sequential(*list(base.children())[:-1])
        self.proj=nn.Sequential(nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(inplace=True),nn.Dropout(.15))
        self.classifier=nn.Linear(256,nclass)
        for p in self.backbone.parameters():p.requires_grad=False
    def embedding(self,features):return nn.functional.normalize(self.proj(features),dim=1)

def triplet(e,labels,margin=.2):
    d=torch.cdist(e,e);same=labels[:,None]==labels[None];positive=same&~torch.eye(len(e),dtype=torch.bool)
    negative=~same;valid=positive.any(1)&negative.any(1)
    if not valid.any():return e.sum()*0
    pos=d.masked_fill(~positive,-1).max(1).values;neg=d.masked_fill(~negative,1e6).min(1).values
    return torch.relu(pos[valid]-neg[valid]+margin).mean()

def input_tensor(row,augment_seed=None):
    image=imread(row['image']);mask=imread(row['mask'],True);crop,m=masked_crop(image,mask)
    if augment_seed is not None:
        rng=np.random.default_rng(augment_seed)
        if rng.random()<.5:crop=np.ascontiguousarray(crop[:,::-1]);m=np.ascontiguousarray(m[:,::-1])
        gain=rng.uniform(.86,1.14);channels=rng.uniform(.95,1.05,3)
        crop=np.clip(crop.astype(float)*gain*channels,0,255).astype(np.uint8);crop[m==0]=128
    rgb=cv2.cvtColor(cv2.resize(crop,(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return torch.from_numpy(((rgb-[.485,.456,.406])/[.229,.224,.225]).astype(np.float32).transpose(2,0,1))

def encode_rows(network,rows,label):
    features=[];network.backbone.eval();start=time.time()
    with torch.inference_mode():
        for i in range(0,len(rows),12):
            tensors=[input_tensor(row,seed) for row,seed in rows[i:i+12]]
            features.extend(network.backbone(torch.stack(tensors)).flatten(1).numpy())
            if i%240==0:print(label,i,'/',len(rows),'elapsed',round(time.time()-start),flush=True)
    return np.asarray(features,dtype=np.float32)

def extract(kind):
    network=Net(1);network.eval()
    if kind=='real':
        rows=json.loads((ROOT/'real_rows.json').read_text(encoding='utf-8'));split=json.loads((ROOT/'real_protocol.json').read_text(encoding='utf-8'))
        for row in rows:
            if hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()!=row['image_sha256']:raise RuntimeError('Source image changed')
            if hashlib.sha256(Path(row['mask']).read_bytes()).hexdigest()!=row['mask_sha256']:raise RuntimeError('Source mask changed')
        x=encode_rows(network,[(r,None) for r in rows],'Real RGB')
        aug=encode_rows(network,[(rows[i],73001+i) for i in split['train_indices']],'Real augmentation')
        np.savez(ROOT/'real_features.npz',features=x,train_aug=aug)
    else:
        records=[]
        for index in range(100):
            rec=json.loads((ROOT/'synthetic'/f'SYN_{index:03d}'/'records.json').read_text(encoding='utf-8'));records.extend(rec['rows'])
        if len({r['identity'] for r in records})!=100 or len(records)!=2400:raise RuntimeError('Expected 100 synthetic identities x 24 views')
        if len({r['geometry_hash'] for r in records})!=100:raise RuntimeError('Duplicate synthetic geometry identities')
        x=encode_rows(network,[(r,None) for r in records],'Synthetic RGB')
        np.savez(ROOT/'synthetic_features.npz',features=x)
        (ROOT/'synthetic_rows.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('FEATURES_READY',kind,flush=True)

def embed(network,x):
    network.proj.eval()
    with torch.inference_mode():return network.embedding(torch.as_tensor(x,dtype=torch.float32)).numpy()

def balanced_indices(labels,rng,p=8,k=4):
    pools=collections.defaultdict(list)
    for i,label in enumerate(labels):pools[int(label)].append(i)
    chosen=rng.choice(list(pools),min(p,len(pools)),replace=False)
    return np.array([i for label in chosen for i in rng.choice(pools[int(label)],k,replace=len(pools[int(label)])<k)])

def evaluate_validation(net,real,rows,split):
    qi=split['val_query_indices'];gi=split['val_gallery_indices']
    return retrieval(embed(net,real[qi]),[rows[i]['identity'] for i in qi],embed(net,real[gi]),[rows[i]['identity'] for i in gi])

def train(name,weight,epochs=50):
    rows=json.loads((ROOT/'real_rows.json').read_text(encoding='utf-8'));split=json.loads((ROOT/'real_protocol.json').read_text(encoding='utf-8'))
    synrows=json.loads((ROOT/'synthetic_rows.json').read_text(encoding='utf-8'));real=np.load(ROOT/'real_features.npz');syn=np.load(ROOT/'synthetic_features.npz')['features']
    classes={identity:i for i,identity in enumerate(split['train_ids']+sorted({r['identity'] for r in synrows}))};nreal=len(split['train_ids'])
    torch.manual_seed(318);net=Net(len(classes));nn.init.zeros_(net.classifier.bias)
    opt=torch.optim.AdamW(list(net.proj.parameters())+list(net.classifier.parameters()),lr=4e-4,weight_decay=1e-4)
    ix=split['train_indices'];xr=np.vstack([real['features'][ix],real['train_aug']]);lr=np.array([classes[rows[i]['identity']] for i in ix]*2)
    ls=np.array([classes[r['identity']] for r in synrows]);xr=torch.from_numpy(xr);xs=torch.from_numpy(syn)
    yl=torch.tensor(lr);ys=torch.tensor(ls);history=[];bestscore=-1;beststate=None;bestepoch=None;out=ROOT/'models'/name;out.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        net.proj.train();total=[]
        for step in range(32):
            ri=balanced_indices(lr,np.random.default_rng(160001+epoch*100+step));x=xr[ri]
            if weight>0:
                si=balanced_indices(ls,np.random.default_rng(900003+epoch*100+step));x=torch.cat([x,xs[si]])
            opt.zero_grad();e=net.embedding(x);logits=net.classifier(e)
            loss=nn.functional.cross_entropy(logits[:len(ri),:nreal],yl[ri],label_smoothing=.03)+.5*triplet(e[:len(ri)],yl[ri])
            if weight>0:loss=loss+weight*(nn.functional.cross_entropy(logits[len(ri):,nreal:],ys[si]-nreal,label_smoothing=.03)+.5*triplet(e[len(ri):],ys[si]))
            loss.backward();opt.step();total.append(float(loss))
        val=evaluate_validation(net,real['features'],rows,split);score=.6*val['rank1']+.4*val['map']
        result=dict(epoch=epoch,loss=float(np.mean(total)),val_rank1=val['rank1'],val_rank5=val['rank5'],val_map=val['map'],selection_score=score)
        history.append(result)
        if score>bestscore+1e-9:bestscore=score;beststate=copy.deepcopy(net.state_dict());bestepoch=epoch;bestmetrics=val
        if epoch==1 or epoch%5==0:print(name,json.dumps(result),flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    state=dict(model=beststate,epoch=bestepoch,classes=classes,metrics=bestmetrics,selected_by='validation identities only',
               train_ids=split['train_ids'],synthetic_identities=100 if weight>0 else 0,synthetic_weight=weight,
               backbone='frozen ImageNet ResNet18 CPU',imagenet_checkpoint=str(IMAGENET),imagenet_sha256=hashlib.sha256(IMAGENET.read_bytes()).hexdigest(),
               test_ids_never_trained=split['test_ids'],test_not_used_in_selection=True)
    torch.save(state,out/'best.pt');(out/'validation.json').write_text(json.dumps(dict(score=bestscore,epoch=bestepoch,metrics=bestmetrics,weight=weight),indent=2),encoding='utf-8')
    print('TRAINED',name,'best epoch',bestepoch,'validation score',bestscore,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--features',choices=['real','synthetic']);parser.add_argument('--run');parser.add_argument('--weight',type=float,default=0);parser.add_argument('--epochs',type=int,default=50);args=parser.parse_args()
    if args.features:extract(args.features)
    elif args.run:train(args.run,args.weight,args.epochs)
    else:parser.error('Choose --features or --run')
