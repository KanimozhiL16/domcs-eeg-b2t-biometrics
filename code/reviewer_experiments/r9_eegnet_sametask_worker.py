# EEGNet same-task single-seed worker (one GPU). Writes seed json: id-acc + verif EER.
import os, sys, argparse, json, time
ap=argparse.ArgumentParser()
ap.add_argument("--seed",type=int,required=True); ap.add_argument("--gpu",type=int,required=True)
ap.add_argument("--data",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_curve
DEV=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); EP,BS=40,256
class EEGNet(nn.Module):
    def __init__(s,C=64,T=256,n=109,F1=8,D=2,F2=16,kl=64,drop=0.25):
        super().__init__()
        s.b1=nn.Sequential(nn.Conv2d(1,F1,(1,kl),padding=(0,kl//2),bias=False),nn.BatchNorm2d(F1))
        s.dw=nn.Sequential(nn.Conv2d(F1,F1*D,(C,1),groups=F1,bias=False),nn.BatchNorm2d(F1*D),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(drop))
        s.sep=nn.Sequential(nn.Conv2d(F1*D,F1*D,(1,16),padding=(0,8),groups=F1*D,bias=False),nn.Conv2d(F1*D,F2,(1,1),bias=False),
                            nn.BatchNorm2d(F2),nn.ELU(),nn.AvgPool2d((1,8)),nn.Dropout(drop))
        s.flat=nn.Flatten(); s.cls=nn.Linear(F2*(T//32),n)
    def feat(s,x): return s.flat(s.sep(s.dw(s.b1(x))))
    def forward(s,x): return s.cls(s.feat(x))
d=np.load(a.data,allow_pickle=True); X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed)
idx=rng.permutation(len(X)); cut=int(0.8*len(idx)); tr,te=idx[:cut],idx[cut:]
Xtr,ytr=X[tr],y[tr]; Xte,yte=X[te],y[te]
m=EEGNet().to(DEV); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4); t0=time.time()
for ep in range(EP):
    m.train(); perm=rng.permutation(len(Xtr))
    for i in range(0,len(perm),BS):
        b=perm[i:i+BS]; xb=torch.tensor(Xtr[b],device=DEV).unsqueeze(1); yb=torch.tensor(ytr[b],device=DEV)
        opt.zero_grad(); F.cross_entropy(m(xb),yb).backward(); opt.step()
m.eval()
with torch.no_grad():
    pr=[]; fte=[]
    for i in range(0,len(Xte),1024):
        xb=torch.tensor(Xte[i:i+1024],device=DEV).unsqueeze(1); pr.append(m(xb).argmax(1).cpu().numpy()); fte.append(F.normalize(m.feat(xb),dim=1).cpu().numpy())
    pred=np.concatenate(pr); acc=float((pred==yte).mean()*100); fte=np.concatenate(fte)
    ftr=[]
    for i in range(0,len(Xtr),1024):
        xb=torch.tensor(Xtr[i:i+1024],device=DEV).unsqueeze(1); ftr.append(F.normalize(m.feat(xb),dim=1).cpu().numpy())
    ftr=np.concatenate(ftr)
subs=np.unique(y); C=np.stack([ (lambda c:c/(np.linalg.norm(c)+1e-9))(ftr[ytr==u].mean(0)) for u in subs]); ix={u:i for i,u in enumerate(subs)}
g,im=[],[]
for i in range(len(fte)):
    sims=C@fte[i]; ti=ix[yte[i]]; g.append(sims[ti]); im.extend(np.delete(sims,ti).tolist())
sc=np.concatenate([np.array(g),np.array(im)]); lb=np.concatenate([np.ones(len(g)),np.zeros(len(im))])
fpr,tpr,_=roc_curve(lb,sc); fnr=1-tpr; eer=float(fpr[np.nanargmin(np.abs(fpr-fnr))]*100)
os.makedirs(a.out,exist_ok=True)
json.dump({"seed":a.seed,"id_acc":acc,"verif_eer":eer},open(os.path.join(a.out,f"seed_{a.seed}.json"),"w"))
print(f"EEGNet seed{a.seed} GPU{a.gpu} same-task id-acc={acc:.2f}% verif-EER={eer:.2f}% ({(time.time()-t0)/60:.1f}min)")
