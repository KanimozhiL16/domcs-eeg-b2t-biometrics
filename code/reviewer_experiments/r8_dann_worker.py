# DANN single-seed worker (one GPU). Matched encoder + ArcFace(REST) + GRL state-adversary.
import os, sys, argparse, time
ap=argparse.ArgumentParser()
ap.add_argument("--seed",type=int,required=True); ap.add_argument("--gpu",type=int,required=True)
ap.add_argument("--data",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EP,BS,LR,WD=60,256,3e-4,1e-4; ARC_S,ARC_M=32.0,0.50
class CB(nn.Module):
    def __init__(s,ic,oc,k):
        super().__init__(); s.net=nn.Sequential(nn.Conv1d(ic,oc,k,padding=k//2,bias=False),nn.BatchNorm1d(oc),nn.ELU(inplace=True))
    def forward(s,x): return s.net(x)
class GR(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lam): ctx.lam=lam; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return g.neg()*ctx.lam, None
class DANN(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=nn.Sequential(CB(64,64,7),CB(64,128,5),CB(128,256,3),nn.AdaptiveAvgPool1d(1))
        s.id_fc=nn.Linear(256,128,bias=False); s.id_n=nn.LayerNorm(128)
        s.dom=nn.Sequential(nn.Linear(256,128),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(128,2))
    def forward(s,x,lam=0.0):
        f=s.enc(x).squeeze(-1); z=F.normalize(s.id_n(s.id_fc(f)),dim=-1); return z, s.dom(GR.apply(f,lam))
class ArcFace(nn.Module):
    def __init__(s,n=109):
        super().__init__(); s.W=nn.Parameter(torch.FloatTensor(n,128)); nn.init.xavier_uniform_(s.W)
    def forward(s,z,y):
        W=F.normalize(s.W,dim=1); c=F.linear(z,W).clamp(-1+1e-7,1-1e-7)
        oh=torch.zeros_like(c).scatter_(1,y.view(-1,1),1.0); return (c*(1-oh)+torch.cos(torch.acos(c)+ARC_M)*oh)*ARC_S
d=np.load(a.data,allow_pickle=True); X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
sess=np.array([int(str(s).lstrip("Rr")) for s in d["session"]],dtype=np.int64); state=(sess>=3).astype(np.int64)
torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed)
idx=rng.permutation(len(X)); nv=int(0.1*len(idx)); ti=idx[nv:]; Xt,yt,st=X[ti],y[ti],state[ti]
Xv=X[idx[:nv]]; yv=y[idx[:nv]]; sv=state[idx[:nv]]
m=DANN().to(DEV); arc=ArcFace().to(DEV); P=list(m.parameters())+list(arc.parameters())
opt=torch.optim.Adam(P,lr=LR,weight_decay=WD); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP,eta_min=1e-6)
best=1e9; bsd=None; t0=time.time()
for ep in range(EP):
    m.train(); arc.train(); perm=rng.permutation(len(Xt)); lam=2.0/(1.0+np.exp(-10*ep/EP))-1.0
    for i in range(0,len(perm),BS):
        b=perm[i:i+BS]; xb=torch.tensor(Xt[b],device=DEV); yb=torch.tensor(yt[b],device=DEV); sb=torch.tensor(st[b],device=DEV)
        r=(st[b]==0); opt.zero_grad(); z,dom=m(xb,lam); L=F.cross_entropy(dom,sb)
        if r.sum()>1: L=L+F.cross_entropy(arc(z[r],yb[r]),yb[r])
        L.backward(); torch.nn.utils.clip_grad_norm_(P,1.0); opt.step()
    sch.step()
    m.eval(); arc.eval(); vl=[]
    with torch.no_grad():
        for j in range(0,len(Xv),BS*4):
            xv=torch.tensor(Xv[j:j+BS*4],device=DEV); yvv=torch.tensor(yv[j:j+BS*4],device=DEV)
            rv=(sv[j:j+BS*4]==0); z,_=m(xv,0.0)
            if rv.sum()>1: vl.append(F.cross_entropy(arc(z[rv],yvv[rv]),yvv[rv]).item())
    v=float(np.mean(vl)) if vl else 1e9
    if v<best: best=v; bsd={k:vv.cpu().clone() for k,vv in m.state_dict().items()}
od=os.path.join(a.out,f"seed_{a.seed}"); os.makedirs(od,exist_ok=True)
torch.save({"model_state":bsd,"activation":"ELU","seed":a.seed,"best_val":best},os.path.join(od,"model_best.pt"))
print(f"DANN seed{a.seed} GPU{a.gpu} best_val={best:.4f} ({(time.time()-t0)/60:.1f}min)")
