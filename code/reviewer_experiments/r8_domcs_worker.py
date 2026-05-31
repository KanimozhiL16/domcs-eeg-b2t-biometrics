# DOMCS single-seed worker (one GPU). Called by RUN_ALL_8GPU.py. Sets GPU before importing torch.
import os, sys, argparse, time
ap=argparse.ArgumentParser()
ap.add_argument("--seed",type=int,required=True); ap.add_argument("--gpu",type=int,required=True)
ap.add_argument("--data",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EP,BS,LR,WD=60,256,3e-4,1e-4; ARC_S,ARC_M,TAU=32.0,0.50,0.07; LAM_ST,LAM_OR,LAM_SU=0.50,0.10,0.30
class CB(nn.Module):
    def __init__(s,ic,oc,k):
        super().__init__(); s.net=nn.Sequential(nn.Conv1d(ic,oc,k,padding=k//2,bias=False),nn.BatchNorm1d(oc),nn.ELU(inplace=True))
    def forward(s,x): return s.net(x)
class DOMCS(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=nn.Sequential(CB(64,64,7),CB(64,128,5),CB(128,256,3),nn.AdaptiveAvgPool1d(1))
        s.id_fc=nn.Linear(256,128,bias=False); s.id_n=nn.LayerNorm(128)
        s.st_fc=nn.Linear(256,128,bias=False); s.st_n=nn.LayerNorm(128)
    def forward(s,x):
        f=s.enc(x).squeeze(-1); return (F.normalize(s.id_n(s.id_fc(f)),dim=-1), F.normalize(s.st_n(s.st_fc(f.detach())),dim=-1))
class ArcFace(nn.Module):
    def __init__(s,n=109):
        super().__init__(); s.W=nn.Parameter(torch.FloatTensor(n,128)); nn.init.xavier_uniform_(s.W)
    def forward(s,z,y):
        W=F.normalize(s.W,dim=1); c=F.linear(z,W).clamp(-1+1e-7,1-1e-7)
        oh=torch.zeros_like(c).scatter_(1,y.view(-1,1),1.0); return (c*(1-oh)+torch.cos(torch.acos(c)+ARC_M)*oh)*ARC_S
def supcon(z,y,tau=TAU):
    n=z.size(0); z=F.normalize(z,dim=1); sim=torch.mm(z,z.T)/tau; sim=sim-sim.detach().max(1,True)[0]
    eye=torch.eye(n,device=z.device); m=y.unsqueeze(0).eq(y.unsqueeze(1)).float()*(1-eye); p=m.sum(1).clamp(min=1)
    e=torch.exp(sim)*(1-eye); lp=sim-torch.log(e.sum(1,True)+1e-9); return (-(m*lp).sum(1)/p).mean()
def orth(zi,zs): return torch.abs(F.cosine_similarity(zi,zs,dim=-1)).mean()
d=np.load(a.data,allow_pickle=True); X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
sess=np.array([int(str(s).lstrip("Rr")) for s in d["session"]],dtype=np.int64); state=(sess>=3).astype(np.int64)
torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed)
idx=rng.permutation(len(X)); nv=int(0.1*len(idx)); ti=idx[nv:]; Xt,yt,st=X[ti],y[ti],state[ti]
m=DOMCS().to(DEV); arc=ArcFace().to(DEV); sh=nn.Linear(128,2).to(DEV)
P=list(m.parameters())+list(arc.parameters())+list(sh.parameters())
opt=torch.optim.Adam(P,lr=LR,weight_decay=WD); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP,eta_min=1e-6)
Xv=X[idx[:nv]]; yv=y[idx[:nv]]; sv=state[idx[:nv]]; best=1e9; bsd=None; t0=time.time()
for ep in range(EP):
    m.train(); perm=rng.permutation(len(Xt))
    for i in range(0,len(perm),BS):
        b=perm[i:i+BS]; xb=torch.tensor(Xt[b],device=DEV); yb=torch.tensor(yt[b],device=DEV); sb=torch.tensor(st[b],device=DEV)
        r=(st[b]==0); opt.zero_grad(); zi,zs=m(xb); L=LAM_ST*F.cross_entropy(sh(zs),sb)+LAM_OR*orth(zi,zs)
        if r.sum()>1: L=L+F.cross_entropy(arc(zi[r],yb[r]),yb[r])+LAM_SU*supcon(zi[r],yb[r])
        L.backward(); torch.nn.utils.clip_grad_norm_(P,1.0); opt.step()
    sch.step()
    m.eval(); vl=[]
    with torch.no_grad():
        for j in range(0,len(Xv),BS*4):
            xv=torch.tensor(Xv[j:j+BS*4],device=DEV); yvv=torch.tensor(yv[j:j+BS*4],device=DEV); svv=torch.tensor(sv[j:j+BS*4],device=DEV)
            rv=(sv[j:j+BS*4]==0); zi,zs=m(xv); v=LAM_ST*F.cross_entropy(sh(zs),svv)+LAM_OR*orth(zi,zs)
            if rv.sum()>1: v=v+F.cross_entropy(arc(zi[rv],yvv[rv]),yvv[rv])+LAM_SU*supcon(zi[rv],yvv[rv])
            vl.append(v.item())
    v=float(np.mean(vl))
    if v<best: best=v; bsd={k:vv.cpu().clone() for k,vv in m.state_dict().items()}
od=os.path.join(a.out,f"seed_{a.seed}"); os.makedirs(od,exist_ok=True)
torch.save({"model_state":bsd,"activation":"ELU","seed":a.seed,"best_val":best},os.path.join(od,"model_best.pt"))
print(f"DOMCS seed{a.seed} GPU{a.gpu} best_val={best:.4f} ({(time.time()-t0)/60:.1f}min)")
