#!/usr/bin/env python3
# =====================================================================================
#  R6b — BED EXTERNAL VALIDATION with EXACT EEGMMIDB worker architecture (Cin=14)
#  Trains DOMCS from scratch on BED using the IDENTICAL architecture, losses, and
#  hyperparameters as ablation_worker_elu.py (E1 Full) — the ONLY change is the first
#  conv accepts 14 channels instead of 64. Protocol = validated Colab: enroll=r01+r02,
#  probe=r03 (cross-session). Exhaustive all-impostor EER (same scorer as R1-R3). 5 seeds.
#  Goal: confirm the ~22.93% BED result holds with the architecture identical to the
#  main model, so the paper can state "same model, 14-ch input" honestly.
# =====================================================================================
import os, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, roc_auc_score

BED   = "/home/nvidia/24PHD1237/BED_DATASET/BED_win2s_step1s_fs128.npz"
OUT   = "/home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG/R6b_bed_exactarch"; os.makedirs(OUT,exist_ok=True)
DEV   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEEDS = [1,2,3,4,5]; K=3
# EXACT worker hyperparameters (ablation_worker_elu.py)
N_EPOCHS,BS,LR,WD = 60,256,3e-4,1e-4
ARC_S,ARC_M,TAU   = 32.0,0.50,0.07
LAM_STATE,LAM_ORTH,LAM_SUP = 0.50,0.10,0.30

# ---------- EXACT worker architecture (only Cin differs: 14 not 64) ----------
class ConvBlock(nn.Module):
    def __init__(s,ic,oc,k):
        super().__init__(); s.net=nn.Sequential(nn.Conv1d(ic,oc,k,padding=k//2,bias=False),
                                                nn.BatchNorm1d(oc),nn.ELU(inplace=True))
    def forward(s,x): return s.net(x)
def _enc(cin): return nn.Sequential(ConvBlock(cin,64,7),ConvBlock(64,128,5),
                                    ConvBlock(128,256,3),nn.AdaptiveAvgPool1d(1))
class DOMCSStandard(nn.Module):                 # f.detach() in state branch (E1 Full)
    def __init__(s,cin=14):
        super().__init__(); s.enc=_enc(cin)
        s.id_fc=nn.Linear(256,128,bias=False); s.id_n=nn.LayerNorm(128)
        s.st_fc=nn.Linear(256,128,bias=False); s.st_n=nn.LayerNorm(128)
    def forward(s,x):
        f=s.enc(x).squeeze(-1)
        return (F.normalize(s.id_n(s.id_fc(f)),          dim=-1),
                F.normalize(s.st_n(s.st_fc(f.detach())), dim=-1))
class ArcFaceHead(nn.Module):                   # worker-exact: s=32, m=0.50
    def __init__(s,n):
        super().__init__(); s.s=ARC_S; s.m=ARC_M
        s.W=nn.Parameter(torch.FloatTensor(n,128)); nn.init.xavier_uniform_(s.W)
    def forward(s,z,y):
        W=F.normalize(s.W,dim=1); c=F.linear(z,W).clamp(-1+1e-7,1-1e-7)
        oh=torch.zeros_like(c).scatter_(1,y.view(-1,1),1.0)
        return (c*(1-oh)+torch.cos(torch.acos(c)+s.m)*oh)*s.s
class StateHead(nn.Module):
    def __init__(s): super().__init__(); s.fc=nn.Linear(128,2)
    def forward(s,z): return s.fc(z)
def supcon(z,y,tau=TAU):                         # worker-exact
    n=z.size(0); z=F.normalize(z,dim=1); sim=torch.mm(z,z.T)/tau
    sim=sim-sim.detach().max(1,keepdim=True)[0]; eye=torch.eye(n,device=z.device)
    mask=y.unsqueeze(0).eq(y.unsqueeze(1)).float()*(1-eye); pos=mask.sum(1).clamp(min=1)
    exp=torch.exp(sim)*(1-eye); log_p=sim-torch.log(exp.sum(1,keepdim=True)+1e-9)
    return (-(mask*log_p).sum(1)/pos).mean()
def orth_loss(zi,zs): return torch.abs(F.cosine_similarity(zi,zs,dim=-1)).mean()  # |cos|, worker-exact

# ---------- BED data: enroll=r01+r02, probe=r03 (validated protocol) ----------
d=np.load(BED,allow_pickle=True)
X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
runs=np.asarray([str(r) for r in (d["session"] if "session" in d.files else d["runs"])],dtype=object)
assert X.shape[1]==14 and X.shape[2]==256 and "r03" in set(runs)
SUBJ=sorted(np.unique(y)); s2i={s:i for i,s in enumerate(SUBJ)}; NS=len(SUBJ)
tr_mask=np.isin(runs,["r01","r02"]); te_mask=(runs=="r03")
Xtr_all,ytr_all=X[tr_mask],y[tr_mask]; Xte,yte=X[te_mask],y[te_mask]
print(f"BED: X={X.shape} subj={NS} | enroll(r01+r02)={tr_mask.sum()} probe(r03)={te_mask.sum()}")

@torch.no_grad()
def embed(m,Xn,bs=2048):
    m.eval(); o=[]
    for i in range(0,len(Xn),bs):
        zi,_=m(torch.from_numpy(Xn[i:i+bs]).to(DEV)); o.append(zi.cpu().numpy())
    return np.concatenate(o)
def exhaustive_eer(m):
    zg=embed(m,Xtr_all); zt=embed(m,Xte)
    G=np.zeros((NS,K,128),np.float32)
    for s in SUBJ:
        c=KMeans(K,n_init=10,random_state=0).fit(zg[ytr_all==s]).cluster_centers_
        G[s2i[s]]=c/(np.linalg.norm(c,axis=1,keepdims=True)+1e-12)
    Gt=torch.tensor(G.reshape(NS*K,128),device=DEV); P=torch.tensor(zt,device=DEV); gen=[]; imp=[]
    with torch.no_grad():
        for i in range(0,len(P),8192):
            S=(P[i:i+8192]@Gt.T).view(-1,NS,K).max(2).values.cpu().numpy()
            yb=yte[i:i+8192]; r=np.arange(len(yb)); gi=np.array([s2i[s] for s in yb])
            gen.append(S[r,gi]); M=S.copy(); M[r,gi]=np.nan; imp.append(M[~np.isnan(M)])
    gen=np.concatenate(gen); imp=np.concatenate(imp)
    sc=np.r_[gen,imp].astype(np.float64); lb=np.r_[np.ones_like(gen),np.zeros_like(imp)]
    fpr,tpr,_=roc_curve(lb,sc); fnr=1-tpr; i=np.nanargmin(np.abs(fnr-fpr))
    return (fpr[i]+fnr[i])/2*100, float(roc_auc_score(lb,sc))

def train_seed(seed):
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    idx=rng.permutation(len(Xtr_all)); nval=int(0.10*len(idx)); vi,ti=idx[:nval],idx[nval:]
    Xt,yt=Xtr_all[ti],ytr_all[ti]; Xv,yv=Xtr_all[vi],ytr_all[vi]
    st=np.zeros(len(yt),np.int64)                      # all enroll-state (worker: identity on state0)
    m=DOMCSStandard(14).to(DEV); arc=ArcFaceHead(NS).to(DEV); sh=StateHead().to(DEV)
    P=list(m.parameters())+list(arc.parameters())+list(sh.parameters())
    opt=torch.optim.Adam(P,lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=N_EPOCHS,eta_min=1e-6)
    best=1e9; best_sd=None
    for ep in range(1,N_EPOCHS+1):
        m.train(); perm=rng.permutation(len(Xt))
        for i in range(0,len(perm),BS):
            b=perm[i:i+BS]; xb=torch.FloatTensor(Xt[b]).to(DEV); yb=torch.LongTensor(yt[b]).to(DEV)
            sb=torch.LongTensor(st[b]).to(DEV)
            opt.zero_grad(); zi,zs=m(xb)
            loss=F.cross_entropy(arc(zi,yb),yb)+LAM_SUP*supcon(zi,yb)
            loss=loss+LAM_STATE*F.cross_entropy(sh(zs),sb)+LAM_ORTH*orth_loss(zi,zs)
            loss.backward(); torch.nn.utils.clip_grad_norm_(P,1.0); opt.step()
        sch.step()
        # val (identity arc+supcon only, proxy)
        m.eval()
        with torch.no_grad():
            vl=[]
            for i in range(0,len(Xv),BS*4):
                xv=torch.FloatTensor(Xv[i:i+BS*4]).to(DEV); yvb=torch.LongTensor(yv[i:i+BS*4]).to(DEV)
                ziv,_=m(xv); vl.append((F.cross_entropy(arc(ziv,yvb),yvb)+LAM_SUP*supcon(ziv,yvb)).item())
            v=float(np.mean(vl))
        if v<best: best=v; best_sd={k:vv.cpu().clone() for k,vv in m.state_dict().items()}
    m.load_state_dict(best_sd); return m

print("\nR6b: BED with EXACT worker architecture (Cin=14), 5 seeds")
ee=[]; aa=[]
for s in SEEDS:
    t0=time.time(); m=train_seed(s); e,a=exhaustive_eer(m)
    ee.append(e); aa.append(a); print(f"  seed {s}: EER={e:.3f}%  AUC={a:.4f}  ({time.time()-t0:.0f}s)")
ee=np.array(ee); aa=np.array(aa)
print(f"\n  DOMCS-EEG (exact arch) BED: EER={ee.mean():.2f} +/- {ee.std(ddof=1):.2f}%   AUC={aa.mean():.4f}")
json.dump({"eer":ee.tolist(),"auc":aa.tolist(),"eer_mean":[float(ee.mean()),float(ee.std(ddof=1))],
           "auc_mean":float(aa.mean()),"arch":"exact worker DOMCSStandard, Cin=14, s=32, orth=|cos|, 3 conv blocks"},
          open(OUT+"/bed_exactarch_results.json","w"),indent=2)
print(f"Saved -> {OUT}/bed_exactarch_results.json")
print("Compare to Colab simplified-arch result EER=22.93%. Same enroll(r01+r02)/probe(r03), exhaustive scorer.")
