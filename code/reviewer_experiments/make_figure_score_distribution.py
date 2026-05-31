# =====================================================================================
#  FIG 9 (FAST) — Genuine vs Impostor cosine distribution, GPU-vectorized exhaustive scorer.
#  SAME logic as before (each probe vs ALL subjects, max over K prototypes, all-impostor)
#  but scoring is batched matmul on GPU -> seconds, not minutes. Trains seed3 inline if no ckpt.
# =====================================================================================
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans; from sklearn.metrics import roc_curve
mpl.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif","font.size":9,
  "axes.titlesize":9,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
  "figure.dpi":300,"savefig.bbox":"tight"})
DATA = "/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz"   # <-- EDIT IF NEEDED
CKPT = "/home/nvidia/24PHD1237/R8_domcs_10seed/seed_3/model_best.pt"               # absolute; trained inline if absent
OUT  = "figures_TIFS"; os.makedirs(OUT, exist_ok=True)
DEV  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:",DEV, "| cuda available:", torch.cuda.is_available())
if DEV.type=="cpu": print("  *** WARNING: running on CPU -> slow. Check torch CUDA build / GPU runtime. ***")
K=3; SEED=3; EP,BS,LR,WD=60,256,3e-4,1e-4; ARC_S,ARC_M,TAU=32.0,0.50,0.07; LAM_ST,LAM_OR,LAM_SU=0.50,0.10,0.30
GEN_C, IMP_C = "#009E73", "#D55E00"

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
    def zid(s,x):
        f=s.enc(x).squeeze(-1); return F.normalize(s.id_n(s.id_fc(f)),dim=-1)
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

@torch.no_grad()
def zid_batch(m,Xa,bs=8192):
    Z=[]
    for i in range(0,len(Xa),bs): Z.append(m.zid(torch.tensor(Xa[i:i+bs],dtype=torch.float32,device=DEV)))
    return torch.cat(Z,0)

d=np.load(DATA,allow_pickle=True); X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
sess=np.array([int(str(s).lstrip("Rr")) for s in d["session"]],dtype=np.int64); state=(sess>=3).astype(np.int64)
m=DOMCS().to(DEV)
if os.path.exists(CKPT):
    ck=torch.load(CKPT,map_location="cpu"); m.load_state_dict(ck.get("model_state",ck),strict=False); print("loaded",CKPT)
else:
    print("no checkpoint -> training seed 3 inline..."); torch.manual_seed(SEED); rng=np.random.default_rng(SEED)
    idx=rng.permutation(len(X)); ti=idx[int(0.1*len(idx)):]; Xt,yt,st=X[ti],y[ti],state[ti]
    arc=ArcFace().to(DEV); sh=nn.Linear(128,2).to(DEV); P=list(m.parameters())+list(arc.parameters())+list(sh.parameters())
    opt=torch.optim.Adam(P,lr=LR,weight_decay=WD); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP,eta_min=1e-6); t0=time.time()
    for ep in range(EP):
        m.train(); perm=rng.permutation(len(Xt))
        for i in range(0,len(perm),BS):
            b=perm[i:i+BS]; xb=torch.tensor(Xt[b],device=DEV); yb=torch.tensor(yt[b],device=DEV); sb=torch.tensor(st[b],device=DEV)
            r=(st[b]==0); opt.zero_grad(); zi,zs=m(xb); L=LAM_ST*F.cross_entropy(sh(zs),sb)+LAM_OR*orth(zi,zs)
            if r.sum()>1: L=L+F.cross_entropy(arc(zi[r],yb[r]),yb[r])+LAM_SU*supcon(zi[r],yb[r])
            L.backward(); torch.nn.utils.clip_grad_norm_(P,1.0); opt.step()
        sch.step()
    print(f"trained in {(time.time()-t0)/60:.1f} min")
m.eval()

# ---- build per-subject prototypes (REST, KMeans K=3) ----
rest=np.isin(sess,[1,2]); task=sess>=3
Zr=zid_batch(m,X[rest]).cpu().numpy(); yr=y[rest]; subs=np.unique(y); Pl=[]
for u in subs:
    zu=Zr[yr==u]
    c=KMeans(K,n_init=5,random_state=0).fit(zu).cluster_centers_ if len(zu)>=K else np.repeat(zu.mean(0,keepdims=True),K,0)
    Pl.append(c/(np.linalg.norm(c,axis=1,keepdims=True)+1e-9))
P=torch.tensor(np.stack(Pl),dtype=torch.float32,device=DEV)      # (S,K,128)
S=P.shape[0]; Pflat=P.reshape(S*K,128)                           # (S*K,128)
sub2i={u:i for i,u in enumerate(subs)}; yt_idx=np.array([sub2i[v] for v in y[task]])

# ---- VECTORIZED exhaustive scoring on GPU (each probe vs all S subjects, max over K) ----
t0=time.time(); gen=[]; imp=[]; Xt=X[task]
with torch.no_grad():
    for i in range(0,len(Xt),4096):
        zt=m.zid(torch.tensor(Xt[i:i+4096],dtype=torch.float32,device=DEV))    # (b,128)
        sims=(zt@Pflat.T).reshape(-1,S,K).amax(2)                              # (b,S) max over prototypes
        idx=torch.tensor(yt_idx[i:i+4096],device=DEV); ar=torch.arange(len(idx),device=DEV)
        g=sims[ar,idx]                                                         # genuine
        msk=torch.ones_like(sims,dtype=torch.bool); msk[ar,idx]=False
        gen.append(g.cpu().numpy()); imp.append(sims[msk].cpu().numpy())       # all impostors
gen=np.concatenate(gen); imp=np.concatenate(imp)
print(f"scored {len(gen)} genuine + {len(imp):,} impostor in {time.time()-t0:.1f}s")
sc=np.concatenate([gen,imp]); lb=np.concatenate([np.ones_like(gen),np.zeros_like(imp)])
fpr,tpr,thr=roc_curve(lb,sc); fnr=1-tpr; kk=np.nanargmin(np.abs(fpr-fnr)); eer=fpr[kk]*100; tau=thr[kk]
print(f"EER={eer:.2f}%  threshold={tau:.3f}")

fig,ax=plt.subplots(figsize=(3.5,2.6)); bins=np.linspace(-0.2,1.0,80)
ax.hist(imp,bins=bins,density=True,color=IMP_C,alpha=0.55,label=f"Impostor (n={len(imp):,})")
ax.hist(gen,bins=bins,density=True,color=GEN_C,alpha=0.65,label=f"Genuine (n={len(gen):,})")
ax.axvline(tau,color="k",ls="--",lw=1.0); ax.text(tau,ax.get_ylim()[1]*0.9,f"  EER={eer:.2f}%",fontsize=7,va="top")
ax.set_xlabel("Cosine similarity score"); ax.set_ylabel("Density"); ax.set_title("Genuine vs. impostor scores (EEGMMIDB, B2T)")
ax.legend(loc="upper left",frameon=False); fig.tight_layout()
fig.savefig(f"{OUT}/FIG_score_distribution.pdf"); fig.savefig(f"{OUT}/FIG_score_distribution.png",dpi=300)
print(f"Saved -> {OUT}/FIG_score_distribution.pdf (+png)")
