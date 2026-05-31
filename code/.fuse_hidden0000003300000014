#!/usr/bin/env python3
# =====================================================================================
#  R7-VIZ — BED disentanglement visualization for DOMCS (exact arch, Cin=14, seed 3).
#  Trains DOMCS on BED (enroll r01+r02), then: (a) t-SNE z_id by subject, (b) t-SNE z_id
#  by session (shows session-invariance), (c) |cos(z_id,z_state)| panel. TIFS-compliant
#  (7.16in, pdf.fonttype=42, >=8pt, vector PDF + 300dpi PNG, Wong palette).
# =====================================================================================
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
mpl.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif","font.size":8,
  "axes.titlesize":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
  "axes.linewidth":0.6,"figure.dpi":300,"savefig.bbox":"tight"})
COL=7.16; CB=["#0072B2","#D55E00","#009E73"]
BED="/home/nvidia/24PHD1237/BED_DATASET/BED_win2s_step1s_fs128.npz"
OUT="/home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG/figures_TIFS"; os.makedirs(OUT,exist_ok=True)
DEV=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED=3; EP=60; BS=256; S_ARC,M_ARC,TAU=32.0,0.50,0.07; LS,LO,LSU=0.50,0.10,0.30
torch.manual_seed(SEED); np.random.seed(SEED)

class CB_(nn.Module):
    def __init__(s,ic,oc,k):
        super().__init__(); s.net=nn.Sequential(nn.Conv1d(ic,oc,k,padding=k//2,bias=False),nn.BatchNorm1d(oc),nn.ELU(inplace=True))
    def forward(s,x): return s.net(x)
class DOMCS(nn.Module):
    def __init__(s,cin=14):
        super().__init__(); s.enc=nn.Sequential(CB_(cin,64,7),CB_(64,128,5),CB_(128,256,3),nn.AdaptiveAvgPool1d(1))
        s.id_fc=nn.Linear(256,128,bias=False); s.id_n=nn.LayerNorm(128)
        s.st_fc=nn.Linear(256,128,bias=False); s.st_n=nn.LayerNorm(128)
    def forward(s,x):
        f=s.enc(x).squeeze(-1); return (F.normalize(s.id_n(s.id_fc(f)),-1),F.normalize(s.st_n(s.st_fc(f.detach())),-1))
class Arc(nn.Module):
    def __init__(s,n):
        super().__init__(); s.W=nn.Parameter(torch.FloatTensor(n,128)); nn.init.xavier_uniform_(s.W)
    def forward(s,z,y):
        W=F.normalize(s.W,1); c=F.linear(z,W).clamp(-1+1e-7,1-1e-7)
        oh=torch.zeros_like(c).scatter_(1,y.view(-1,1),1.); return (c*(1-oh)+torch.cos(torch.acos(c)+M_ARC)*oh)*S_ARC
def supcon(z,y,tau=TAU):
    n=z.size(0); z=F.normalize(z,1); sim=torch.mm(z,z.T)/tau; sim=sim-sim.detach().max(1,True)[0]
    eye=torch.eye(n,device=z.device); m=y.unsqueeze(0).eq(y.unsqueeze(1)).float()*(1-eye); p=m.sum(1).clamp(min=1)
    e=torch.exp(sim)*(1-eye); lp=sim-torch.log(e.sum(1,True)+1e-9); return (-(m*lp).sum(1)/p).mean()
def orth(zi,zs): return torch.abs(F.cosine_similarity(zi,zs,-1)).mean()

d=np.load(BED,allow_pickle=True); X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
runs=np.asarray([str(r) for r in (d["session"] if "session" in d.files else d["runs"])],dtype=object)
SUBJ=sorted(np.unique(y)); s2i={s:i for i,s in enumerate(SUBJ)}; NS=len(SUBJ)
trm=np.isin(runs,["r01","r02"]); Xtr,ytr=X[trm],y[trm]
print(f"BED train {trm.sum()} windows, {NS} subjects — training DOMCS seed 3...")
m=DOMCS(14).to(DEV); arc=Arc(NS).to(DEV); sth=nn.Linear(128,2).to(DEV)
opt=torch.optim.Adam(list(m.parameters())+list(arc.parameters())+list(sth.parameters()),lr=3e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EP,eta_min=1e-6); rng=np.random.default_rng(SEED)
for ep in range(1,EP+1):
    m.train(); perm=rng.permutation(len(Xtr))
    for i in range(0,len(perm),BS):
        b=perm[i:i+BS]; xb=torch.FloatTensor(Xtr[b]).to(DEV); yb=torch.LongTensor(ytr[b]).to(DEV)
        opt.zero_grad(); zi,zs=m(xb)
        L=F.cross_entropy(arc(zi,yb),yb)+LSU*supcon(zi,yb)+LS*F.cross_entropy(sth(zs),torch.zeros(len(yb),dtype=torch.long,device=DEV))+LO*orth(zi,zs)
        L.backward(); torch.nn.utils.clip_grad_norm_(list(m.parameters()),1.0); opt.step()
    sch.step()
m.eval()
# sample for viz: all sessions r01/r02/r03, ~10 subjects
pick=SUBJ[:10]; idx=[]
for s in pick:
    for rr in ["r01","r02","r03"]:
        si=np.where((y==s)&(runs==rr))[0]
        idx+=list(rng.choice(si,min(40,len(si)),replace=False))
idx=np.array(idx)
with torch.no_grad():
    zi,zs=m(torch.from_numpy(X[idx]).to(DEV)); zi=zi.cpu().numpy(); zs=zs.cpu().numpy()
    cosv=np.abs(np.sum(zi*zs,1)).mean()
emb=TSNE(2,init="pca",perplexity=30,random_state=0).fit_transform(zi)
sub=y[idx]; ses=runs[idx]
fig,ax=plt.subplots(1,3,figsize=(COL,2.6))
for i,s in enumerate(pick):
    mks=sub==s; ax[0].scatter(emb[mks,0],emb[mks,1],s=6,color=plt.cm.tab20(i%20),lw=0)
ax[0].set_title("(a) BED $z_{id}$ by subject"); ax[0].set_xticks([]); ax[0].set_yticks([])
cols={"r01":CB[0],"r02":CB[1],"r03":CB[2]}; mk={"r01":"o","r02":"^","r03":"s"}
for rr in ["r01","r02","r03"]:
    mks=ses==rr; ax[1].scatter(emb[mks,0],emb[mks,1],s=6,color=cols[rr],marker=mk[rr],label=rr,lw=0)
ax[1].set_title("(b) BED $z_{id}$ by session"); ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].legend()
ax[2].bar([0],[cosv],color=CB[0],width=.5); ax[2].set_ylim(0,max(0.05,cosv*1.5))
ax[2].set_xticks([0]); ax[2].set_xticklabels(["$|\\cos(z_{id},z_{s})|$"]); ax[2].set_ylabel("value")
ax[2].set_title(f"(c) Disentanglement\n|cos|={cosv:.4f}")
fig.tight_layout(); fig.savefig(f"{OUT}/FIG_BED_disentangle.pdf"); fig.savefig(f"{OUT}/FIG_BED_disentangle.png",dpi=300)
print(f"|cos(z_id,z_state)| on BED = {cosv:.4f}")
print(f"Saved -> {OUT}/FIG_BED_disentangle.pdf/.png")
