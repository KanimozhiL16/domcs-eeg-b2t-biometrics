#!/usr/bin/env python3
# =====================================================================================
#  TIFS FIGURES — PART 2 (embedding-based; loads E1 ELU model, seed_3 representative)
#  Figures: t-SNE (z_id by subject / by state), CMC rank-1..10, per-subject EER boxplot,
#           gallery-K sensitivity, temporal drift R03->R14 (Pearson r).
#  Same TIFS style as part 1: 7.16/3.5in, pdf.fonttype=42, >=8pt, vector PDF + 300dpi PNG.
#  All recomputed from 08_ablation_ELU_FINAL/E1 (one ELU family). No stale numbers.
# =====================================================================================
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans; from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve; from scipy import stats
mpl.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"sans-serif","font.size":8,
  "axes.titlesize":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
  "axes.linewidth":0.6,"figure.dpi":300,"savefig.bbox":"tight"})
COL,HALF=7.16,3.5; CB=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9"]
ROOT="/home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG"
CKPT=f"{ROOT}/08_ablation_ELU_FINAL/E1/seed_3/model_best.pt"
DATA="/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz"
OUT=f"{ROOT}/figures_TIFS"; os.makedirs(OUT,exist_ok=True)
DEV=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); K=3; RNG=np.random.default_rng(0)
def save(fig,n): fig.savefig(f"{OUT}/{n}.pdf"); fig.savefig(f"{OUT}/{n}.png",dpi=300); plt.close(fig); print(f"  saved {n}")

class CB_(nn.Module):
    def __init__(s,ic,oc,k):
        super().__init__(); s.net=nn.Sequential(nn.Conv1d(ic,oc,k,padding=k//2,bias=False),nn.BatchNorm1d(oc),nn.ELU(inplace=True))
    def forward(s,x): return s.net(x)
class DOMCS(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=nn.Sequential(CB_(64,64,7),CB_(64,128,5),CB_(128,256,3),nn.AdaptiveAvgPool1d(1))
        s.id_fc=nn.Linear(256,128,bias=False); s.id_n=nn.LayerNorm(128)
        s.st_fc=nn.Linear(256,128,bias=False); s.st_n=nn.LayerNorm(128)
    def forward(s,x): f=s.enc(x).squeeze(-1); return F.normalize(s.id_n(s.id_fc(f)),dim=-1)
ck=torch.load(CKPT,map_location=DEV,weights_only=False); assert ck["activation"]=="ELU"
M=DOMCS(); M.load_state_dict(ck["model_state"],strict=True); M.eval().to(DEV)
d=np.load(DATA,allow_pickle=True); X=d["X"].astype(np.float32); y=d["y"].astype(np.int64)
sess=np.array([int(str(s).lstrip("Rr")) for s in d["session"]]); rest=np.isin(sess,[1,2]); task=~rest
SUBJ=sorted(np.unique(y)); s2i={s:i for i,s in enumerate(SUBJ)}; NS=len(SUBJ)
@torch.no_grad()
def emb(Xn,bs=4096):
    o=[]
    for i in range(0,len(Xn),bs): o.append(M(torch.from_numpy(Xn[i:i+bs]).to(DEV)).cpu().numpy())
    return np.concatenate(o)
print("Embedding..."); zr=emb(X[rest]); yr=y[rest]; zt=emb(X[task]); yt=y[task]
def gallery(kk=K):
    G=np.zeros((NS,kk,128),np.float32)
    for s in SUBJ:
        c=KMeans(kk,n_init=10,random_state=0).fit(zr[yr==s]).cluster_centers_
        G[s2i[s]]=c/(np.linalg.norm(c,axis=1,keepdims=True)+1e-12)
    return G
def eer(gen,imp):
    sc=np.r_[gen,imp].astype(np.float64); lb=np.r_[np.ones_like(gen),np.zeros_like(imp)]
    fpr,tpr,_=roc_curve(lb,sc); fnr=1-tpr; i=np.nanargmin(np.abs(fnr-fpr)); return (fpr[i]+fnr[i])/2*100

# ---------- t-SNE (z_id by subject + by state), double column ----------
def fig_tsne(nsub=12,per=120):
    pick=SUBJ[:nsub]; idx=[]
    for s in pick:
        ri=np.where((y==s)&rest)[0]; ti=np.where((y==s)&task)[0]
        idx+=list(RNG.choice(ri,min(per//2,len(ri)),replace=False))+list(RNG.choice(ti,min(per//2,len(ti)),replace=False))
    idx=np.array(idx); Z=np.concatenate([emb(X[idx])]) ; emb2=TSNE(2,init="pca",perplexity=30,random_state=0).fit_transform(Z)
    sub=y[idx]; isr=rest[idx]
    fig,ax=plt.subplots(1,2,figsize=(COL,3))
    for i,s in enumerate(pick):
        m=sub==s; ax[0].scatter(emb2[m,0],emb2[m,1],s=6,color=plt.cm.tab20(i%20),lw=0)
    ax[0].set_title("(a) $z_{id}$ by subject"); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].scatter(emb2[isr,0],emb2[isr,1],s=6,color=CB[0],label="REST",lw=0)
    ax[1].scatter(emb2[~isr,0],emb2[~isr,1],s=6,color=CB[1],label="TASK",marker="^",lw=0)
    ax[1].set_title("(b) $z_{id}$ by state (co-located)"); ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].legend()
    fig.tight_layout(); save(fig,"FIG_tsne")

# ---------- CMC rank-1..10 (single column) ----------
def fig_cmc():
    G=gallery(); C=torch.tensor(G.reshape(NS*K,128),device=DEV); P=torch.tensor(zt,device=DEV); ranks=[]
    with torch.no_grad():
        for i in range(0,len(P),8192):
            S=(P[i:i+8192]@C.T).view(-1,NS,K).max(2).values.cpu().numpy()
            order=np.argsort(-S,1); yb=yt[i:i+8192]
            for j,s in enumerate(yb): ranks.append(np.where(order[j]==s2i[s])[0][0])
    ranks=np.array(ranks); cmc=[100*np.mean(ranks<k) for k in range(1,11)]
    fig,ax=plt.subplots(figsize=(HALF,2.5)); ax.plot(range(1,11),cmc,marker="o",color=CB[0],lw=1.4,ms=4)
    ax.set_xlabel("Rank"); ax.set_ylabel("Identification (%)"); ax.set_title(f"CMC (Rank-1={cmc[0]:.1f}%)")
    ax.grid(alpha=.3,lw=.4); ax.set_ylim(min(cmc)-2,100.5); fig.tight_layout(); save(fig,"FIG_cmc")

# ---------- per-subject EER boxplot + K sensitivity (double column) ----------
def fig_persubj_k():
    G=gallery(); Gt=torch.tensor(G.reshape(NS*K,128),device=DEV); P=torch.tensor(zt,device=DEV)
    with torch.no_grad():
        Sall=np.concatenate([(P[i:i+8192]@Gt.T).view(-1,NS,K).max(2).values.cpu().numpy() for i in range(0,len(P),8192)])
    per=[]
    for s in SUBJ:
        m=yt==s; gi=s2i[s]; gen=Sall[m,gi]; imp=Sall[m][:, [j for j in range(NS) if j!=gi]].ravel()
        per.append(eer(gen,imp))
    ks=[1,2,3]; ek=[]
    for kk in ks:
        Gk=gallery(kk); Ck=torch.tensor(Gk.reshape(NS*kk,128),device=DEV)
        with torch.no_grad():
            Sk=np.concatenate([(P[i:i+8192]@Ck.T).view(-1,NS,kk).max(2).values.cpu().numpy() for i in range(0,len(P),8192)])
        r=np.arange(len(yt)); gidx=np.array([s2i[s] for s in yt]); g=Sk[r,gidx]
        Mk=Sk.copy(); Mk[r,gidx]=np.nan; ek.append(eer(g,Mk[~np.isnan(Mk)]))
    fig,ax=plt.subplots(1,2,figsize=(COL,2.5))
    ax[0].boxplot(per,vert=True,widths=.5); ax[0].set_ylabel("Per-subject EER (%)")
    ax[0].set_title(f"(a) Per-subject (med={np.median(per):.2f}%)"); ax[0].set_xticks([])
    ax[1].bar(range(len(ks)),ek,color=CB[0],width=.5); ax[1].set_xticks(range(len(ks)))
    ax[1].set_xticklabels([f"K={k}" for k in ks]); ax[1].set_ylabel("EER (%)"); ax[1].set_title("(b) Gallery size")
    fig.tight_layout(); save(fig,"FIG_persubject_K")

# ---------- temporal drift R03..R14 (single column) ----------
def fig_drift():
    G=gallery(); Gt=torch.tensor(G.reshape(NS*K,128),device=DEV); runs=list(range(3,15)); ev=[]
    for rn in runs:
        m=sess[task]==rn; zz=zt[m]; yy=yt[m]; P=torch.tensor(zz,device=DEV)
        with torch.no_grad():
            S=np.concatenate([(P[i:i+8192]@Gt.T).view(-1,NS,K).max(2).values.cpu().numpy() for i in range(0,len(P),8192)])
        r=np.arange(len(yy)); gi=np.array([s2i[s] for s in yy]); g=S[r,gi]; Mm=S.copy(); Mm[r,gi]=np.nan
        ev.append(eer(g,Mm[~np.isnan(Mm)]))
    rr,pp=stats.pearsonr(runs,ev)
    fig,ax=plt.subplots(figsize=(HALF,2.5)); ax.plot(runs,ev,marker="o",color=CB[0],lw=1.2,ms=4)
    z=np.polyfit(runs,ev,1); ax.plot(runs,np.polyval(z,runs),ls="--",color=CB[1],lw=1)
    ax.set_xlabel("Task run (R03-R14)"); ax.set_ylabel("EER (%)"); ax.grid(alpha=.3,lw=.4)
    ax.set_title(f"Temporal drift (r={rr:.2f}, p={pp:.1e})"); fig.tight_layout(); save(fig,"FIG_temporal_drift")

print("Generating TIFS figures (part 2, from model)...")
fig_tsne(); fig_cmc(); fig_persubj_k(); fig_drift()
print(f"Done -> {OUT}")
