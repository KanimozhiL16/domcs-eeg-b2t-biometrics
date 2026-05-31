"""
Cell 14 — Four Additional Experiments for IEEE TIFS Reviewer-Proofing
======================================================================
EXP A: λ_orth sensitivity (λ ∈ {0.01, 0.05, 0.10, 0.20, 0.50})  ~50 min
EXP B: Intra-state EER (same-state baseline to quantify cross-state gap) ~3 min
EXP C: Enrollment size sensitivity (30/60/90/120 REST windows) ~5 min
EXP D: Inference timing (GPU + CPU µs/window) ~1 min

Run: python Cell_14_FourExperiments.py
All outputs → 14_reviewer_experiments/
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, copy
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.cluster import KMeans

# ─── PATHS ────────────────────────────────────────────────────────
BASE      = Path("/home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG")
DATA_PATH = Path("/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz")
CKPT_SEED3 = BASE / "01_checkpoints/seed_3/model_best.pt"
OUT_DIR   = BASE / "14_reviewer_experiments"; OUT_DIR.mkdir(exist_ok=True)
N_GPU     = torch.cuda.device_count()
DEVICE    = torch.device("cuda:0")
print(f"GPUs={N_GPU}  Device={DEVICE}  Output={OUT_DIR}")

# ─── SHARED ARCHITECTURE (ELU — unchanged throughout) ─────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, pad):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.ELU())
    def forward(self, x): return self.net(x)

class EEGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=ConvBlock(64,64,7,3); self.conv2=ConvBlock(64,128,5,2)
        self.conv3=ConvBlock(128,256,3,1); self.pool=nn.AdaptiveAvgPool1d(1)
    def forward(self,x): return self.pool(self.conv3(self.conv2(self.conv1(x)))).squeeze(-1)

class IdentityBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(256,128,bias=False); self.norm=nn.LayerNorm(128)
    def forward(self,f): return F.normalize(self.norm(self.fc(f)),dim=1)

class StateBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(256,128,bias=False); self.norm=nn.LayerNorm(128)
    def forward(self,f): return F.normalize(self.norm(self.fc(f.detach())),dim=1)

class DOMCSModel(nn.Module):
    """Full model for training (includes state branch + heads)"""
    def __init__(self, n_subj=109):
        super().__init__()
        self.encoder=EEGEncoder(); self.id_branch=IdentityBranch()
        self.state_branch=StateBranch()
        self.arc_w=nn.Parameter(F.normalize(torch.randn(n_subj,128),dim=1))
        self.state_cls=nn.Linear(128,2)
    def forward(self,x):
        f=self.encoder(x)
        z_id=self.id_branch(f); z_st=self.state_branch(f)
        return z_id, z_st, f

class DOMCSInference(nn.Module):
    """Inference only (no state branch)"""
    def __init__(self):
        super().__init__()
        self.encoder=EEGEncoder(); self.id_branch=IdentityBranch()
    def forward(self,x): return self.id_branch(self.encoder(x))

# ─── SHARED HELPERS ───────────────────────────────────────────────
@torch.no_grad()
def embed_all(model, X, batch=2048):
    dl=DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float32)),
                  batch_size=batch,shuffle=False,num_workers=4,pin_memory=True)
    out=[model(xb.to(DEVICE)).cpu() for (xb,) in dl]
    return F.normalize(torch.cat(out,0),dim=1).numpy()

def build_gallery(z, y, subj2idx, K=3):
    gallery={}
    for sid,s_idx in subj2idx.items():
        idx=np.where(y==sid)[0]
        km=KMeans(n_clusters=min(K,len(idx)),random_state=42,n_init=10)
        km.fit(z[idx]); p=km.cluster_centers_
        gallery[s_idx]=(p/(np.linalg.norm(p,axis=1,keepdims=True)+1e-12)).astype(np.float32)
    return gallery

def score_1v1(z, y, gallery, subj2idx, rng_seed=3):
    rng=np.random.default_rng(rng_seed); N_sub=len(subj2idx)
    s_idxs=np.array([subj2idx[yi] for yi in y])
    gen_sc=np.array([float(np.max(z[i]@gallery[s_idxs[i]].T)) for i in range(len(z))])
    t_idxs=np.array([rng.choice([j for j in range(N_sub) if j!=s_idxs[i]])
                     for i in range(len(z))])
    imp_sc=np.array([float(np.max(z[i]@gallery[t_idxs[i]].T)) for i in range(len(z))])
    sc=np.concatenate([gen_sc,imp_sc]); lb=np.concatenate([np.ones(len(z)),np.zeros(len(z))])
    fpr,tpr,_=roc_curve(lb,sc,drop_intermediate=True)
    fnr=1-tpr; idx=np.argmin(np.abs(fpr-fnr))
    return float((fpr[idx]+fnr[idx])/2)*100, sk_auc(fpr,tpr)

def load_inference_model(ckpt_path):
    m=DOMCSInference().to(DEVICE)
    ck=torch.load(ckpt_path,map_location=DEVICE,weights_only=False)
    state=ck.get('model_state',ck.get('model_state_dict',ck))
    res=m.load_state_dict(state,strict=False)
    assert res.missing_keys==[], f"Missing: {res.missing_keys}"
    m.eval()
    if N_GPU>1: m=nn.DataParallel(m)
    return m

# ─── LOAD DATA ────────────────────────────────────────────────────
print("\nLoading data...")
data=np.load(DATA_PATH,allow_pickle=True)
X_all=data['X'].astype(np.float32); y_all=data['y'].astype(np.int64)
sessions=np.array([int(str(s).lstrip('R')) for s in data['session']])
rest_mask=np.isin(sessions,[1,2]); r1_mask=(sessions==1); r2_mask=(sessions==2)
X_rest=X_all[rest_mask]; y_rest=y_all[rest_mask]
X_r1=X_all[r1_mask]; y_r1=y_all[r1_mask]   # R01 only
X_r2=X_all[r2_mask]; y_r2=y_all[r2_mask]   # R02 only
X_task=X_all[~rest_mask]; y_task=y_all[~rest_mask]
subj_ids=sorted(np.unique(y_all)); subj2idx={s:i for i,s in enumerate(subj_ids)}
N_SUB=len(subj_ids)
print(f"  REST={len(X_rest):,}  R01={len(X_r1):,}  R02={len(X_r2):,}  TASK={len(X_task):,}  Subjects={N_SUB}")

# Embed once with Seed 3 model
print("Embedding with Seed 3 model (reused across EXP B,C,D)...")
inf_model=load_inference_model(CKPT_SEED3)
z_rest=embed_all(inf_model,X_rest); z_r1=embed_all(inf_model,X_r1)
z_r2=embed_all(inf_model,X_r2);    z_task=embed_all(inf_model,X_task)
print("  ✓ Embeddings ready")


# ═══════════════════════════════════════════════════════════════════
# EXP B — Intra-state EER (same-state baseline)
# Gallery=R01 (REST session 1), Probe=R02 (REST session 2)
# Quantifies the same-state EER to contrast with B2T cross-state
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  EXP B — Intra-State EER (Gallery=R01, Probe=R02)")
print("="*65)

# Gallery from R01 REST session only
gal_r1=build_gallery(z_r1,y_r1,subj2idx,K=3)
# Probe: R02 REST → same state as gallery
eer_intra, auc_intra = score_1v1(z_r2,y_r2,gal_r1,subj2idx)
# Reference: B2T cross-state
gal_rest=build_gallery(z_rest,y_rest,subj2idx,K=3)
eer_b2t, auc_b2t = score_1v1(z_task,y_task,gal_rest,subj2idx)

cross_state_penalty = eer_b2t - eer_intra
print(f"\n  Same-state (Gallery=R01, Probe=R02): EER={eer_intra:.4f}%  AUC={auc_intra:.4f}")
print(f"  Cross-state B2T (Gallery=R01+R02, Probe=TASK): EER={eer_b2t:.4f}%  AUC={auc_b2t:.4f}")
print(f"  Cross-state penalty: Δ={cross_state_penalty:+.4f} pp")
print(f"\n  Paper text:")
print(f"  'Same-state verification (gallery=R01, probe=R02) achieves EER={eer_intra:.2f}%,")
print(f"   while B2T cross-state (gallery=REST, probe=TASK) achieves EER={eer_b2t:.2f}%")
print(f"   (+{cross_state_penalty:.2f} pp), confirming the cross-state challenge is non-trivial'")

expB={'same_state_eer':eer_intra,'same_state_auc':auc_intra,
      'b2t_cross_state_eer':eer_b2t,'b2t_cross_state_auc':auc_b2t,
      'cross_state_penalty_pp':cross_state_penalty}


# ═══════════════════════════════════════════════════════════════════
# EXP C — Enrollment Size Sensitivity
# Vary REST windows per subject: 30, 60, 90, 120 (full)
# Gallery built from subsampled REST embeddings
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  EXP C — Enrollment Size Sensitivity")
print("="*65)

enroll_results={}
for n_enroll in [30, 60, 90, 120]:
    rng_e=np.random.default_rng(42)
    z_enroll=[]; y_enroll=[]
    for sid in subj_ids:
        idx=np.where(y_rest==sid)[0]
        sel=rng_e.choice(idx,min(n_enroll,len(idx)),replace=False)
        z_enroll.append(z_rest[sel]); y_enroll.append(y_rest[sel])
    z_enroll=np.concatenate(z_enroll); y_enroll=np.concatenate(y_enroll)
    gal_e=build_gallery(z_enroll,y_enroll,subj2idx,K=3)
    eer_e,auc_e=score_1v1(z_task,y_task,gal_e,subj2idx)
    enroll_results[n_enroll]={'eer':eer_e,'auc':auc_e}
    print(f"  n={n_enroll:3d} windows/subject  EER={eer_e:.4f}%  AUC={auc_e:.4f}")

# Plot
fig,ax=plt.subplots(figsize=(6,4))
ns=[30,60,90,120]; eers=[enroll_results[n]['eer'] for n in ns]
ax.plot(ns,eers,'o-',color='#1f77b4',lw=2,ms=9,markerfacecolor='white',markeredgewidth=2)
for n,e in zip(ns,eers): ax.annotate(f'{e:.2f}%',(n,e),textcoords="offset points",xytext=(0,10),ha='center',fontsize=9)
ax.set_xlabel('Enrollment windows per subject',fontsize=12)
ax.set_ylabel('EER (%)',fontsize=12)
ax.set_title('Enrollment Size Sensitivity (B2T, K=3, 1-vs-1)',fontsize=11)
ax.set_xticks(ns); ax.grid(True,alpha=0.3,ls='--'); ax.set_ylim(bottom=0)
plt.tight_layout(); fig.savefig(OUT_DIR/'FIG_enrollment_sensitivity.pdf',dpi=300,bbox_inches='tight')
fig.savefig(OUT_DIR/'FIG_enrollment_sensitivity.png',dpi=300,bbox_inches='tight'); plt.close()
print(f"  ✓ Figure saved")


# ═══════════════════════════════════════════════════════════════════
# EXP D — Inference Timing
# Measure embedding latency: GPU + CPU
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  EXP D — Inference Timing")
print("="*65)

timing_model = DOMCSInference().to(DEVICE)
ck=torch.load(CKPT_SEED3,map_location=DEVICE,weights_only=False)
state=ck.get('model_state',ck.get('model_state_dict',ck))
timing_model.load_state_dict(state,strict=False); timing_model.eval()

# GPU timing — batch=1 (real-time), batch=256 (throughput)
timing_results={}
for batch_sz in [1, 32, 256]:
    x_dummy=torch.randn(batch_sz,64,256,device=DEVICE)
    # Warmup
    for _ in range(20): _ = timing_model(x_dummy)
    torch.cuda.synchronize()
    # Measure 200 iterations
    t0=time.perf_counter()
    N_iter=200
    for _ in range(N_iter): timing_model(x_dummy)
    torch.cuda.synchronize()
    elapsed_ms=(time.perf_counter()-t0)*1000
    ms_per_batch=elapsed_ms/N_iter
    us_per_window=ms_per_batch/batch_sz*1000
    throughput=batch_sz/(ms_per_batch/1000)
    timing_results[f'gpu_batch{batch_sz}']={
        'ms_per_batch':ms_per_batch,'us_per_window':us_per_window,'throughput_wps':throughput}
    print(f"  GPU batch={batch_sz:3d}: {ms_per_batch:.2f} ms/batch | {us_per_window:.1f} µs/window | {throughput:.0f} windows/sec")

# CPU timing — batch=1
cpu_model=DOMCSInference(); cpu_model.load_state_dict(state,strict=False); cpu_model.eval()
x_cpu=torch.randn(1,64,256)
for _ in range(5): _ = cpu_model(x_cpu)
t0=time.perf_counter()
for _ in range(50): cpu_model(x_cpu)
cpu_ms=(time.perf_counter()-t0)*1000/50
timing_results['cpu_batch1']={'ms_per_window':cpu_ms,'us_per_window':cpu_ms*1000}
print(f"  CPU batch=  1: {cpu_ms:.1f} ms/window")

# 2-second EEG window → real-time constraint
fs=128; window_samples=256; window_dur_ms=2000
gpu_b1_ms=timing_results['gpu_batch1']['ms_per_batch']
print(f"\n  Window duration = {window_dur_ms} ms (2s at 128Hz)")
print(f"  GPU latency (batch=1) = {gpu_b1_ms:.2f} ms → real-time factor = {window_dur_ms/gpu_b1_ms:.0f}×")
print(f"  System is {'✓ real-time capable' if gpu_b1_ms < window_dur_ms else '✗ NOT real-time'} on GPU")


# ═══════════════════════════════════════════════════════════════════
# EXP A — λ_orth Sensitivity (requires training)
# λ_orth ∈ {0.01, 0.05, 0.10 (existing), 0.20, 0.50}
# Uses 2 seeds only (sufficient for sensitivity trend)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  EXP A — λ_orth Sensitivity (2 seeds × 4 new λ values)")
print("="*65)

# λ=0.10 already exists — reuse from Cell 3 checkpoints
# Only train λ ∈ {0.01, 0.05, 0.20, 0.50} × seeds {1, 3}
LAMBDA_VALS = [0.01, 0.05, 0.20, 0.50]
SEEDS_LAM   = [1, 3]   # 2 seeds sufficient for sensitivity trend
EPOCHS      = 60
LR          = 3e-4
BATCH_SZ    = 256
LAM_STATE   = 0.50   # fixed
LAM_SUP     = 0.30   # fixed (SupCon)

def arcface_loss(z, y, arc_w, s=32.0, m=0.5):
    """ArcFace loss on z_id using REST windows only."""
    w=F.normalize(arc_w,dim=1)
    cos=z@w.T; cos=cos.clamp(-1+1e-7,1-1e-7)
    theta=cos.acos()
    theta_m=theta+m
    logits=s*(torch.where(torch.arange(w.shape[0],device=z.device).unsqueeze(0)==y.unsqueeze(1),
                          theta_m.cos(), cos))
    return F.cross_entropy(logits,y)

def supcon_loss(z, y, tau=0.07):
    """Supervised contrastive loss."""
    z=F.normalize(z,dim=1); sim=z@z.T/tau
    mask=(y.unsqueeze(0)==y.unsqueeze(1)).float()
    mask.fill_diagonal_(0)
    pos=mask.sum(1); valid=(pos>0)
    if not valid.any(): return torch.tensor(0.0,device=z.device)
    sim=sim-sim.max(dim=1,keepdim=True).values.detach()
    exp_sim=sim.exp()
    log_prob=sim-torch.log(exp_sim.sum(dim=1,keepdim=True)+1e-8)
    loss=-(mask*log_prob).sum(1)[valid]/(pos[valid]+1e-8)
    return loss.mean()

def train_lambda(lam_orth, seed, out_ckpt_path):
    """Train one model variant with given λ_orth, return val_loss."""
    torch.manual_seed(seed); np.random.seed(seed)
    model=DOMCSModel(N_SUB).to(DEVICE)
    if N_GPU>1: model=nn.DataParallel(model)
    opt=Adam(model.parameters(),lr=LR,weight_decay=1e-4)
    sched=CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-6)

    # Build dataset
    X_t=torch.tensor(X_all,dtype=torch.float32)
    y_t=torch.tensor(y_all,dtype=torch.long)
    state_t=torch.zeros(len(X_all),dtype=torch.long)
    state_t[~rest_mask]=1   # REST=0, TASK=1
    rng_val=np.random.default_rng(seed*100)
    val_idx=rng_val.choice(len(X_all),int(0.1*len(X_all)),replace=False)
    train_mask=np.ones(len(X_all),bool); train_mask[val_idx]=False
    tr_ds=TensorDataset(X_t[train_mask],y_t[train_mask],state_t[train_mask])
    va_ds=TensorDataset(X_t[val_idx],  y_t[val_idx],  state_t[val_idx])
    tr_dl=DataLoader(tr_ds,batch_size=BATCH_SZ,shuffle=True, num_workers=4,pin_memory=True,drop_last=True)
    va_dl=DataLoader(va_ds,batch_size=BATCH_SZ,shuffle=False,num_workers=4,pin_memory=True)

    best_val=1e9; best_state=None
    for ep in range(1,EPOCHS+1):
        model.train(); tot=0; nb=0
        for xb,yb,sb in tr_dl:
            xb,yb,sb=xb.to(DEVICE),yb.to(DEVICE),sb.to(DEVICE)
            z_id,z_st,_=model(xb)
            # Core module
            m_core=model.module if hasattr(model,'module') else model
            rest_mask_b=(sb==0)
            # ArcFace + SupCon on REST only
            l_arc=arcface_loss(z_id[rest_mask_b],yb[rest_mask_b],m_core.arc_w) if rest_mask_b.any() else torch.tensor(0.,device=DEVICE)
            l_sup=supcon_loss(z_id[rest_mask_b],yb[rest_mask_b]) if rest_mask_b.any() else torch.tensor(0.,device=DEVICE)
            # State on all
            l_st=F.cross_entropy(m_core.state_cls(z_st),sb)
            # Orth on all
            l_orth=torch.mean(torch.abs(F.cosine_similarity(z_id,z_st,dim=1)))
            loss=l_arc + LAM_SUP*l_sup + LAM_STATE*l_st + lam_orth*l_orth
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); tot+=loss.item(); nb+=1
        sched.step()
        # Val
        model.eval(); vtot=0; vn=0
        with torch.no_grad():
            for xb,yb,sb in va_dl:
                xb,yb,sb=xb.to(DEVICE),yb.to(DEVICE),sb.to(DEVICE)
                z_id,z_st,_=model(xb)
                m_core=model.module if hasattr(model,'module') else model
                rb=(sb==0)
                l_arc=arcface_loss(z_id[rb],yb[rb],m_core.arc_w) if rb.any() else torch.tensor(0.,device=DEVICE)
                l_sup=supcon_loss(z_id[rb],yb[rb]) if rb.any() else torch.tensor(0.,device=DEVICE)
                l_st=F.cross_entropy(m_core.state_cls(z_st),sb)
                l_orth=torch.mean(torch.abs(F.cosine_similarity(z_id,z_st,dim=1)))
                v=l_arc+LAM_SUP*l_sup+LAM_STATE*l_st+lam_orth*l_orth
                vtot+=v.item(); vn+=1
        val_loss=vtot/vn
        if val_loss<best_val:
            best_val=val_loss
            m=model.module if hasattr(model,'module') else model
            best_state=copy.deepcopy(m.state_dict())
        if ep%15==0 or ep==EPOCHS:
            print(f"    ep={ep:2d}  train={tot/nb:.4f}  val={val_loss:.4f}  best={best_val:.4f}")

    # Save
    out_ckpt_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model_state':best_state,'best_val':best_val},out_ckpt_path)
    return best_val

# Load λ=0.10 existing EER from Cell 3 (reuse)
lambda_results = {}
eer_existing_seeds=[]
for s in SEEDS_LAM:
    m=load_inference_model(BASE/f"01_checkpoints/seed_{s}/model_best.pt")
    z_r=embed_all(m,X_rest); gal=build_gallery(z_r,y_rest,subj2idx,K=3)
    z_tk=embed_all(m,X_task); eer_s,_=score_1v1(z_tk,y_task,gal,subj2idx)
    eer_existing_seeds.append(eer_s)
    print(f"  λ=0.10 Seed {s}: EER={eer_s:.4f}%")
lambda_results[0.10]={'eers':eer_existing_seeds,'mean_eer':float(np.mean(eer_existing_seeds))}
print(f"  λ=0.10 mean EER={np.mean(eer_existing_seeds):.4f}%")

# Train new λ values
t_start=time.time()
for lam in LAMBDA_VALS:
    print(f"\n  Training λ_orth={lam} ({len(SEEDS_LAM)} seeds)...")
    lam_dir=OUT_DIR/f"lambda_orth_{lam:.2f}"
    eers_lam=[]
    for s in SEEDS_LAM:
        ckpt=lam_dir/f"seed_{s}/model_best.pt"
        if ckpt.exists():
            print(f"    Seed {s}: checkpoint exists, skipping training")
        else:
            print(f"    Seed {s}: training...")
            train_lambda(lam, s, ckpt)
        # Eval
        m=DOMCSInference().to(DEVICE)
        ck=torch.load(ckpt,map_location=DEVICE,weights_only=False)
        state=ck.get('model_state',ck)
        m.load_state_dict(state,strict=False); m.eval()
        if N_GPU>1: m=nn.DataParallel(m)
        z_r=embed_all(m,X_rest); gal=build_gallery(z_r,y_rest,subj2idx,K=3)
        z_tk=embed_all(m,X_task); eer_s,_=score_1v1(z_tk,y_task,gal,subj2idx)
        eers_lam.append(eer_s)
        print(f"    Seed {s}: EER={eer_s:.4f}%")
    lambda_results[lam]={'eers':eers_lam,'mean_eer':float(np.mean(eers_lam))}
    print(f"  λ={lam}  mean EER={np.mean(eers_lam):.4f}%")

print(f"\nEXP A total time: {(time.time()-t_start)/60:.1f} min")

# λ summary
print(f"\n  {'λ_orth':>8}  {'Mean EER (%)':>14}  {'ΔEER vs 0.10':>14}")
base_eer=lambda_results[0.10]['mean_eer']
for lam in sorted(lambda_results.keys()):
    r=lambda_results[lam]; delta=r['mean_eer']-base_eer
    print(f"  {lam:>8.2f}  {r['mean_eer']:>13.4f}%  {delta:>+13.4f}pp")

# λ sensitivity figure
fig,ax=plt.subplots(figsize=(7,4))
lams_sorted=sorted(lambda_results.keys())
mean_eers=[lambda_results[l]['mean_eer'] for l in lams_sorted]
ax.plot(lams_sorted,mean_eers,'o-',color='#1f77b4',lw=2,ms=9,
        markerfacecolor='white',markeredgewidth=2)
for l,e in zip(lams_sorted,mean_eers):
    ax.annotate(f'{e:.3f}%',(l,e),textcoords="offset points",xytext=(0,10),ha='center',fontsize=9)
ax.axvline(0.10,color='#e74c3c',ls='--',lw=1.5,label='Selected λ=0.10')
ax.set_xscale('log'); ax.set_xlabel('λ_orth (log scale)',fontsize=12)
ax.set_ylabel('Mean EER (%)',fontsize=12)
ax.set_title('λ_orth Sensitivity (B2T EER, 2-seed mean)',fontsize=11)
ax.legend(fontsize=10); ax.grid(True,alpha=0.3,ls='--')
plt.tight_layout(); fig.savefig(OUT_DIR/'FIG_lambda_sensitivity.pdf',dpi=300,bbox_inches='tight')
fig.savefig(OUT_DIR/'FIG_lambda_sensitivity.png',dpi=300,bbox_inches='tight'); plt.close()
print("  ✓ Figure saved: FIG_lambda_sensitivity.pdf/png")


# ─── SAVE ALL RESULTS ─────────────────────────────────────────────
all_results={
    "expA_lambda_sensitivity": {str(k):v for k,v in lambda_results.items()},
    "expB_intrastate": expB,
    "expC_enrollment_sensitivity": {str(k):v for k,v in enroll_results.items()},
    "expD_timing": timing_results,
}
with open(OUT_DIR/"cell14_all_results.json","w") as f:
    json.dump(all_results,f,indent=2)
print(f"\n✓ All results saved: {OUT_DIR}/cell14_all_results.json")

# ─── FINAL SUMMARY ────────────────────────────────────────────────
print("\n" + "="*65)
print("  CELL 14 COMPLETE — Reviewer-proofing experiments")
print("="*65)
print(f"\n  EXP B: Same-state EER={eer_intra:.4f}% vs B2T={eer_b2t:.4f}% (Δ={cross_state_penalty:+.4f}pp)")
print(f"  EXP C: Enrollment sensitivity: {enroll_results[30]['eer']:.3f}%(n=30) → {enroll_results[120]['eer']:.3f}%(n=120)")
print(f"  EXP D: GPU batch=1: {timing_results['gpu_batch1']['ms_per_batch']:.2f}ms/window | CPU: {timing_results['cpu_batch1']['ms_per_window']:.1f}ms/window")
print(f"  EXP A: λ sensitivity table above")
print(f"\n  All figures → {OUT_DIR}/")
