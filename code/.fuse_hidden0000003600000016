"""
Cell 10 — Per-Run Temporal Drift (Fig. 4) + Security ROC (Fig. 7)
==================================================================
WHAT THIS GENERATES:
  Fig. 4: EER per session run R03–R14 with linear regression + Pearson r
  Fig. 7: ROC under random forgery vs identity-optimized (skilled) forgery

PATHS (no changes needed):
  Checkpoint : FRESH_EXP/01_checkpoints/seed_3/model_best.pt
  Data       : EEGMMIDB_win2s_step1s_fs128.npz
  Output     : FRESH_EXP/11_temporal_drift/

RUNTIME: ~8 min on A100
"""

import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────────
DATA_PATH = Path("/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz")
CKPT_PATH = Path("/home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG/01_checkpoints/seed_3/model_best.pt")
OUT_DIR   = Path("/home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG/11_temporal_drift")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 512
K_GALLERY   = 3
REST_SESS   = [1, 2]           # R01, R02 → gallery enrollment
TASK_SESS   = list(range(3, 15))  # R03–R14 → probe runs

print(f"Device: {DEVICE}")
print(f"Output: {OUT_DIR}")

# ─── MODEL (inference-only, ELU, same as Cell 9A) ─────────────
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
        self.conv1 = ConvBlock(64, 64,  7, 3)
        self.conv2 = ConvBlock(64, 128, 5, 2)
        self.conv3 = ConvBlock(128,256, 3, 1)
        self.pool  = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        return self.pool(self.conv3(self.conv2(self.conv1(x)))).squeeze(-1)

class IdentityBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc   = nn.Linear(256, 128, bias=False)
        self.norm = nn.LayerNorm(128)
    def forward(self, f):
        return F.normalize(self.norm(self.fc(f)), dim=1)

class DOMCSInferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = EEGEncoder()
        self.id_branch = IdentityBranch()
    def forward(self, x):
        return self.id_branch(self.encoder(x))

def load_model():
    m = DOMCSInferenceModel().to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
    res   = m.load_state_dict(state, strict=False)
    assert res.missing_keys == [], f"Missing keys: {res.missing_keys}"
    m.eval()
    nparams = sum(p.numel() for p in m.parameters())
    print(f"  ✓ Model loaded | params={nparams:,} | ignored={len(res.unexpected_keys)} state_branch keys")
    return m

# ─── DATA ─────────────────────────────────────────────────────
def load_data():
    data     = np.load(DATA_PATH, allow_pickle=True)
    X        = data['X'].astype(np.float32)
    y        = data['y'].astype(np.int64)
    sess_raw = data['session']
    sessions = np.array([int(str(s).lstrip('R')) for s in sess_raw])
    print(f"  Data: X={X.shape}, subjects={len(np.unique(y))}, "
          f"sessions={sorted(set(sessions.tolist()))}")
    return X, y, sessions

# ─── EMBEDDINGS ───────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, X_np, y_np):
    ds = TensorDataset(torch.tensor(X_np), torch.tensor(y_np))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    E, Y = [], []
    for xb, yb in dl:
        E.append(model(xb.to(DEVICE)).cpu().numpy())
        Y.append(yb.numpy())
    E = np.concatenate(E); Y = np.concatenate(Y)
    E = (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    return E, Y.astype(np.int64)

# ─── GALLERY ──────────────────────────────────────────────────
def build_gallery(E, Y, K=3):
    pvecs, powner = [], []
    for sid in sorted(np.unique(Y)):
        idx   = np.where(Y == sid)[0]
        k_use = min(K, len(idx))
        km    = KMeans(n_clusters=k_use, random_state=42, n_init=10)
        km.fit(E[idx])
        for c in km.cluster_centers_:
            c = c / (np.linalg.norm(c) + 1e-12)
            pvecs.append(c.astype(np.float32)); powner.append(int(sid))
    return np.stack(pvecs), np.array(powner, dtype=np.int64)

# ─── SCORING ──────────────────────────────────────────────────
def compute_scores(E_probe, Y_probe, P, P_owner):
    """Returns arrays of (scores, labels) for EER computation."""
    scores, labels = [], []
    subj_ids = sorted(np.unique(P_owner))
    for i in range(len(E_probe)):
        e = E_probe[i]; tid = int(Y_probe[i])
        sim = e @ P.T      # cosine similarity to all prototypes
        # Genuine: max score vs own prototypes
        own = (P_owner == tid)
        if own.sum() > 0:
            scores.append(float(sim[own].max())); labels.append(1)
        # Impostors: max score vs each other subject
        for sid in subj_ids:
            if sid == tid: continue
            imp = (P_owner == sid)
            scores.append(float(sim[imp].max())); labels.append(0)
    return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int8)

def eer_from_scores(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auc_val      = sk_auc(fpr, tpr)
    fnr          = 1.0 - tpr
    idx          = np.nanargmin(np.abs(fpr - fnr))
    eer          = float((fpr[idx] + fnr[idx]) / 2.0 * 100)
    return eer, auc_val, fpr, tpr

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("  CELL 10 — Temporal Drift + Security ROC")
print("=" * 65)

model = load_model()
X, y, sessions = load_data()

# ── Gallery (REST: R01 + R02) ──────────────────────────────────
rest_mask = np.isin(sessions, REST_SESS)
print(f"\nBuilding gallery from REST ({rest_mask.sum()} windows)...")
E_gal, Y_gal = extract_embeddings(model, X[rest_mask], y[rest_mask])
P, P_owner   = build_gallery(E_gal, Y_gal, K=K_GALLERY)
print(f"  Gallery: {len(P)} prototypes for {len(np.unique(P_owner))} subjects (K={K_GALLERY})")

# ── Fig. 4: EER per run R03–R14 ───────────────────────────────
print("\nComputing per-run EER (R03 → R14)...")
run_eers, run_aucs, valid_runs = [], [], []

for run_id in TASK_SESS:
    mask = (sessions == run_id)
    if mask.sum() == 0:
        print(f"  R{run_id:02d}: NO DATA — skip"); continue
    E_pr, Y_pr     = extract_embeddings(model, X[mask], y[mask])
    sc, lb         = compute_scores(E_pr, Y_pr, P, P_owner)
    eer, auc_v, _, _ = eer_from_scores(sc, lb)
    run_eers.append(eer); run_aucs.append(auc_v); valid_runs.append(run_id)
    print(f"  R{run_id:02d}: EER={eer:.4f}%  AUC={auc_v:.4f}  "
          f"(n_windows={mask.sum()})")

r_val, p_val = pearsonr(valid_runs, run_eers)
print(f"\n  Pearson r = {r_val:.4f}  (p = {p_val:.2e})")
print(f"  EER range: {min(run_eers):.4f}% (R{valid_runs[int(np.argmin(run_eers))]:02d})"
      f" → {max(run_eers):.4f}% (R{valid_runs[int(np.argmax(run_eers))]:02d})")

# ── Fig. 7: ROC — random vs skilled forgery ───────────────────
print("\nComputing Security ROC (random vs skilled forgery)...")

# Use ALL task windows for security ROC
task_mask    = np.isin(sessions, TASK_SESS)
E_all, Y_all = extract_embeddings(model, X[task_mask], y[task_mask])
subj_ids     = sorted(np.unique(Y_all))

genuine_sc, random_sc, skilled_sc = [], [], []

for sid in subj_ids:
    sid_mask = (Y_all == sid)
    if sid_mask.sum() == 0: continue
    E_sub    = E_all[sid_mask]                      # all probe windows for this subject
    own_mask = (P_owner == sid)
    if own_mask.sum() == 0: continue

    # Genuine: mean best-prototype score for this subject's probes
    gen_scores = (E_sub @ P[own_mask].T).max(axis=1)
    genuine_sc.extend(gen_scores.tolist())

    # Random forgery: ALL impostor probe-to-other-gallery scores
    imp_mask = (P_owner != sid)
    imp_sims = E_sub @ P[imp_mask].T            # (n_sub_windows, n_imp_protos)
    # one random impostor score per genuine window = best score from random other subjects
    random_sc.extend(imp_sims.max(axis=1).tolist())

    # Skilled forgery: for this target, find the single BEST impostor match in the dataset
    # i.e., which OTHER subject's embedding is most similar to THIS subject's gallery
    own_proto_mean = P[own_mask].mean(axis=0)
    own_proto_mean = own_proto_mean / (np.linalg.norm(own_proto_mean) + 1e-12)
    # All other subjects' probe embeddings scored against this subject's gallery
    other_mask = (Y_all != sid)
    E_other    = E_all[other_mask]
    skilled_scores_for_sid = (E_other @ P[own_mask].T).max(axis=1)
    skilled_sc.append(float(skilled_scores_for_sid.max()))  # worst-case single attacker

genuine_sc = np.array(genuine_sc);  random_sc  = np.array(random_sc)
skilled_sc = np.array(skilled_sc)

n_gen = len(genuine_sc); n_rnd = len(random_sc); n_skl = len(skilled_sc)
print(f"  Genuine: {n_gen}  |  Random impostors: {n_rnd}  |  Skilled (per-target best): {n_skl}")

# ROC: genuine vs random forgery
sc_rnd = np.concatenate([genuine_sc, random_sc])
lb_rnd = np.concatenate([np.ones(n_gen), np.zeros(n_rnd)])
fpr_rnd, tpr_rnd, _ = roc_curve(lb_rnd, sc_rnd)
auc_rnd = sk_auc(fpr_rnd, tpr_rnd)

# ROC: genuine (per-subject mean) vs skilled forgery (per-subject best impostor)
# Use subject-level means to align dimensions
gen_means = np.array([E_all[Y_all == sid].mean(axis=0) for sid in subj_ids
                      if (Y_all == sid).sum() > 0 and (P_owner == sid).sum() > 0])
gen_means_norm = gen_means / (np.linalg.norm(gen_means, axis=1, keepdims=True) + 1e-12)
gen_self_scores = np.array([float((gen_means_norm[i] @ P[P_owner == subj_ids[i]].T).max())
                             for i in range(len(subj_ids))
                             if (Y_all == subj_ids[i]).sum() > 0 and (P_owner == subj_ids[i]).sum() > 0])
gen_self_scores = gen_self_scores[:len(skilled_sc)]   # align lengths

sc_skl = np.concatenate([gen_self_scores, skilled_sc])
lb_skl = np.concatenate([np.ones(len(gen_self_scores)), np.zeros(len(skilled_sc))])
fpr_skl, tpr_skl, _ = roc_curve(lb_skl, sc_skl)
auc_skl = sk_auc(fpr_skl, tpr_skl)

print(f"  Random forgery  AUC = {auc_rnd:.4f}")
print(f"  Skilled forgery AUC = {auc_skl:.4f}")

# ── SAVE JSON ─────────────────────────────────────────────────
results = {
    "cell": "Cell_10",
    "checkpoint": "seed_3",
    "valid_runs":  valid_runs,
    "run_eers":    run_eers,
    "run_aucs":    run_aucs,
    "pearson_r":   float(r_val),
    "pearson_p":   float(p_val),
    "security_roc": {
        "random_forgery_auc":  float(auc_rnd),
        "skilled_forgery_auc": float(auc_skl),
        "n_genuine":  int(n_gen),
        "n_random":   int(n_rnd),
        "n_skilled":  int(n_skl),
    }
}
json_path = OUT_DIR / "cell10_results.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  ✓ JSON saved: {json_path}")

# ═══════════════════════════════════════════════════════════════
# FIG. 4 — Temporal Drift
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))

# Scatter + line
ax.plot(valid_runs, run_eers, 'o-', color='#1f77b4', linewidth=2.0,
        markersize=8, markerfacecolor='white', markeredgewidth=2.0, zorder=3)

# Regression line
z     = np.polyfit(valid_runs, run_eers, 1)
x_fit = np.linspace(min(valid_runs), max(valid_runs), 100)
ax.plot(x_fit, np.polyval(z, x_fit), '--', color='#d62728', linewidth=1.8,
        label=f'Linear fit  r = {r_val:.3f},  p = {p_val:.1e}', zorder=2)

# Shade ±1 std region
mean_eer = np.mean(run_eers); std_eer = np.std(run_eers)
ax.axhspan(mean_eer - std_eer, mean_eer + std_eer, alpha=0.10, color='#1f77b4')

ax.set_xlabel("Session Run", fontsize=13)
ax.set_ylabel("EER (%)", fontsize=13)
ax.set_title("Per-Session EER: Temporal Drift (B2T Protocol)", fontsize=14)
ax.set_xticks(valid_runs)
ax.set_xticklabels([f"R{r:02d}" for r in valid_runs], fontsize=9)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(min(valid_runs) - 0.3, max(valid_runs) + 0.3)
plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(OUT_DIR / f"FIG_4_temporal_drift.{ext}", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Fig. 4 saved: FIG_4_temporal_drift.pdf/png")

# ═══════════════════════════════════════════════════════════════
# FIG. 7 — Security ROC
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(fpr_rnd, tpr_rnd,  '-',  color='#1f77b4', linewidth=2.2,
        label=f'Random Forgery     (AUC = {auc_rnd:.4f})')
ax.plot(fpr_skl, tpr_skl,  '--', color='#d62728', linewidth=2.2,
        label=f'Skilled Forgery    (AUC = {auc_skl:.4f})')
ax.plot([0, 1], [0, 1], ':',  color='gray', linewidth=1, alpha=0.6)

ax.set_xlabel("False Acceptance Rate (FAR)", fontsize=13)
ax.set_ylabel("True Acceptance Rate (TAR)", fontsize=13)
ax.set_title("Security ROC: Random vs Skilled Forgery\n(DOMCS-EEG, Seed 3)", fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([-0.01, 1.0]); ax.set_ylim([0.0, 1.02])
plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(OUT_DIR / f"FIG_7_security_roc.{ext}", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Fig. 7 saved: FIG_7_security_roc.pdf/png")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CELL 10 COMPLETE")
print("=" * 65)
print(f"  Pearson r (run_index vs EER) = {r_val:.4f}  (p = {p_val:.2e})")
print(f"  EER by run:")
for r_id, eer in zip(valid_runs, run_eers):
    print(f"    R{r_id:02d}: {eer:.4f}%")
print(f"\n  Security ROC:")
print(f"    Random  forgery AUC = {auc_rnd:.4f}")
print(f"    Skilled forgery AUC = {auc_skl:.4f}")
print(f"\n  All outputs saved to: {OUT_DIR}/")
