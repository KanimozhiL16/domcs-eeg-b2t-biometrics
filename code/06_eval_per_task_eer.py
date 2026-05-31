"""
Cell 8-3 — Ablation |cos| per Variant + Per-Task EER Table
============================================================
WHAT THIS DOES:
  Part A: Loads ablation checkpoints E1-E3 (those with state_branch),
          computes mean |cos(z_id, z_state)| on TASK probe windows.
          Fills the MISSING |cos| column in TABLE VI.

  Part B: Groups Cell 10 per-run EER by motor task type
          (Left fist / Right fist / Both fists / Both feet).

WHY |cos| IS CRITICAL:
  E2 (no orth loss) EER=2.59% vs E1 (full) EER=2.62% — only 0.03pp difference.
  Without |cos|, reviewer dismisses orth loss as "no EER improvement".
  E1: |cos| should be LOW   (orth loss enforced)
  E2: |cos| should be HIGHER (no orth loss → entanglement remains)

ARCHITECTURE NOTE:
  Checkpoints in 08_ablation/ were trained with GELU (Cell_8_1).
  This script uses GELU to correctly load those checkpoints.
  Future re-training should use ELU (model_definition_FIXED.py).

RUNTIME: ~3-5 min on A100
"""

import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────── PATHS ───────────────────────────
DATA_PATH    = "/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz"
ABLATION_DIR = "/home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG/08_ablation"
OUT_DIR      = "/home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG/08_ablation"
N_PROBE      = 8000
SEED         = 42

# ─────────── ABLATION VARIANT CONFIG ───────────
# (label, checkpoint_subdir, has_state_branch, EER_from_ablation)
ABLATION_VARIANTS = [
    ("E1 (Full DOMCS-EEG)",  "E1",  True,  2.62),
    ("E2 (No Orth Loss)",    "E2",  True,  2.59),
    ("E3 (No SupCon)",       "E3",  True,  3.04),
    ("E4 (No State Branch)", "E4",  False, 2.62),
    ("E5 (ArcFace Only)",    "E5",  False, 7.64),
]

# ─────────── PER-RUN EER FROM CELL 10 ───────────
PER_RUN_EER = {
    "R03": 1.0671, "R04": 1.1398, "R05": 1.5597, "R06": 1.1083,
    "R07": 2.5718, "R08": 2.7437, "R09": 2.4890, "R10": 3.1914,
    "R11": 2.5476, "R12": 2.4041, "R13": 3.3421, "R14": 2.7795,
}
TASK_GROUPS = {
    "Left Fist":  ["R03", "R07", "R11"],
    "Right Fist": ["R04", "R08", "R12"],
    "Both Fists": ["R05", "R09", "R13"],
    "Both Feet":  ["R06", "R10", "R14"],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────── ARCHITECTURE (GELU — matches Cell_8_1 checkpoints) ──────
class DOMCSAblation(nn.Module):
    """DOMCS-EEG with state_branch. Uses GELU to match Cell_8_1 checkpoints."""
    def __init__(self, n_channels=64, id_dim=128, state_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 64,  kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),  nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(64,  128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.GELU())
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.GELU())
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.id_proj = nn.Linear(256, id_dim,    bias=False)
        self.id_norm = nn.LayerNorm(id_dim)
        self.st_proj = nn.Linear(256, state_dim, bias=False)
        self.st_norm = nn.LayerNorm(state_dim)

    def forward(self, x):
        h = self.pool(self.conv3(self.conv2(self.conv1(x)))).squeeze(-1)
        z_id    = F.normalize(self.id_norm(self.id_proj(h)),          dim=-1)
        z_state = F.normalize(self.st_norm(self.st_proj(h.detach())), dim=-1)
        return z_id, z_state


class DOMCSAblationNoState(nn.Module):
    """DOMCS without state_branch — for E4/E5."""
    def __init__(self, n_channels=64, id_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 64,  kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),  nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(64,  128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.GELU())
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.GELU())
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.id_proj = nn.Linear(256, id_dim, bias=False)
        self.id_norm = nn.LayerNorm(id_dim)

    def forward(self, x):
        h    = self.pool(self.conv3(self.conv2(self.conv1(x)))).squeeze(-1)
        z_id = F.normalize(self.id_norm(self.id_proj(h)), dim=-1)
        return z_id, None


# ─────────────────── HELPERS ──────────────────────────────────
def find_checkpoint(ablation_dir, subdir):
    """Try multiple likely checkpoint locations."""
    candidates = [
        os.path.join(ablation_dir, subdir, "model_best.pt"),
        os.path.join(ablation_dir, subdir, "seed_1", "model_best.pt"),
        os.path.join(ablation_dir, subdir, "seed_3", "model_best.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_checkpoint(ckpt_path, has_state_branch):
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    state = (ckpt.get('model_state') or ckpt.get('model_state_dict') or
             ckpt.get('state_dict') or ckpt)
    model = DOMCSAblation() if has_state_branch else DOMCSAblationNoState()
    model = model.to(DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def compute_cos(model, X_probe, batch_size=512):
    """Mean |cos(z_id, z_state)| over probe windows."""
    vals = []
    with torch.no_grad():
        for i in range(0, len(X_probe), batch_size):
            xb = torch.FloatTensor(X_probe[i:i+batch_size]).to(DEVICE)
            z_id, z_state = model(xb)
            if z_state is None:
                return None
            vals.append(torch.abs(F.cosine_similarity(z_id, z_state, dim=-1)).cpu().numpy())
    return float(np.mean(np.concatenate(vals)))


def to_py(obj):
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    if isinstance(obj, dict):  return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_py(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("  CELL 8-3 — Ablation |cos| per Variant + Per-Task EER")
print("=" * 65)

# ─── LOAD DATA ───
print("\nLoading TASK probe windows...")
data = np.load(DATA_PATH, allow_pickle=True)
X    = data['X'].astype(np.float32)

# Session labels stored as 'R01', 'R02', ... — strip 'R' prefix
sess = np.array([int(str(s).lstrip('R').lstrip('r')) for s in data['session']])

X_task = X[sess >= 3]
print(f"  TASK windows: {len(X_task)}")

rng     = np.random.default_rng(SEED)
idx     = rng.choice(len(X_task), size=min(N_PROBE, len(X_task)), replace=False)
X_probe = X_task[idx]
print(f"  Sampled {len(X_probe)} probe windows for |cos| computation")

# ─── PART A: |cos| per ablation variant ───
print("\n" + "─" * 65)
print("  PART A — |cos(z_id, z_state)| per Ablation Variant")
print("─" * 65)

os.makedirs(OUT_DIR, exist_ok=True)
results_cos = {}

for label, subdir, has_state, eer in ABLATION_VARIANTS:
    ckpt_path = find_checkpoint(ABLATION_DIR, subdir)

    if ckpt_path is None:
        print(f"  ✗  {label}: no checkpoint found under {ABLATION_DIR}/{subdir}/")
        results_cos[label] = {"eer": eer, "cos": None, "has_state": has_state,
                               "note": "checkpoint not found"}
        continue

    model, n_params = load_checkpoint(ckpt_path, has_state)

    if not has_state:
        print(f"  {label}: EER={eer:.2f}% | |cos|=N/A (no state branch) | params={n_params:,}")
        results_cos[label] = {"eer": eer, "cos": None, "has_state": False}
    else:
        cos_val = compute_cos(model, X_probe)
        print(f"  {label}: EER={eer:.2f}% | |cos|={cos_val:.4f} | params={n_params:,}")
        results_cos[label] = {"eer": eer, "cos": cos_val, "has_state": True}

    del model
    torch.cuda.empty_cache()

# ─── PART B: Per-task EER ───
print("\n" + "─" * 65)
print("  PART B — Per-Task EER (from Cell 10 per-run results)")
print("─" * 65)

task_eer = {}
for task_name, runs in TASK_GROUPS.items():
    eers     = [PER_RUN_EER[r] for r in runs]
    mean_eer = float(np.mean(eers))
    std_eer  = float(np.std(eers, ddof=1))
    task_eer[task_name] = {"runs": runs, "eers": eers,
                            "mean": mean_eer, "std": std_eer}
    print(f"  {task_name:<14} ({'/'.join(runs)}): "
          f"EER={mean_eer:.2f}% ± {std_eer:.2f}%  "
          f"[{min(eers):.2f}%–{max(eers):.2f}%]")

# ─── SAVE JSON ───
json_path = os.path.join(OUT_DIR, "cell83_cos_pertask_results.json")
with open(json_path, 'w') as f:
    json.dump(to_py({"cos_per_variant": results_cos, "per_task_eer": task_eer}), f, indent=2)
print(f"\n  ✓ JSON saved: {json_path}")

# ─── GENERATE LaTeX TABLE VI ───
print("\nGenerating TABLE_VI_with_cos.tex ...")

def fmt_cos(cos_val, has_state):
    if not has_state: return r"\multicolumn{1}{c}{—}"
    if cos_val is None: return "—"
    return f"{cos_val:.4f}"

latex_rows = ""
for label, subdir, has_state, eer in ABLATION_VARIANTS:
    r       = results_cos.get(label, {})
    cos_val = r.get("cos", None)
    bold    = label.startswith("E1")
    eer_str = f"\\textbf{{{eer:.2f}}}" if bold else f"{eer:.2f}"
    c_str   = fmt_cos(cos_val, has_state)
    if bold and cos_val is not None:
        c_str = f"\\textbf{{{cos_val:.4f}}}"
    latex_rows += f"{label} & {eer_str} & {c_str} \\\\\n"

latex = (
    r"\begin{table}[!t]" + "\n"
    r"\caption{Ablation Study on EEGMMIDB (B2T Protocol). "
    r"EER$\downarrow$ = Equal Error Rate (\%); "
    r"$|\cos(\mathbf{z}_\text{id},\mathbf{z}_\text{state})|$$\downarrow$ = mean absolute "
    r"cosine similarity (lower = better disentanglement). "
    r"N/A: variant has no state branch.}" + "\n"
    r"\label{tab:ablation}" + "\n"
    r"\centering" + "\n"
    r"\begin{tabular}{lcc}" + "\n"
    r"\hline" + "\n"
    r"Variant & EER (\%)$\downarrow$ & $|\cos(\mathbf{z}_\text{id},\mathbf{z}_\text{state})|$$\downarrow$ \\" + "\n"
    r"\hline" + "\n"
    + latex_rows +
    r"\hline" + "\n"
    r"\end{tabular}" + "\n"
    r"\end{table}" + "\n"
)

tex_path = os.path.join(OUT_DIR, "TABLE_VI_with_cos.tex")
with open(tex_path, 'w') as f:
    f.write(latex)
print(f"  ✓ LaTeX saved: {tex_path}")

# ─── FINAL SUMMARY ───
print("\n" + "=" * 65)
print("  CELL 8-3 COMPLETE")
print("=" * 65)
print("\n|cos(z_id, z_state)| summary:")
for label, subdir, has_state, eer in ABLATION_VARIANTS:
    r = results_cos.get(label, {})
    c = r.get("cos", None)
    if not has_state:
        print(f"  {label}: EER={eer:.2f}% | |cos|=N/A")
    elif c is not None:
        print(f"  {label}: EER={eer:.2f}% | |cos|={c:.4f}")
    else:
        print(f"  {label}: EER={eer:.2f}% | |cos|=FAILED (checkpoint missing)")

print("\nPer-Task EER summary:")
for task_name, vals in task_eer.items():
    print(f"  {task_name:<14}: {vals['mean']:.2f}% ± {vals['std']:.2f}%")

print(f"\nAll outputs → {OUT_DIR}")
print("KEY CHECK: E2 |cos| should be > E1 |cos| (orth loss reduces entanglement)")
print("This proves orth loss improves disentanglement even when EER difference is tiny.")
