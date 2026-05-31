"""
Cell_8_MultiGPU_Launcher.py
============================
Paste the content of this file as a single Jupyter notebook cell.

Runs ALL 25 ablation jobs (5 variants × 5 seeds) across 8 GPUs in parallel.
Each job trains one (variant, seed) on one GPU independently.
No checkpoint reuse — always trains fresh to a NEW output directory.

Estimated time: ~45-60 min (vs ~4+ hrs sequential)
"""

import subprocess, os, sys, time, json
from itertools import product

# ══════════════════════════════════════════════════════════════
#  CONFIG — edit only these three lines
# ══════════════════════════════════════════════════════════════
DATA_PATH  = "/home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz"
WORKER     = "/home/nvidia/24PHD1237/RESULTS_PIPELINE/scripts/train_domcs_worker.py"
OUT_DIR    = "/home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG/08_ablation_ELU_FINAL"
N_GPUS     = 8
# ══════════════════════════════════════════════════════════════

os.makedirs(OUT_DIR, exist_ok=True)

# All 25 jobs: (variant, seed)
VARIANTS = ["E1", "E2", "E3", "E4", "E5"]
SEEDS    = [1, 2, 3, 4, 5]
jobs     = list(product(VARIANTS, SEEDS))  # 25 total

print(f"{'='*60}")
print(f"  MULTI-GPU ABLATION LAUNCHER")
print(f"  Jobs: {len(jobs)} ({len(VARIANTS)} variants × {len(SEEDS)} seeds)")
print(f"  GPUs: {N_GPUS}")
print(f"  Output: {OUT_DIR}")
print(f"{'='*60}\n")

def launch_job(variant, seed, gpu_id):
    """Launch one training job on one GPU, return the Popen handle."""
    cmd = [
        sys.executable, WORKER,
        "--variant", variant,
        "--seed",    str(seed),
        "--gpu",     str(gpu_id),
        "--data",    DATA_PATH,
        "--out",     OUT_DIR,
    ]
    log_path = os.path.join(OUT_DIR, f"log_{variant}_seed{seed}.txt")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_path

# Run in batches of N_GPUS
completed = []
failed    = []
total     = len(jobs)
t_start   = time.time()

for batch_start in range(0, total, N_GPUS):
    batch = jobs[batch_start : batch_start + N_GPUS]
    print(f"── Batch {batch_start//N_GPUS + 1} / {-(-total//N_GPUS)} "
          f"({len(batch)} jobs) ──────────────────────")

    # Launch all jobs in this batch
    running = []
    for i, (variant, seed) in enumerate(batch):
        gpu_id = i % N_GPUS
        proc, log = launch_job(variant, seed, gpu_id)
        running.append((variant, seed, gpu_id, proc, log))
        print(f"  ▶ {variant}/seed_{seed} → GPU {gpu_id}  [PID {proc.pid}]")

    print(f"  Waiting for batch to finish...\n")

    # Wait for all jobs in this batch
    for variant, seed, gpu_id, proc, log in running:
        proc.wait()
        rc = proc.returncode
        if rc == 0:
            ckpt = os.path.join(OUT_DIR, variant, f"seed_{seed}", "model_best.pt")
            exists = "✓" if os.path.exists(ckpt) else "✗ MISSING"
            print(f"  {exists} {variant}/seed_{seed} (GPU {gpu_id}) — rc={rc}")
            if os.path.exists(ckpt):
                completed.append((variant, seed))
            else:
                failed.append((variant, seed))
        else:
            print(f"  ✗ {variant}/seed_{seed} (GPU {gpu_id}) — rc={rc} FAILED")
            print(f"    → check log: {log}")
            failed.append((variant, seed))

    elapsed = (time.time()-t_start)/60
    done = batch_start + len(batch)
    print(f"\n  Progress: {done}/{total} jobs | {elapsed:.1f} min elapsed\n")

# ── FINAL SUMMARY ──────────────────────────────────────────────
total_time = (time.time()-t_start)/60
print(f"\n{'='*60}")
print(f"  ABLATION COMPLETE — {total_time:.1f} min total")
print(f"{'='*60}")
print(f"  ✓ Completed: {len(completed)}/{total}")
if failed:
    print(f"  ✗ Failed:    {len(failed)}: {failed}")

# ── VERIFY ALL CHECKPOINTS ──────────────────────────────────────
print("\n  Checkpoint verification:")
all_ok = True
for v in VARIANTS:
    for s in SEEDS:
        p = os.path.join(OUT_DIR, v, f"seed_{s}", "model_best.pt")
        ok = os.path.exists(p)
        print(f"    {v}/seed_{s}: {'✓' if ok else '✗ MISSING'}")
        if not ok: all_ok = False

if all_ok:
    print("\n  ✓ ALL 25 CHECKPOINTS PRESENT — ready for evaluation")
else:
    print("\n  ⚠ Some checkpoints missing — check logs in OUT_DIR")

print(f"\n  Output directory: {OUT_DIR}")
print(f"  Log files: {OUT_DIR}/log_*.txt")
