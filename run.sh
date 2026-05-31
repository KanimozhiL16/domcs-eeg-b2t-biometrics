#!/usr/bin/env bash
# =====================================================================
#  Master reproduction script — DOMCS-EEG (IEEE TIFS)
#  Regenerates the headline metrics and figures from the bundled ELU
#  checkpoints. No retraining required.
#
#  Before running, set the two paths below to your local copies:
#    DATA  = the windowed EEGMMIDB .npz  (see data/README.md to build it)
#    CKPT  = the bundled checkpoints dir (checkpoints/08_ablation_ELU_FINAL)
#  On Code Ocean these live under /data ; results are written to /results.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/code"

export DATA="${DATA:-../data/EEGMMIDB_win2s_step1s_fs128.npz}"
export CKPT="${CKPT:-../checkpoints/08_ablation_ELU_FINAL}"

echo "[1/3] Main EER + disentanglement + ablation (Tables I, III, VI)"
python 02_eval_main_metrics_disentanglement.py        # -> ../results/final_metrics_ELU.json

echo "[2/3] Figures from verified result JSONs (no model load)"
python 12_make_figures_from_results.py                # -> ../figures/*

echo "[3/3] Figures from the E1 model (t-SNE, CMC, per-subject, drift)"
python 13_make_figures_from_model.py                  # loads E1 seed_3

echo "DONE. Compare ../results/final_metrics_ELU.json to the bundled copy:"
echo "   expect EER 2.40 +/- 0.22 %, |cos| 0.0016, balanced state-probe M1 65.6 %."
