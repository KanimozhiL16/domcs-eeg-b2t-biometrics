# DOMCS-EEG — Reproducibility Package

**Paper:** *Security-Aware Cross-State EEG Biometric Verification via Protocol Realism and Representation Disentanglement* (IEEE TIFS submission, T-IFS-26761-2026).
**Authors:** Kanimozhi L., Shridevi S. — Centre for Neuroinformatics, Vellore Institute of Technology, Chennai.

This package contains the code, verified result files, figures, and model checkpoints needed to reproduce every number, table, and figure in the manuscript. It is organized for IEEE T-IFS reproducibility / Code Ocean.

> **Integrity note.** Only the **ELU** model lineage and the **exhaustive all-impostor Baseline-to-Task (B2T)** scorer are included. An earlier **GELU** lineage was a known activation bug and has been **excluded** — with one disclosed exception, the supplementary per-task script `06_eval_per_task_eer.py` (see §0 of `REPRODUCIBILITY.md`). Every script here was run on the author's 8×A100 Brev instance; the JSON files in `results/` are the exact outputs the paper reports. See `REPRODUCIBILITY.md` for the full keep/discard provenance.

## Layout
```
DOMCS_EEG_REPRODUCIBILITY/
├── README.md                 ← you are here
├── REPRODUCIBILITY.md        ← full provenance: script → notebook cell → output → paper number; keep/discard log
├── requirements.txt          ← Python dependencies
├── environment.md            ← hardware / CUDA / framework versions
├── code/                     ← canonical scripts (deduplicated to the final, correct version)
│   └── reviewer_experiments/ ← R8 (10-seed stats), R9 (EEGNet same-task), R10 (transfer attacks), FIG9
├── results/                  ← verified JSON outputs (the exact numbers in the paper)
├── figures/                  ← final TIFS-compliant figures (PDF + PNG)
├── checkpoints/              ← model weights (ELU): 25 ablation + R2 baseline + R8 10-seed DOMCS/DANN
└── data/                     ← README with public-dataset download instructions (data not bundled; see below)
```

## Datasets (public — not bundled due to size/licence)
- **EEGMMIDB** (primary): PhysioNet EEG Motor Movement/Imagery, 109 subjects, 64 ch, 160 Hz. https://physionet.org/content/eegmmidb/1.0.0/
- **BED** (consumer cross-device): Emotiv EPOC+, 14 ch, 21 subjects. https://doi.org/10.1109/JIOT.2021.3061727
- Preprocessing → `data/README.md`. Expected windowed array: `EEGMMIDB_win2s_step1s_fs128.npz` with keys `X` (N,64,256), `y` (subject id), `session` (run index; state = session≥3).

## Quick start (reproduce the headline numbers without retraining)
The result JSONs and checkpoints are included, so you can regenerate every table/figure from the saved ELU checkpoints — no training required.
```bash
pip install -r requirements.txt
# 1) Main EER + disentanglement metrics + ablation (Tables I, III, VI) from the 25 ELU checkpoints:
python code/02_eval_main_metrics_disentanglement.py        # → results/final_metrics_ELU.json  (EER 2.40%, |cos| 0.0016, M1 65.6%)
# 2) All paper figures from the verified JSONs + E1 model:
python code/12_make_figures_from_results.py  # FIG2, FIG4, security, adversarial, stats
python code/13_make_figures_from_model.py # t-SNE, CMC, per-subject, drift  (loads E1 seed_3)
```
Point the `DATA`/`CKPT` paths at the top of each script to your local `.npz` and `checkpoints/` directory.

## Full pipeline (retrain from scratch, 8 GPUs ≈ 45 min)
```bash
python code/01_train_all_models_multigpu.py      # 25 jobs (E1–E5 × 5 seeds) → checkpoints/08_ablation_ELU_FINAL/
python code/02_eval_main_metrics_disentanglement.py        # main EER + ablation + disentanglement
python code/04_security_threat_model_T0_T5.py        # threat model T0–T5 (Table — security)
python code/05_stats_domcs_vs_dann_5seed.py# DOMCS vs DANN
python code/06_eval_per_task_eer.py # per-task EER
python code/06_bed_crossdevice_verification.py        # BED cross-device (EER 22.87%)
python code/08_bed_benchmark_launcher.py          # BED 9-method benchmark
python code/10_temporal_drift_security_roc.py
python code/11_sensitivity_experiments.py       # per-subject/K, λ, enrollment sensitivity
# Reviewer-hardening additions (V3.1):
python code/reviewer_experiments/run_all_reviewer_experiments_8gpu.py   # R8 10-seed DOMCS+DANN, R9 EEGNet, then R8c stats
python code/reviewer_experiments/r10_transfer_blackbox_attacks.py
python code/reviewer_experiments/make_figure_score_distribution.py
```

## Headline verified numbers (all traceable to `results/`)
| Quantity | Value | Source file |
|---|---|---|
| Main B2T EER (EEGMMIDB, 5 seeds) | **2.40 ± 0.22 %** | `results/final_metrics_ELU.json` |
| Inter-subspace cosine \|cos(z_id,z_state)\| | **0.0016** | `results/final_metrics_ELU.json` |
| Balanced state probe on z_id (M1) | **65.6 %** (>50% chance → residual coupling) | `results/final_metrics_ELU.json` |
| Ablation E1→E5 EER / E2:E1 \|cos\| ratio | 2.40 → 7.22 % / **43×** | `results/final_metrics_ELU.json` |
| 10-seed DOMCS vs matched DANN | 2.33±0.14 vs 2.34±0.19, p=0.45 (**parity**) | `results/R8_10seed_stats.json` |
| EEGNet same-task control | id-acc 98.8 %, EER 2.63 % (vs 47.9 % B2T) | `results/R9_eegnet_sametask.json` |
| Targeted-impersonation / transfer (R10) | WB 34.5/38.0 @ε.01; transfer 2.5 % | `code/reviewer_experiments/R10_...py` |
| BED cross-device (5 seeds) | **22.87 ± 0.70 %**, AUC 0.846, \|cos\| 0.0150 | `results/bed_exactarch_results.json` |
| Temporal drift | Pearson r ≈ 0.83–0.86 | `results/cell10_results.json` |

See `REPRODUCIBILITY.md` for the complete script→output→paper-table 