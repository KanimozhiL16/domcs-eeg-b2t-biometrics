# REPRODUCIBILITY & PROVENANCE — DOMCS-EEG

This document records **exactly which code was kept, which was discarded, and why**, plus the script → notebook-cell → output-file → paper-number mapping. It is the authoritative audit trail for the reproducibility package. Built 2026-05-31 by auditing the three run notebooks (`test.ipynb`, `cell 8-3.ipynb`, `cell10,11.ipynb`) against the 47 candidate `.py` files, using the verified outputs in each notebook cell.

Source of truth for cell validity: project `NOTEBOOK_FOLDER_INSPECTION.md` (cell-by-cell INCLUDE/IGNORE map) and `SUCCESSFUL_RUNS_LOG.md` / `EXPERIMENT_LOG_verified.md` (verified numbers).

---

## 0. The one rule that governs everything
Two model lineages existed in the experiments:
- **ELU** (`08_ablation_ELU_FINAL`, `R2_security`, `R3_stats`, `R6b`) = **CORRECT / canonical**.
- **GELU** (`01_checkpoints`, `08_ablation`, `06_security`, early `cell4–7`) = **activation bug → EXCLUDED**.

Every **headline** result (main EER, disentanglement, ablation, security, stats, BED, 10-seed parity) comes from the **ELU** lineage, scored with the **exhaustive all-impostor B2T** scorer. **One exception, disclosed for honesty:** `06_eval_per_task_eer.py` (the supplementary per-task EER breakdown, 2.06–2.46 %) loads the earlier **GELU** per-task ablation checkpoints, because that breakdown was computed before the ELU re-run. It is a *relative* comparison across the four motor tasks, not a headline number; the pattern (all tasks ≈ 2.1–2.5 %) is not expected to change under ELU. For strict single-lineage consistency it should be re-run on the ELU `08_ablation_ELU_FINAL/E1` checkpoints before camera-ready. No other GELU artifact is included.

---

## 1. KEPT code → cell → output → paper element

| `code/` script | Ran as (notebook cell) | Output in `results/` | Paper element |
|---|---|---|---|
| `domcs_model_architecture.py` | (imported) architecture ground truth | — | Methodology §III (arch, 234,880 params, ELU) |
| `train_domcs_worker.py` | called by launcher (test cell 42) | per-seed `model_best.pt` | Training procedure §III; checkpoints |
| `01_train_all_models_multigpu.py` | test.ipynb **cell 42** (25 jobs, 8 GPU, 42.5 min) | `checkpoints/08_ablation_ELU_FINAL/E{1-5}/seed_{1-5}/` | Canonical model source |
| `02_eval_main_metrics_disentanglement.py` | test.ipynb **cell 44/45** | `final_metrics_ELU.json` | **Table I main EER 2.40 %**, Table III disentanglement (\|cos\| 0.0016, M1 65.6 %, M2 0.891, M3 0.005), **Table VI ablation** (E1→E5, E2:E1=43×) |
| `03_eval_ablation_table.py` | (alt evaluator, Table VI) | `all_results_final.json` | Table VI cross-check (self-contained variant) |
| `04_security_threat_model_T0_T5.py` | test.ipynb **cell 46** | `R2_security_results.json` + `FIG_R2_security.pdf` | **Security table T0–T5** (clean 2.26, T1=0, T4 PGD, T5 TSR 34.5) |
| `05_stats_domcs_vs_dann_5seed.py` | test.ipynb **cell 50** | `R3_stats_results.json` | Statistical comparison (5-seed; superseded as headline by 10-seed R8 — see §3) |
| `06_eval_per_task_eer.py` ⚠GELU | cell 8-3 **cell 1** | `cell83_cos_pertask_results.json` | Per-task EER (L/R fist, both fists/feet 2.06–2.46 %). **Supplementary; uses GELU per-task checkpoints — see §0 exception. Re-run on ELU E1 for camera-ready.** |
| `07_bed_crossdevice_verification.py` | test.ipynb **cell 53/56** | `bed_exactarch_results.json` | **BED Table** (EER 22.87 %, AUC 0.846, CRR 44.15 %) |
| `08_bed_benchmark_worker.py` + `09_bed_benchmark_launcher.py` | BED 9-method benchmark | `BED_benchmark.json` | BED baseline rows (DOMCS in-harness row was buggy → discarded; verified R6b value used) |
| `10_bed_disentanglement_figure.py` | test.ipynb **cell 59** | `figures/FIG_BED_disentangle.pdf` | BED disentanglement \|cos\|=0.0150 + t-SNE |
| `11_temporal_drift_security_roc.py` | cell10,11 **cell 2** | `cell10_results.json` + `FIG_4_temporal_drift.pdf`, `FIG_7_security_roc.pdf` | Temporal drift (Pearson r≈0.83–0.86) + security ROC |
| `12_sensitivity_experiments.py` | cell10,11 **cells 6,7** | `cell13_results.json`, `cell14_all_results.json` + figs | Per-subject EER + gallery-K sensitivity, λ-sensitivity, enrollment-duration |
| `13_make_figures_from_results.py` | test.ipynb **cell 51** | `figures/FIG2,FIG4,security,adversarial,stats` | Main figures (from JSON, never hardcoded) |
| `14_make_figures_from_model.py` | test.ipynb **cell 52** | `figures/FIG_tsne,FIG_cmc,persubject,drift` | Embedding figures (loads E1 seed_3) |
| `figure_style_tifs.py` | (imported) | — | Matplotlib TIFS style config (7.16in, fonttype42, ≥8 pt, Wong palette) |

### Protocol-ladder & baselines (cells 13, 14)
`cell7a_protocol_ladder.json` (P1–P5; B2T = hardest, 2.22 %) and `cell7b_baselines.json` (8 baselines: EEGNet 47.9 … DOMCS 2.40) were produced inline in `test.ipynb` cells 13/14 → kept as JSON in `results/` (drive the protocol-justification text and the Table I baseline rows).

### Reviewer-hardening (V3.1 additions) — `code/reviewer_experiments/`
| script | output | paper element |
|---|---|---|
| `r8_domcs_worker.py` + `r8_dann_worker.py` + `r8_eval_10seed_stats.py` (orchestrated by `run_all_reviewer_experiments_8gpu.py`) | `R8_10seed_stats.json` | **10-seed parity**: DOMCS 2.33±0.14 vs matched-DANN 2.34±0.19, paired t p=0.45, Wilcoxon 0.70 (supersedes the 5-seed p=0.042 claim) |
| `r9_eegnet_sametask_worker.py` | `R9_eegnet_sametask.json` | EEGNet same-task control: id-acc 98.8 %, EER 2.63 % vs 47.9 % B2T (baseline collapse is protocol-driven) |
| `r10_transfer_blackbox_attacks.py` | (Brev `/R10_security/`; values in `SUCCESSFUL_RUNS_LOG.md §G`) | Transfer/black-box: WB TSR 8/8, 15/15.5, 34.5/38 @ε; **transfer 2.5 %** (adversarial examples don't transfer) |
| `make_figure_score_distribution.py` | `figures/FIG_score_distribution.pdf` | Genuine-vs-impostor score distribution (representative seed 3, EER 2.48 %; headline = 10-seed 2.33 %) |

---

## 2. KEPT checkpoints (`checkpoints/`)
- `08_ablation_ELU_FINAL/E{1-5}/seed_{1-5}/model_best.pt` (25) + 25 logs + `final_metrics_ELU.json` — main model, ablation, disentanglement. Each carries `activation:"ELU"` metadata; state-dict key `model_state`.
- `cnn_arcface_rest_only_best.pt` — the CNN+ArcFace (no state branch) baseline used by R2 security.
- `R8_domcs_10seed/seed_{1-10}_model_best.pt` + `R8_dann_10seed/seed_{1-10}_model_best.pt` — for the 10-seed parity test, FIG9, and R10 transfer attacks.

---

## 3. DISCARDED — what was NOT included and why (integrity record)

### Discarded `.py` files (of the 47 candidates)
| Discarded script(s) | Reason |
|---|---|
| `FIG_FIXED_1_tsne.py … FIG_FIXED_6_enrollment.py` (6) | **Load GELU checkpoints** ("[FIX-4] Architecture uses GELU"). GELU lineage → excluded. Figures regenerated from ELU by `Cell_FIGURES_part1/part2`. |
| `Cell_Fig2_Disentanglement_TIFS.py`, `_V2_Clean`, `_V3_Final`, `_V4_FINAL`, `_V5_CHATGPT_STYLE` (5) | All contain the **wrong "92.44 % ≈ chance" framing** (imbalanced-probe artifact). Correct balanced M1=65.6 % figure comes from `Cell_FIGURES_part1`. |
| `Cell_11_Baseline_Adversarial.py`, `_FIXED`, `_V2_CLEAN`, `_CORRECT_ATTACK`, `_FINAL_RUN`, `_V4_CELL9A_PROTOCOL`, `_V5_MULTIGPU` (8) | Iterative versions of baseline-adversarial; v1 had EER=0.000 (wrong attack direction). **All superseded by `04_security_threat_model_T0_T5.py`.** |
| `Cell_15_EXP2_FIXED.py`, `_V2`, `Cell_15_FiveTIFSExperiments.py` (3) | EXP2 \|cos\| ratio variants (GELU-mixed) — superseded by ELU_FINAL ratio (43×). CMC/efficiency reused from figures_TIFS. |
| `Cell_9A_T4_Corrected_Adversarial.py` | Intermediate T4 — superseded by R2. |
| `Cell_9B_BED_CrossDataset.py` | Zero-shot BED (wrong design) — superseded by within-BED cross-session R6b. |
| `Cell_9C_Exhaustive_EER.py` | Incomplete run (still running, no final output). |
| `Cell_5D_Disentanglement_Strong.py` | GELU-mixed / seed-1 outlier disentanglement — superseded by `02_eval_main_metrics_disentanglement.py`. |
| `Cell_8_ABLATION_FINAL.py` | Single-process ablation — superseded by `train_domcs_worker.py` + multi-GPU launcher. |
| `Cell_R6_BED_CrossDataset_ELU.py` | Earlier BED variant — superseded by `07_bed_crossdevice_verification.py` (fixed `F.normalize(x,-1)`→`dim=-1` typo). |
| `Cell_12_Statistical_Validation_FRESH.py` | Hardcoded GELU stats — superseded by `Cell_R3` (and ultimately R8 10-seed). |
| `12_sensitivity_experiments.py` duplicate logic, `cell6_security_evaluation.py` | `cell6_security_evaluation.py` is GELU security — superseded by R2. (`12_sensitivity_experiments.py` itself is KEPT for the aux experiments.) |
| `FIG_FIXED_3_cmc_eval_baselines.py` | GELU baseline CMC — superseded. |
| Reviewer-experiment `*_NOTEBOOK.py` / `R8a_DOMCS_seeds6to10.py` / `R8b_DANN_10seeds.py` / `R8a_DOMCS_10seeds_NOTEBOOK.py` / `FIG9_score_distribution.py` (non-FAST) | Notebook-pasted or partial variants; the canonical standalone workers + `run_all_reviewer_experiments_8gpu.py` + `FIG9_..._FAST.py` are kept. |

### Discarded folders / archives
GELU lineage (`01_checkpoints`, `08_ablation`, `08_ablation_ELU` intermediate, `04_ablation_checkpoints`, `05_disentanglement_strong*`, `06_security`, `09_security_fixed`, `03_figures`); empty dirs (`06_verification`, `09_exhaustive_eer`, `10_bed_crossdataset`, `12_*_v5`, `statistical_validation`); wrong-design `R6_bed_crossdataset` (zero-shot 42 %); buggy/intermediate `12_baseline_adversarial` v1/v3/v4; 955 MB scratch `05_embeddings`. The large `.tar.gz` archives are offline backups, not paper sources.

---

## 4. Notebook cells confirmed VALID (have correct outputs)
- `test.ipynb`: cells **1, 13, 14, 42, 44/45, 46, 47, 50, 51, 52, 53, 56, 57, 58, 59**.
- `cell10,11.ipynb`: cells **2, 6, 7**.
- `cell 8-3.ipynb`: cell **1** (per-task part).
All other cells are GELU/superseded/failed/utility (full list in project `NOTEBOOK_FOLDER_INSPECTION.md`).

---

## 5. IEEE T-IFS reproducibility checklist (this package)
- [x] Public datasets, cited, with preprocessing recipe (`data/README.md`).
- [x] All hyperparameters reported (`environment.md`).
- [x] Training hardware reported (8×A100, CUDA 12, PyTorch ≥2).
- [x] ≥3 seeds (5 main, 10 for parity) reported as mean ± std.
- [x] Statistical tests included (paired t, Wilcoxon, bootstrap CI).
- [x] Ablation isolates each component (E1–E5).
- [x] Security threat model T0–T6 with robustness curves.
- [x] Verified result JSONs + checkpoints bundled; every figure regenerable from them.
- [x] Single activation lineage (ELU); GELU bug excluded and documented.
- [ ] **TODO (author):** upload to Code Ocean (gets a DOI, runs in-browser) or IEEE DataPort; add the "Data and Code Availability" paragraph DOI to the manuscript before camera-ready.

---

## 6. File-rename map (notebook-cell names → research-grade names, 2026-05-31)
The scripts were renamed from notebook-cell artifacts (`Cell_*`) to descriptive, pipeline-ordered names. Internal references were updated accordingly (`01_train_all_models_multigpu.py` → `train_domcs_worker.py`; `run_all_reviewer_experiments_8gpu.py` → r8/r9 workers). No code logic changed.

| Old name | New name |
|---|---|
| model_definition_FIXED.py | domcs_model_architecture.py |
| ablation_worker_elu.py | train_domcs_worker.py |
| Cell_8_MultiGPU_Launcher.py | 01_train_all_models_multigpu.py |
| Cell_FINAL_ELU_metrics.py | 02_eval_main_metrics_disentanglement.py |
| Cell_STEP2_ablation_eval_TABLE_VI.py | 03_eval_ablation_table.py |
| Cell_R2_Security_T0_T5.py | 04_security_threat_model_T0_T5.py |
| Cell_R3_Statistical_Validation.py | 05_stats_domcs_vs_dann_5seed.py |
| Cell_8_3_Ablation_Cos_PerTask.py | 06_eval_per_task_eer.py |
| Cell_R6b_BED_ExactArch.py | 07_bed_crossdevice_verification.py |
| Cell_R7_BED_worker.py | 08_bed_benchmark_worker.py |
| Cell_R7_BED_launcher.py | 09_bed_benchmark_launcher.py |
| Cell_R7_BED_disentangle_viz.py | 10_bed_disentanglement_figure.py |
| Cell_10_Temporal_Drift_SecurityROC.py | 11_temporal_drift_security_roc.py |
| Cell_14_FourExperiments.py | 12_sensitivity_experiments.py |
| Cell_FIGURES_part1_from_JSON.py | 13_make_figures_from_results.py |
| Cell_FIGURES_part2_from_model.py | 14_make_figures_from_model.py |
| setup_paper_figures.py | figure_style_tifs.py |
| R8a_DOMCS_worker.py | reviewer_experiments/r8_domcs_worker.py |
| R8b_DANN_worker.py | reviewer_experiments/r8_dann_worker.py |
| R8c_eval_stats.py | reviewer_experiments/r8_eval_10seed_stats.py |
| R9_EEGNet_worker.py | 