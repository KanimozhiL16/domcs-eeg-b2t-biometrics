# DOMCS-EEG — Reproducibility Package

**Paper:** *Security-Aware Cross-State EEG Biometric Verification via Protocol Realism and Representation Disentanglement* (IEEE TIFS, T-IFS-26761-2026).
**Authors:** Kanimozhi L., Shridevi S. — Centre for Neuroinformatics, Vellore Institute of Technology, Chennai.
**Repository:** https://github.com/KanimozhiL16/domcs-eeg-b2t-biometrics

This package reproduces every number, table, and figure in the manuscript. It is organised so a reader can verify the work at **four increasing levels of effort** — from a 30-second check that needs nothing but this repo, up to a full 8-GPU retrain.

> **Integrity note.** Only the **ELU** model lineage and the **exhaustive all-impostor Baseline-to-Task (B2T)** scorer are included. An earlier GELU variant was a known activation bug and is excluded. Every JSON in `results/` is the exact output of the executed pipeline. Scripts read their data/checkpoint locations from the `DATA` and `CKPT` environment variables, so they run anywhere without editing.

---

# What "reproducible" means (IEEE T-IFS)
IEEE T-IFS expects (i) the code is available, (ii) the **reported numbers can be obtained by running it**, and (iii) the data/models are accessible or clearly described. A **numeric match is the proof**; figure regeneration is optional. This repo gives you both.

---

# THE 4 WAYS TO REPRODUCE

| Way | Needs data? | Needs GPU? | Needs checkpoints? | Time | What it proves |
|-----|:-----------:|:----------:|:------------------:|------|----------------|
| **1. Verify numbers** | No | No | No | ~30 s | The paper's numbers are the exact pipeline outputs |
| **2. Regenerate figures** | No | No | No | ~1 min | The paper's figures come from those same numbers |
| **3. Re-evaluate from checkpoints** | Yes (EEG `.npz`) | optional | Yes | ~5 min | The numbers come from the trained models, not stored JSON |
| **4. Full retrain from scratch** | Yes | Yes (8×GPU) | No (creates them) | ~45 min | The whole result reproduces end-to-end |

First: `git clone https://github.com/KanimozhiL16/domcs-eeg-b2t-biometrics.git && cd domcs-eeg-b2t-biometrics`
Then `pip install -r requirements.txt` (Ways 3–4; Way 1 needs nothing, Way 2 needs only matplotlib).

---

## WAY 1 — Verify the numbers (no data, no GPU, no checkpoints)
Prints every headline metric straight from `results/*.json`, next to the paper value.

**Laptop / terminal**
```bash
python reproduce_no_data.py
```

**Google Colab** (paste into one cell)
```python
!git clone https://github.com/KanimozhiL16/domcs-eeg-b2t-biometrics.git
%cd domcs-eeg-b2t-biometrics
!python reproduce_no_data.py
```
✅ Expected (matches the manuscript):
```
[1] EER 2.40 +/- 0.22 % | AUC 0.996 | |cos| 0.0016 | state-probe 65.6 % | M2 0.891 | M3 0.005
[2] Ablation E1 2.40 .. E5 7.22 % ; orthogonality E2:E1 |cos| ratio = 43.1x
[3] Security: clean 2.26/2.50 ; PGD@.01 19.73/24.29 ; TSR 34.5/45.5  (DOMCS / CNN+ArcFace)
[4] 10-seed DOMCS 2.33 = DANN 2.34 (p=0.45, parity) ; EEGNet same-task 98.8 %/2.63 % vs 47.88 % B2T
[5] BED 22.87 +/- 0.70 % | AUC 0.846
```

---

## WAY 2 — Regenerate the figures (no data, no GPU, no checkpoints)
Rebuilds the main paper figures from the same verified `results/*.json` (no model load).

**Laptop / terminal**
```bash
pip install matplotlib numpy
python code/12_make_figures_from_results.py      # writes PDFs/PNGs into figures/
```

**Google Colab**
```python
!git clone https://github.com/KanimozhiL16/domcs-eeg-b2t-biometrics.git
%cd domcs-eeg-b2t-biometrics
!pip install matplotlib numpy -q
!python code/12_make_figures_from_results.py
```
✅ `figures/` is regenerated (disentanglement, ablation, security, adversarial, stats). Compare to the committed copies.

---

## WAY 3 — Re-evaluate from the trained checkpoints (needs data, no training)
Recomputes the metrics from the saved ELU model weights — this is the real reproducibility test. Needs the windowed EEGMMIDB `.npz` (see `data/README.md` to download/build it; ~9.9 GB) and the checkpoints in `checkpoints/08_ablation_ELU_FINAL/`.

**Laptop / terminal**
```bash
pip install -r requirements.txt
export DATA="/full/path/to/EEGMMIDB_win2s_step1s_fs128.npz"
export CKPT="$(pwd)/checkpoints/08_ablation_ELU_FINAL"
cd code && python 02_eval_main_metrics_disentanglement.py     # -> ../results/final_metrics_ELU.json
```

**Google Colab** (after uploading the `.npz` to the session or mounting Drive)
```python
!git clone https://github.com/KanimozhiL16/domcs-eeg-b2t-biometrics.git
%cd domcs-eeg-b2t-biometrics
!pip install -r requirements.txt -q
import os
os.environ["DATA"]="/content/EEGMMIDB_win2s_step1s_fs128.npz"          # <-- your path
os.environ["CKPT"]="/content/domcs-eeg-b2t-biometrics/checkpoints/08_ablation_ELU_FINAL"
!cd code && python 02_eval_main_metrics_disentanglement.py
```
✅ Writes `results/final_metrics_ELU.json` with **EER 2.40 %, |cos| 0.0016, M1 65.6 %**.
Other evaluations the same way: `04_security_threat_model_T0_T5.py`, `05_stats_domcs_vs_dann_5seed.py`, `06_bed_crossdevice_verification.py`.

---

## WAY 4 — Full retrain from scratch (needs data + GPUs, recreates everything)
Trains all 25 ELU models (E1–E5 × 5 seeds) then re-evaluates. ~45 min on 8 GPUs.

**Terminal (multi-GPU machine)**
```bash
pip install -r requirements.txt
export DATA="/full/path/to/EEGMMIDB_win2s_step1s_fs128.npz"
export CKPT="$(pwd)/checkpoints/08_ablation_ELU_FINAL"
cd code
python 01_train_all_models_multigpu.py        # trains 25 jobs -> checkpoints/08_ablation_ELU_FINAL/
python 02_eval_main_metrics_disentanglement.py # main EER + disentanglement + ablation
python 04_security_threat_model_T0_T5.py       # threat model T0-T5
python 05_stats_domcs_vs_dann_5seed.py         # DOMCS vs DANN
python 06_bed_crossdevice_verification.py      # BED cross-device (EER 22.87 %)
# reviewer-hardening (10-seed stats, EEGNet, transfer attacks, score-dist figure):
cd reviewer_experiments && python run_all_reviewer_experiments_8gpu.py && python r10_transfer_blackbox_attacks.py && python make_figure_score_distribution.py
```
✅ Retrained EER is within seed variance of 2.40 %.

---

# Datasets
- **EEGMMIDB** (primary, public): PhysioNet EEG Motor Movement/Imagery — 109 subjects, 64 ch. https://physionet.org/content/eegmmidb/1.0.0/  Preprocessing recipe → `data/README.md`. Expected file `EEGMMIDB_win2s_step1s_fs128.npz` (keys `X`,`y`,`session`).
- **BED** (consumer cross-device): **license-restricted** — request from the authors at https://doi.org/10.5281/zenodo.4309471 . We ship only the BED scripts, not the BED data.

# Layout
```
reproduce_no_data.py        Way 1 — verify numbers (no data/GPU/ckpt)
HOW_TO_REPRODUCE.md         long-form reproduction notes
code/                       canonical scripts (env-overridable paths) + reviewer_experiments/
results/                    verified JSON outputs (the exact paper numbers)
figures/                    final TIFS figures (PDF + PNG)
checkpoints/                ELU weights: 25 ablation + R2 baseline + R8 10-seed DOMCS/DANN
data/README.md             how to obtain/build the datasets (data not bundled)
requirements.txt           Python dependencies   |   environment.md  hardware/HP
```

# Headline numbers (from results/)
EER **2.40 ± 0.22 %** · AUC 0.996 · |cos| **0.0016** · state-probe **65.6 %** · ablation 2.40→7.22 (E2:E1 |cos| **43×**) · 10-seed DOMCS **2.33 %** = DANN 2.34 % (parity) · EEGNet same-task 98.8 %/2.63 % vs 47.88 % B2T · BED **22.87 ± 0.70 %** / AUC 0.846.
