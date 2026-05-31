# Security-Aware Cross-State EEG Biometric Verification (DOMCS-EEG)

Official code, trained models, and verified results for the IEEE Transactions on Information
Forensics and Security manuscript *"Security-Aware Cross-State EEG Biometric Verification via
Protocol Realism and Representation Disentanglement"* (T-IFS-26761-2026).

**Authors:** Kanimozhi L., Shridevi S. — Centre for Neuroinformatics, Vellore Institute of Technology, Chennai, India.
**ORCID:** 0009-0003-2093-0969 (Kanimozhi L.), 0000-0002-0038-7212 (Shridevi S.)

---

## 1. Overview

DOMCS-EEG performs cross-state EEG biometric verification: a user is enrolled at rest and verified
while performing an arbitrary task (the Baseline-to-Task, B2T, protocol). A one-dimensional
convolutional encoder feeds an identity branch and a stop-gradient state branch constrained to be
orthogonal, trained with an additive angular-margin loss, supervised contrastive loss, a state
classifier, and an orthogonality penalty. The system is evaluated as a security primitive under an
explicit threat model (forgery, gallery leakage, signal corruption, white-box FGSM/PGD, and
black-box transfer).

This repository lets a reader confirm every reported number and figure at four increasing levels of
effort, from a check that needs only the repository to a full multi-GPU retraining.

---

## 2. Repository structure

```
.
├── reproduce_no_data.py        # one-command numeric verification (no data, GPU, or weights)
├── requirements.txt            # Python dependencies
├── environment.md              # hardware, CUDA, framework versions, hyperparameters
├── HOW_TO_REPRODUCE.md         # extended reproduction notes
├── code/                       # canonical pipeline scripts (paths set via DATA / CKPT env vars)
│   └── reviewer_experiments/   # 10-seed statistics, EEGNet control, transfer attacks, score distribution
├── results/                    # verified JSON outputs — the exact values reported in the paper
├── figures/                    # publication figures (PDF + PNG)
├── checkpoints/                # ELU model weights: 25 ablation + baseline + 10-seed DOMCS/DANN
└── data/                       # dataset acquisition and preprocessing instructions (data not bundled)
```

---

## 3. Requirements

Python 3.10, PyTorch >= 2.0, NumPy, SciPy, scikit-learn, Matplotlib (see `requirements.txt`).
Training was performed on 8x NVIDIA A100 GPUs; evaluation runs on a single GPU or CPU. Full
hardware and hyperparameter details are in `environment.md`.

```bash
git clone https://github.com/KanimozhiL16/domcs-eeg-b2t-biometrics.git
cd domcs-eeg-b2t-biometrics
pip install -r requirements.txt
```

---

## 4. Datasets

| Dataset | Role | Subjects | Channels | Availability |
|---------|------|:--------:|:--------:|--------------|
| EEGMMIDB (PhysioNet) | primary | 109 | 64 | Public: https://physionet.org/content/eegmmidb/1.0.0/ |
| BED | cross-device | 21 | 14 | Restricted (author request): https://doi.org/10.5281/zenodo.4309471 |

Raw recordings are not redistributed here. The preprocessing recipe (band-pass, 128 Hz resampling,
2 s / 1 s windowing) and the expected array format (`EEGMMIDB_win2s_step1s_fs128.npz` with keys
`X`, `y`, `session`) are documented in `data/README.md`. All scripts read the dataset and checkpoint
locations from the `DATA` and `CKPT` environment variables, so no source edits are required.

---

## 5. Reproduction

Following IEEE T-IFS guidance, the primary evidence of reproducibility is that the reported numbers
are obtained by executing the released code. Four reproduction tiers are provided.

### Tier 1 — Verify reported metrics (no dataset, no GPU, no checkpoints; ~30 s)
Prints every headline metric directly from `results/*.json` beside its value in the paper.
```bash
python reproduce_no_data.py
```

### Tier 2 — Regenerate figures (no dataset, no GPU, no checkpoints; ~1 min)
Rebuilds the main figures from the same verified result files.
```bash
python code/12_make_figures_from_results.py        # writes to figures/
```

### Tier 3 — Re-evaluate from trained checkpoints (dataset + checkpoints; ~5 min)
Recomputes the metrics from the released model weights. Requires the EEGMMIDB array and a machine
with sufficient RAM (the array is ~10 GB in memory) or a high-RAM session.
```bash
export DATA=/path/to/EEGMMIDB_win2s_step1s_fs128.npz
export CKPT="$(pwd)/checkpoints/08_ablation_ELU_FINAL"
cd code && python 02_eval_main_metrics_disentanglement.py
```

### Tier 4 — Full retraining (dataset + multi-GPU; ~45 min on 8 GPUs)
Retrains all 25 ELU models and re-evaluates end to end.
```bash
export DATA=/path/to/EEGMMIDB_win2s_step1s_fs128.npz
export CKPT="$(pwd)/checkpoints/08_ablation_ELU_FINAL"
cd code
python 01_train_all_models_multigpu.py
python 02_eval_main_metrics_disentanglement.py
python 04_security_threat_model_T0_T5.py
python 05_stats_domcs_vs_dann_5seed.py
python 06_bed_crossdevice_verification.py
```

A ready-to-run Google Colab cell for Tier 1 and Tier 2:
```python
!git clone https://github.com/KanimozhiL16/domcs-eeg-b2t-biometrics.git
%cd domcs-eeg-b2t-biometrics
!pip install -r requirements.txt -q
!python reproduce_no_data.py
!python code/12_make_figures_from_results.py
```

---

## 6. Results

| Quantity | Value | Source |
|----------|-------|--------|
| EEGMMIDB B2T EER (5 seeds) | 2.40 ± 0.22 % | `results/final_metrics_ELU.json` |
| AUC | 0.996 | `results/final_metrics_ELU.json` |
| Inter-subspace cosine \|cos(z_id, z_state)\| | 0.0016 | `results/final_metrics_ELU.json` |
| Balanced state probe on z_id | 65.6 % (chance 50) | `results/final_metrics_ELU.json` |
| Ablation EER (E1→E5); E2:E1 \|cos\| ratio | 2.40 → 7.22 %; 43× | `results/final_metrics_ELU.json` |
| 10-seed DOMCS vs. strength-matched DANN | 2.33 % vs. 2.34 % (p = 0.45, parity) | `results/R8_10seed_stats.json` |
| EEGNet same-task control | id-acc 98.8 %, EER 2.63 % (vs. 47.9 % under B2T) | `results/R9_eegnet_sametask.json` |
| BED cross-device EER (5 seeds) | 22.87 ± 0.70 %, AUC 0.846 | `results/bed_exactarch_results.json` |

The representation is characterized as exhibiting partial decorrelation with residual coupling
(state-attenuated, not state-free): the inter-subspace cosine is near zero while a balanced probe
still recovers state at 65.6 %.

---

## 7. Model and training summary

A three-block 1-D CNN encoder (ELU activations) produces a 256-dimensional feature, projected into a
128-dimensional identity embedding and a 128-dimensional state embedding (the latter from a
stop-gradient feature). Training objective: ArcFace (s = 32, m = 0.50) + 0.30·SupCon (τ = 0.07) +
0.50·state cross-entropy + 0.10·orthogonality. Optimizer Adam (3e-4 / 1e-4, cosine decay), 60 epochs,
batch size 256, identity losses on resting windows only. Total parameters 234,880 (201,856 at
inference). Full specification: `environment.md`.

---

## 8. Citation

```bibtex
@article{kanimozhi2026domcseeg,
  title   = {Security-Aware Cross-State EEG Biometric Verification via Protocol Realism and Representation Disentanglement},
  author  = {Kanimozhi, L. and Shridevi, S.},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  note    = {Under review (T-IFS-26761-2026)}
}
```

---

## 9. License and data use

Code in this repository is released for academic, non-commercial reproduction of the manuscript.
EEGMMIDB is governed by the PhysioNet license; BED is governed by its authors' access terms. Users
must obtain each dataset from its official source under the corresponding license.

---

## 10. Notes on scope

Only the ELU model lineage and the exhaustive all-impostor B2T scorer are included; an earlier GELU
variant present during development was an activation bug and is excluded. Every file in `results/` is
the exact output of the executed pipeline.
