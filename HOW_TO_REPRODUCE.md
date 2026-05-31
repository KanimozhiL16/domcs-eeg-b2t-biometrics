# How to reproduce DOMCS-EEG

There are **two levels**. Level 1 needs nothing but the repo — perfect for Google Colab or any clone.

---

## What "reproducible" means here (IEEE T-IFS)
IEEE T-IFS expects that (i) the code is available, (ii) the **reported numbers can be obtained by running it**, and (iii) the data/models are accessible or clearly described. A numeric match is the proof — figure regeneration is optional. This repo provides both, but **Level 1 (numbers) is the proof of correctness.**

## Level 1 — Clone & verify the NUMBERS without the dataset (under 1 min, no GPU)
Prints every headline metric directly from the verified result files committed in `results/`,
next to the paper's reported value. No 9.9 GB data, no checkpoints, no GPU.

### On your laptop
```bash
git clone https://github.com/KanimozhiL16/domcs-eeg-b2t-biometrics.git
cd domcs-eeg-b2t-biometrics
pip install -r requirements.txt
python reproduce_no_data.py
```

### On Google Colab (copy these into a cell)
```python
!git clone https://github.com/KanimozhiL16/domcs-eeg-b2t-biometrics.git
%cd domcs-eeg-b2t-biometrics
!pip install -r requirements.txt -q
!python reproduce_no_data.py
```
✅ Expected output (every number matches the manuscript):
```
[1] MAIN: EER 2.40 +/- 0.22 % | AUC 0.996 | |cos| 0.0016 | state-probe 65.6 % | M2 0.891
[2] ABLATION: E1 2.40 .. E5 7.22 % ; orthogonality E2:E1 |cos| ratio = 43.1x
[3] SECURITY: clean 2.26 vs 2.50 ; PGD 19.73 vs 24.29 ; TSR 34.5 vs 45.5 (DOMCS vs CNN+ArcFace)
[4] STATS: 10-seed DOMCS 2.33 % = DANN 2.33 % (parity) ; EEGNet same-task 98.8 %/2.63 % vs 47.88 % B2T
[5] BED: EER 22.87 +/- 0.70 % | AUC 0.846 | CRR 44.15 %
```
This is the reproducibility proof for a reviewer with no data access: the numbers in the
paper are the exact contents of `results/*.json`, produced by the executed pipeline.

(Optional) to also regenerate the figures from those same JSONs:
`python code/12_make_figures_from_results.py`  → writes `figures/`.

---

## Level 2 — Re-derive the numbers from the trained models (needs data + checkpoints)
This recomputes the metrics from the model weights, not the saved JSONs.
The 9.9 GB preprocessed EEG and the model checkpoints are hosted separately (see below).

```bash
# 1) get the data + checkpoints (hosted on Zenodo/IEEE DataPort — DOIs in README "Data and Code Availability")
#    EEGMMIDB_win2s_step1s_fs128.npz   and   checkpoints/08_ablation_ELU_FINAL/
# 2) point the scripts at them and run:
export DATA=/path/to/EEGMMIDB_win2s_step1s_fs128.npz
export CKPT=/path/to/checkpoints/08_ablation_ELU_FINAL
cd code && python 02_eval_main_metrics_disentanglement.py
```
✅ Writes `results/final_metrics_ELU.json` with the same headline numbers (EER 2.40 %, |cos| 0.0016, M1 65.6 %).

Notes:
- EEGMMIDB is public (PhysioNet) — the preprocessed `.npz` may be shared; see `data/README.md` for the windowing recipe to rebuild it from scratch.
- **BED is license-restricted** — request it from the authors (Zenodo `10.5281/zenodo.4309471`); we ship only the BED scripts, not the BED data.
- Every `code/` script reads its data/checkpoint paths from the `DATA` and `CKPT` environment variables (falling back to the original training paths), so they run anywhere without editing.
