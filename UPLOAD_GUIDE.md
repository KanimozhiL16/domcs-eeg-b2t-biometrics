# How to verify & upload the reproducibility package (IEEE T-IFS)

## PART 1 — Verify the package works *before* you upload

Run these in order. Each level is a stronger guarantee; even Level 0 is worth doing.

### Level 0 — figures from saved results (no data, no GPU; ~1 min)
The result JSONs are bundled, so the figure script can run on any laptop:
```bash
cd DOMCS_EEG_REPRODUCIBILITY
pip install -r requirements.txt
cd code && python 13_make_figures_from_results.py
```
✅ Pass = the figures in `../figures/` regenerate (FIG2, FIG4, security, adversarial, stats) and look identical to the bundled ones. This proves the figure pipeline reads the verified numbers correctly.

### Level 1 — metrics from saved checkpoints (needs the .npz + 1 GPU or CPU; ~5 min)
This is the real reproducibility test: recompute the headline numbers from the saved ELU weights, no retraining.
1. Download EEGMMIDB and build `EEGMMIDB_win2s_step1s_fs128.npz` (see `data/README.md`).
2. Edit the `DATA`/`CKPT` paths at the top of `code/02_eval_main_metrics_disentanglement.py` (or `export DATA=... CKPT=...`).
3. Run:
```bash
cd code && python 02_eval_main_metrics_disentanglement.py
```
✅ Pass = `results/final_metrics_ELU.json` shows **EER 2.40 ± 0.22 %**, **|cos| 0.0016**, **M1 65.6 %** — matching the paper and the bundled JSON.

### Level 2 — full retrain (8 GPUs, ~45 min) — optional, strongest
```bash
cd code && python 01_train_all_models_multigpu.py   # 25 jobs → checkpoints/08_ablation_ELU_FINAL
python 02_eval_main_metrics_disentanglement.py
```
✅ Pass = retrained EER is within seed variance of 2.40 %.

### One-command path
`bash run.sh` runs Level 0 + Level 1 together (after you set `DATA`/`CKPT`).

> A reviewer's definition of "reproducible" = *someone else, from your code + data, gets your numbers.* Level 1 demonstrates exactly that. Do Level 1 once yourself; if it matches, the package is sound.

---

## PART 2 — Where to upload

IEEE T-IFS "actively encourages" reproducibility; for deep-learning papers it is effectively expected at review. Use **both** of the two IEEE-endorsed homes:

| Platform | What goes there | What you get | Use for |
|---|---|---|---|
| **Code Ocean** (codeocean.com) — IEEE's recommended **code** platform | `code/`, `requirements.txt`, small checkpoints, `run.sh` | A **"Reproducible Run"** that executes in-browser + a **DOI**; badge shows on IEEE Xplore | The runnable capsule |
| **IEEE DataPort** (ieee-dataport.org) | the windowed `.npz`, full `checkpoints/`, large artifacts | A **DOI** for the data/model artifacts | Big files that don't belong in the capsule |

(Optional: a public **GitHub** repo + **Zenodo** release also gives a DOI; fine as a supplement, but Code Ocean is the one IEEE surfaces on Xplore.)

You do **not** need to upload the raw EEGMMIDB/BED — they are public; you cite them and share only your *derived* preprocessing + code + checkpoints.

---

## PART 3 — Code Ocean, step by step (the runnable capsule)

1. **Account:** go to https://codeocean.com → sign up (free for academic/IEEE authors). IEEE authors can also reach it via the IEEE Author Center → "Code Ocean."
2. **New Capsule → "Create New" → Python** (or "Import from ZIP" and drop `DOMCS_EEG_REPRODUCIBILITY.zip`).
3. **Map the folders to the Code Ocean layout:**
   - Put everything in `code/` (incl. `reviewer_experiments/`) into the capsule **/code**.
   - Put the windowed `.npz` and `checkpoints/` into **/data** (read-only at run time).
   - Leave **/results** empty — scripts write figures/JSON there.
4. **Environment:** Code Ocean → Environment → add Python 3.10 and paste `requirements.txt` (or `pip install -r`). Add a GPU machine type if you want Level 1/2 to run (CPU is fine for Level 0).
5. **Set the run command:** make `run.sh` (included) the capsule's master script, or point it at `code/02_eval_main_metrics_disentanglement.py`. Inside, change `DATA`/`CKPT` to the `/data/...` paths.
6. **Click "Reproducible Run."** Code Ocean executes it end-to-end in a clean container. Green = it reproduces.
7. **Publish** → choose a licence (MIT/BSD for code) → you receive a **DOI** (e.g., `10.24433/CO.xxxxxx.v1`).
8. Keep the capsule **private** until acceptance, then make it public for camera-ready.

## PART 4 — IEEE DataPort, step by step (the data/checkpoints)

1. https://ieee-dataport.org → sign in with your IEEE Account.
2. **"Create Dataset"** → upload the `.npz` + `checkpoints/` (or the whole zip if you prefer one home). Standard upload allows large files.
3. Fill metadata: title (same as paper), abstract, keywords, **Open Access** (recommended) or Standard, licence (CC BY 4.0 is common).
4. Submit → you get a **DOI**.

## PART 5 — Put the DOIs in the paper (required for the claim)

Add/ью update the **"Data and Code Availability"** paragraph (already present in `main.tex`, before the Acknowledgment) with the real DOIs, e.g.:

> *Data and Code Availability.* The EEGMMIDB and BED datasets are public (PhysioNet; IEEE IoT-J). All code, preprocessing, trained checkpoints, and the exact result files are available as a Code Ocean capsule (DOI: 10.24433/CO.XXXXX) and on IEEE DataPort (DOI: 10.21227/XXXX).

Mention the same in the **cover letter** reproducibility line.

---

## Quick checklist
- [ ] Level 1 verification passes locally (EER 2.40 %, |cos| 0.0016 reproduced).
- [ ] `06_eval_per_task_eer.py` re-run on ELU E1 for camera-ready (it currently uses GELU per-task ckpts — supplementary only; see `REPRODUCIBILITY.md` §0).
- [ ] Code Ocean capsule created, "Reproducible Run" green, DOI obtained.
- [ ] IEEE DataPort dataset uploaded, DOI obtained.
- [ ] Both DOIs pasted into the manuscript's Data & Code Availability paragraph and the cover letter.
- [ ] Capsule/dataset kept private until acceptance.
