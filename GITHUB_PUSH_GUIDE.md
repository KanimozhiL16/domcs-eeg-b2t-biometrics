# Push to a PRIVATE GitHub repo + verify reproducibility

Goal: keep everything private until paper acceptance, verify it runs, then flip to public + mint a DOI.
The 9.9 GB EEGMMIDB `.npz` is **excluded by `.gitignore`** — GitHub is code + checkpoints only (~42 MB).

---

## A. One-time setup (skip if you already have git + a GitHub account)
1. Install Git for Windows: https://git-scm.com/download/win (accept defaults).
2. Create a GitHub account: https://github.com (use your VIT email).
3. Open **Git Bash** (right-click in the folder → "Git Bash Here").

## B. Create the PRIVATE repo on GitHub
1. github.com → top-right **+** → **New repository**.
2. Name: `domcs-eeg-tifs`  ·  Description: "DOMCS-EEG IEEE TIFS reproducibility".
3. Select **Private**  (← important: nobody can see it until you choose).
4. Do NOT add README/.gitignore/license (we already have them). Click **Create repository**.
5. Copy the repo URL it shows, e.g. `https://github.com/<your-user>/domcs-eeg-tifs.git`.

## C. Push from your computer (run in Git Bash, inside this folder)
```bash
cd "/c/Users/L.KANIMOZHI/OneDrive/Documents/Claude/Projects/ieee tifs 13 page/DOMCS_EEG_REPRODUCIBILITY"
git init
git add .
git status            # CHECK: no *.npz listed (only code/checkpoints/results/figures/docs)
git commit -m "DOMCS-EEG reproducibility package (code + checkpoints)"
git branch -M main
git remote add origin https://github.com/<your-user>/domcs-eeg-tifs.git
git push -u origin main
```
If it asks to authenticate, log in via the browser popup (or use a Personal Access Token).
✅ After push, the repo on github.com shows code/, checkpoints/, results/, figures/, README.md — and **no .npz**.

> Safety check: in `git status` (before commit) you must NOT see `EEGMMIDB_win2s_step1s_fs128.npz` or any `*BED*win*.npz`. If you do, stop — the `.gitignore` isn't being read; re-run from this folder.

---

## D. Verify reproducibility (do this once, locally — you keep the data on disk)
You have the 9.9 GB `EEGMMIDB_win2s_step1s_fs128.npz` locally. Point the scripts at it:
```bash
# from the DOMCS_EEG_REPRODUCIBILITY folder
export DATA="/c/Users/L.KANIMOZHI/OneDrive/Documents/Claude/Projects/ieee tifs 13 page/1.09 30may26 updated coding/EEGMMIDB_win2s_step1s_fs128.npz"
export CKPT="./checkpoints/08_ablation_ELU_FINAL"
cd code && python 02_eval_main_metrics_disentanglement.py
```
✅ PASS = `results/final_metrics_ELU.json` reports **EER 2.40 ± 0.22 %**, **|cos| 0.0016**, **M1 65.6 %** — matching the bundled copy and the paper. That single match is your proof of reproducibility.

(Figures-only, no data needed: `cd code && python 12_make_figures_from_results.py`.)

---

## E. At acceptance (not now)
1. GitHub repo → Settings → change visibility to **Public**.
2. zenodo.org → log in with GitHub → flip the toggle for `domcs-eeg-tifs`.
3. GitHub → Releases → **Create release** → tag `v1.0` → Publish. Zenodo mints a **code DOI**.
4. Upload the 9.9 GB `.npz` (+ checkpoints if you want a data mirror) to **Zenodo or IEEE DataPort** → **data DOI**.
5. Put both DOIs in the manuscript "Data and Code Availability" paragraph + cover letter.

Until then: repo PRIVATE, data LOCAL. Zero exposure, full reproducibility.
