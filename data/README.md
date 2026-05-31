# Data — download & preprocessing

Raw EEG data is **not bundled** (size + redistribution licence). Both datasets are public.

## 1. EEGMMIDB (primary)
- **Source:** PhysioNet EEG Motor Movement/Imagery Database — 109 subjects, 64 channels, 160 Hz, 14 runs/subject (R01–R14). https://physionet.org/content/eegmmidb/1.0.0/
- **Acquisition system:** BCI2000.
- **Runs used:** R01–R02 = resting (eyes-open/closed) → enrollment gallery; R03–R14 = motor movement/imagery tasks → probe.

### Preprocessing → windowed array
1. Band-pass 1–40 Hz, resample to **128 Hz**.
2. Segment into **2 s windows, 1 s step** (50% overlap).
3. Per-window per-channel z-score.
4. Save as `EEGMMIDB_win2s_step1s_fs128.npz` with keys:
   - `X` : float32 `(N, 64, 256)`  — N ≈ 173,198 windows
   - `y` : int    `(N,)`           — subject id (0–108)
   - `session` : int `(N,)`        — run index; **state = (session ≥ 3)** (REST vs TASK)
5. Point the `DATA=` path at the top of each script in `../code/` to this `.npz`.

## 2. BED (consumer cross-device validation)
- **Source:** BED — a dataset for EEG-based biometrics, Emotiv EPOC+ headset, **14 channels, 21 subjects**, multi-session. https://doi.org/10.1109/JIOT.2021.3061727
- **Protocol used:** within-BED **cross-session** — enroll on sessions r01+r02, probe on r03 (NOT zero-shot). Same 2 s/1 s/128 Hz windowing; `C_in = 14`.

## Notes
- All windowing/splitting is deterministic given the seed (see `../environment.md`).
- No subject overlaps between gallery and probe *runs*; identity supervision is on resting runs only; the 90/10 validation split is drawn within resting data so probe-state windows never affect model selection. All 109 subjects appear in every split (the difficulty is the state shift, not unseen identities).
