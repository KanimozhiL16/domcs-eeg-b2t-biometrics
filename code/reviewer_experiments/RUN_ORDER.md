# R8 + R9 — Two reviewer-hardening experiments (run on Brev)

Goal: convert the paper's weakest point (5 seeds, thin DANN margin, EEGNet-looks-broken)
into a solid one. No architecture changes, no new claims. ~35 min total on the 8xA100 node.

## Prerequisites (paths via env vars; defaults assume your Brev layout)
- DATA = /home/nvidia/24PHD1237/EEGMMIDB/EEGMMIDB_win2s_step1s_fs128.npz
- ROOT = /home/nvidia/24PHD1237/FRESH_EXP_20260528_DOMCS_EEG
- WORKER = $ROOT/ablation_worker_elu.py   (the VERIFIED DOMCS worker)
- Existing DOMCS E1 seeds 1-5 already in $ROOT/08_ablation_ELU_FINAL/E1/seed_{1..5}/

## Run order
1. **DOMCS seeds 6-10** (reuses the exact verified worker; seeds 1-5 untouched)
   `python R8a_DOMCS_seeds6to10.py`            (~10 min)
2. **DANN 10 seeds** (matched encoder + GRL state-adversary; clean baseline)
   `python R8b_DANN_10seeds.py`                (~15 min)
3. **Evaluate + statistics** (10 vs 10, exhaustive scorer; paired t + Wilcoxon + d + CI)
   `python R8c_eval_10seed_stats.py`           (~5 min)
   -> $ROOT/R8_10seed_stats/R8_10seed_stats.json + TABLE_stats_10seed.tex
4. **EEGNet same-task** (defuses the near-chance-baseline perception)
   `python R9_EEGNet_sametask.py`              (~5 min)
   -> $ROOT/R9_eegnet_sametask/R9_eegnet_sametask.json

## What each result does for the paper
- R8c: replaces the 5-seed stats with **10 vs 10**. At n=10 the Wilcoxon test can reach
  significance (its floor drops from 0.0625 at n=5 to <0.002 at n=10), directly answering
  "statistically fragile". Tighter bootstrap CI on the DOMCS mean too.
- R9: shows EEGNet reaches **high identification accuracy / low EER under same-task**, so its
  ~48% baseline-to-task EER is the protocol, not crippled tuning — defuses "looks extreme".

## After running
Paste the two JSON outputs (R8_10seed_stats.json, R9_eegnet_sametask.json) back to me and I
will fold the verified numbers into V3.1 (Table III stats + a one-line EEGNet same-task note),
and update the reviewer response. **All reported values stay traceable to these runs — nothing
is assumed.**

## Honest expectation
- DOMCS 10-seed mean should stay ~2.40% (same code, more seeds); CI tightens.
- DANN 10-seed is a freshly retrained matched baseline; its mean may differ slightly from the
  old 5-seed 2.88%. Report whatever it gives. If DOMCS<DANN holds in >=8/10 seeds, Wilcoxon
  will likely be significant; if the margin is genuinely thin, we report it honestly either way.
