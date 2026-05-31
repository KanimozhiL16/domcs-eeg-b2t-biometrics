# Environment & Hardware

## Training / evaluation hardware
- **GPUs:** 8 × NVIDIA A100 (Brev cloud instance).
- **Full ablation run:** 25 jobs (E1–E5 × 5 seeds) in parallel across the 8 GPUs, **~42.5 min** wall-clock.
- **OS:** Ubuntu 22.04. **CUDA:** 12.x. **Python:** 3.10.
- **Framework:** PyTorch ≥ 2.0.

## Determinism
- Each (variant, seed) job sets its own seed; the per-seed 90/10 validation split is drawn with `np.random.default_rng(seed)`.
- Best-validation checkpoint is saved per job (`model_best.pt`), carrying metadata: `activation:"ELU"`, `variant`, `seed`, `best_epoch`, `use_orth`, `use_arcface`, `lam_sup`. State dict key = `"model_state"`.
- Results are reported as mean ± std over 5 seeds (10 seeds for the R8 DOMCS-vs-DANN parity test).

## Model (exact)
- Total parameters **234,880** (inference-only path **201,856**). Activation **ELU** throughout.
- Encoder: 3 × [Conv1d(k=7/5/3, bias=False) → BatchNorm → ELU] with channels 64→64→128→256, then AdaptiveAvgPool1d(1) → f ∈ ℝ²⁵⁶.
- Identity branch: Linear(256→128, bias=False) → LayerNorm(128) → L2-normalize → z_id.
- State branch: **f.detach()** (stop-gradient) → Linear(256→128) → LayerNorm → L2-normalize → z_state.

## Hyperparameters (exact, as used by `ablation_worker_elu.py`)
| Param | Value |
|---|---|
| ArcFace scale s | 32 |
| ArcFace margin m | 0.50 |
| SupCon temperature τ | 0.07 |
| λ_state | 0.50 |
| λ_orth | 0.10 |
| λ_sup | 0.30 |
| Optimizer | Adam, lr 3e-4 (heads) / 1e-4 (encoder), weight decay 1e-4 |
| LR schedule | cosine → 1e-6 |
| Grad clip | 1.0 |
| Epochs | 60 |
| Batch size | 256 |
| Identity losses | applied on REST (R01–R02) windows only |

> Note: the manuscript Methodology must state ArcFace **s = 32** (the worker uses 32, not 30). λ_sup = 0.30. Orthogonality penalty = mean |cos(z_id, z_state)|.
