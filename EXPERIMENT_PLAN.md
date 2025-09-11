# Induction Heads Topology: Experiment Plan (MATS 9.0)

## Overview
- **Primary question:** In decoder-only LMs, do induction heads (copy specialists) concentrate at a stable relative depth (and with a characteristic bandwidth) across sizes within a family?
- **Models:** Start with `pythia-410m` and `pythia-1b` (same tokenizer/family). Optionally sanity-check on `gpt2` later.
- **Scope:** Inference-only; collect attention patterns; compute head-level induction scores; compare per-layer profiles across sizes; evaluate robustness nulls and a small causal ablation.

## Repo Status
- **Already implemented:**
  - `src/prompts.py`: Synthetic ABAB and repeated-name prompts, plus `find_copy_ops` (extract matching-token (b,t,s) pairs).
  - `src/model_registry.py`: Load models via TransformerLens and tokenize to fixed `seq_len`.
  - `src/runner.py`: Efficient caching of attention (`hook_pattern`) across all layers or per-layer (memory saver).
  - `src/scoring.py`: Head induction scores and per-layer profile (mean or top-quantile).
  - `src/robustness.py`: Shuffled-antecedent null, head-permutation null, and bootstrap for peak/bandwidth.
  - `src/causal.py`: Head ablation context and copy-accuracy deltas (top-k vs random-k heads).
  - `src/plots.py`: Heatmap, per-layer profile, cross-size overlay, nulls band, and causal bar chart.
  - `src/cli.py`: Commands `profile`, `crosssize`, `nulls`, `causal` (fixed to use repo-root paths).
- **Outputs:**
  - Figures saved to `figs/`.
  - Run artifacts (scores, metadata, JSON summaries) saved under `outputs/runs/<timestamp>/`.

-## Setup
- **Python:** 3.10+ recommended. You already have a `.venv` with `uv` — great.
- **Install dependencies:**
  1) Install a PyTorch build compatible with your CUDA version (or CPU): see https://pytorch.org/get-started/locally/
  2) Install remaining packages with uv (PyTorch excluded):
     - `uv pip install -r requirements.txt`
     - Or, after adding `pyproject.toml`, use: `uv sync`
  3) First model load downloads weights from Hugging Face into the cache.
- **GPU/VRAM guidance:**
  - For `pythia-410m`, `fp16`, `seq_len=256–512`, batch 8–16 fits ~14 GB.
  - Use `--memory_saver` to cache one layer at a time to avoid OOM at cost of more wall time.
  - Set `device: "cpu"` in `settings.yaml` if no GPU is available (slower).

## Configuration (`settings.yaml`)
- **models:** list of model ids; use `pythia-410m` and `pythia-1b`.
- **device/dtype:** e.g., `cuda` and `float16`.
- **seq_len:** tokenized sequence length cutoff/pad.
- **prompts:**
  - `n_sequences`: number of prompts total (split across ABAB and NAMES).
  - `lag_values`: list of target lags (e.g., `[1,2,4,8,16,32]`).
  - `task_types`: choose `ABAB`, `NAMES`, or both.
- **evaluation:**
  - `high_score_quantile`: top-q head averaging within layer (e.g., 0.9).
  - `bootstrap_resamples`: number of bootstrap draws for peak/bandwidth.
  - `causal_top_k`/`causal_random_k`: number of heads to ablate.

## Run Plan (T1–T5)

### T1 (3–4h): 410M baseline — heatmap + layer profile
- **Command (uv):**
  - `uv run induct profile --model pythia-410m --memory_saver`
- **Alternative:**
  - `uv run python -m src.cli profile --model pythia-410m --memory_saver`
- **Outputs:**
  - Scores: `outputs/runs/<ts>/scores_pythia-410m.npy`
  - Profile JSON: `outputs/runs/<ts>/profile_pythia-410m.json`
  - Figures: `figs/heatmap_pythia-410m.png`, `figs/profile_pythia-410m.png`
- **Checks:**
  - Head heatmap shows a concentration band at mid–late layers.
  - Profile curve’s peak layer and bandwidth are reported in the JSON.

### T2 (2–3h): Cross-size overlay (410M vs 1B)
- **Command (uv):**
  - `uv run induct crosssize --models pythia-410m pythia-1b --memory_saver`
- **Alternative:**
  - `uv run python -m src.cli crosssize --models pythia-410m pythia-1b --memory_saver`
- **Outputs:**
  - Figure: `figs/cross_size_overlay.png` (relative depth x-axis).
- **Checks:**
  - Curves align in relative depth; 1B may have a slightly sharper band.

### T3 (2–3h): Nulls + Bootstrap CIs
- **Command (uv):**
  - `uv run induct nulls --model pythia-410m --memory_saver`
- **Alternative:**
  - `uv run python -m src.cli nulls --model pythia-410m --memory_saver`
- **Outputs:**
  - JSON: `outputs/runs/<ts>/nulls_pythia-410m.json` (permute-head summary, bootstrap CIs).
  - Figure: `figs/nulls_pythia-410m.png` (observed density vs permutation-null band).
- **Checks:**
  - Observed density deviates above null mean/band around the induction band.
  - Bootstrap CIs for peak relative depth/bandwidth are meaningful (not degenerate).

### T4 (1–2h): Causal spot-check (top-k vs random)
- **Command (uv):**
  - `uv run induct causal --model pythia-410m --topk 3 --memory_saver`
- **Alternative:**
  - `uv run python -m src.cli causal --model pythia-410m --topk 3 --memory_saver`
- **Outputs:**
  - JSON: `outputs/runs/<ts>/causal_pythia-410m.json` (baseline/top/random acc, deltas, sampled random heads).
  - Figure: `figs/causal_delta.png` (Δ copy-accuracy bars).
- **Checks:**
  - `Δ_top` more negative than `Δ_rand` (top heads ablation hurts copying more).

### T5 (2–3h): Finalize plots + write-up blocks
- **Collect:** the five figures: heatmap, profile, cross-size overlay, nulls, causal bar.
- **Write-up:** methods, results, limitations, seeds, hardware, and next steps.

## Optional Extension (Lag Gradient)
- **Idea:** Split `copy_ops` by lag (e.g., short ≤ 8, long > 8) and recompute per-layer profiles; longer lags often peak later.
- **Minimal change:** filter `copy_ops` in-memory by `(t - s)` and call `compute_scores` on the same logits/cache (or re-run per split if needed).
- **Plot:** overlay short vs long lag profiles on relative depth.

## Practical Notes & Risks
- **VRAM pressure:** prefer `--memory_saver` for `pythia-1b` or when using longer `seq_len`/larger batch.
- **Tokenization sanity:** prompts are space-separated tokens to increase single-token likelihood; we rely on `find_copy_ops` to verify exact token ID matches.
- **Same prompts across models:** `crosssize` uses identical text prompts across both models to avoid distribution confounds.
- **Relative depth:** compare by `layer_index / n_layers` for cross-size invariants; avoid comparing head indices across models.

## Is this experiment feasible with this repo?
- **Yes.** This repo already contains the full pipeline described in the plan:
  - Prompt generation, attention caching, head-level scoring, per-layer profiling, null controls, bootstrap CIs, and a causal ablation.
  - The CLI provides end-to-end commands to produce all five plots and JSON artifacts.
  - I fixed path handling in `src/cli.py` so it uses this repo’s root (`settings.yaml`, `figs/`, `outputs/`).
- **What remains:** environment setup (install dependencies) and running the commands above. Optionally extend with the lag gradient split.

## Quickstart Commands (uv)
1) Install deps (after installing a matching PyTorch): `uv pip install -r requirements.txt` (or `uv sync` with pyproject)
2) 410M profile: `uv run induct profile --model pythia-410m --memory_saver`
3) Cross-size: `uv run induct crosssize --models pythia-410m pythia-1b --memory_saver`
4) Nulls: `uv run induct nulls --model pythia-410m --memory_saver`
5) Causal: `uv run induct causal --model pythia-410m --topk 3 --memory_saver`

## Deliverables Checklist
- Heatmap (410M): `figs/heatmap_pythia-410m.png`
- Profile (410M): `figs/profile_pythia-410m.png`
- Cross-size overlay: `figs/cross_size_overlay.png`
- Nulls comparison: `figs/nulls_pythia-410m.png`
- Causal delta bars: `figs/causal_delta.png`
- JSON summaries in `outputs/runs/<timestamp>/`
