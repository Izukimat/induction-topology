# Induction Topology — Reproducibility Guide

This repository contains code to measure and compare where “induction heads” concentrate within decoder-only language models, and how this concentration changes with model size. It generates per-layer profiles, cross-size overlays, null/robustness checks, and a small causal ablation.

- Figures are written to `figs/`.
- JSON/array artifacts are saved under `outputs/runs/<timestamp>/`.
- A command-line tool (`induct`) drives the full pipeline.


## Quick Start

- Python: 3.10+
- OS: Linux/macOS (GPU strongly recommended)
- GPU: 12–24 GB VRAM recommended for Pythia models at default settings. Use `--memory_saver` if you hit OOM.

Option A — pip/venv

```
python -m venv .venv
source .venv/bin/activate
# Install a matching PyTorch build for your system (CUDA/CPU): https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121  # example; choose the right one
# Install this project (provides the `induct` CLI)
pip install -e .
```

Option B — uv (optional)

```
# If you use uv, sync dependencies from pyproject
uv sync
# Install PyTorch separately if needed (see PyTorch site for correct command)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
# Run the CLI through uv
uv run induct --help
```

On first run, model weights download from Hugging Face. Set `HF_HOME` if you want a custom cache location.


## Configuration

All defaults live in `settings.yaml` (edit to change model/device/sequence settings):

- `device`: `cuda` (GPU) or `cpu`
- `dtype`: `float16` (on GPU), `float32` is safer on CPU
- `seq_len`: tokenized sequence length cutoff/pad
- `batch_size`: micro-batch size used by scoring routines
- `seed`: global RNG seed for reproducibility
- `prompts`: prompt count, lag distribution, and task types (`ABAB`, `NAMES`)
- `evaluation`: quantile for top-head averaging, bootstrap resamples, and causal ablation sizes

You can always override prompts per run via flags like `--print_prompts` to inspect the generated text.


## CLI Overview

All commands are available via:

- Installed script: `induct <subcommand> [flags]`
- Module mode: `python -m src.cli <subcommand> [flags]`

Subcommands (see `src/cli.py:569`):

- `profile`: Compute head scores and per-layer profile for one model
- `crosssize`: Overlay profiles for two models (shared prompts)
- `multisize`: Overlay profiles for N models (shared prompts)
- `nulls`: Null comparisons (shuffled antecedent, head-permutation) + bootstrap CIs
- `causal`: Causal spot-check by ablating top-k heads vs random heads
- `lagbuckets`: Compare short vs long lag buckets within one model
- `profiles`: Summarize saved `profile_*.json` runs as a console table

Useful global flags:

- `--memory_saver`: Reduce VRAM by caching one layer at a time
- `--print_prompts`: Print the exact prompt texts used for a run


## Reproduce the Paper’s Results

The steps below reproduce the figures and statistics used in the write-up. They assume the defaults in `settings.yaml`.

1) Single-model profile (heatmap + per-layer curve)

```
induct profile --model pythia-410m --memory_saver
```
- Figures: `figs/heatmap_pythia-410m.png`, `figs/profile_pythia-410m.png`
- Artifacts: `outputs/runs/<ts>/scores_pythia-410m.npy`, `.../profile_pythia-410m.json`

2) Cross-size overlay (relative depth)

```
induct crosssize --models pythia-410m pythia-1b --memory_saver
```
- Figure: `figs/cross_size_overlay.png`
- Uses identical textual prompts for both models to avoid distribution drift.

3) Nulls + bootstrap confidence intervals

```
induct nulls --model pythia-410m --memory_saver
```
- Figure: `figs/nulls_pythia-410m.png`
- JSON: `outputs/runs/<ts>/nulls_pythia-410m.json` (permute-head summary, bootstrap CIs)

4) Causal ablation spot-check

```
induct causal --model pythia-410m --topk 3 --memory_saver
```
- Figure: `figs/causal_delta.png`
- JSON: `outputs/runs/<ts>/causal_pythia-410m.json` with baseline/top/random accuracy and deltas

5) Optional: Long vs short lags

```
induct lagbuckets --model pythia-1b --short_max 8 --memory_saver
```
- Figure: `figs/lag_overlay_pythia-1b.png`
- JSON: `outputs/runs/<ts>/lagbuckets_pythia-1b.json`

6) Summarize profiles you’ve run

```
induct profiles --latest
```
- Prints a table of latest `profile_*.json` per model with peak layer, relative depth, and bandwidth.


## What Each Command Computes

- Prompts (`src/prompts.py:1`): Synthetic ABAB and repeated-name streams with controllable token lags; `find_copy_ops` finds exact-token matches and their source positions.
- Scoring (`src/scoring.py:1`): For each layer and head, average attention weight placed on the matching antecedent positions; returns a `[L, H]` matrix of “copy scores”.
- Profile (`src/scoring.py:124`): Reduce to a per-layer curve by mean or top-quantile mean within each layer; report peak layer, peak relative depth, and bandwidth (central 68% mass).
- Nulls/robustness (`src/robustness.py:1`):
  - Shuffled antecedent: replace source positions with random earlier positions to break true copies.
  - Head permutation: sample head scores across layers to estimate a layer-agnostic null density band.
  - Bootstrap: resample layer contributions to quantify uncertainty in peak/bandwidth.
- Causal (`src/causal.py:1`): Temporarily zero value streams for selected heads during forward and measure Δ in copy accuracy at copy positions; compare top-k vs random-k.
- Execution/caching (`src/runner.py:1`): Efficient attention caching across all layers or one layer at a time to manage memory.
- Plots (`src/plots.py:1`): Heatmaps, layer curves, overlays, null bands, and bar charts saved as PNGs.


## Outputs and Traceability

- Figures: `figs/*.png`
- Run artifacts: `outputs/runs/<timestamp>/` containing:
  - Scores arrays (`scores_<model>.npy`), profile JSONs (`profile_<model>.json`)
  - For multi-model runs, a combined `multisize.json`
  - Causal and nulls JSON summaries
  - `config.yaml` and `prompts.txt` for exact run reproduction

Use `induct profiles --latest` to list the latest per-model runs and their file paths.


## Tips & Troubleshooting

- Out of memory (OOM):
  - Add `--memory_saver` to all commands (one-layer-at-a-time caching)
  - Lower `batch_size` in `settings.yaml`
  - Lower `seq_len` in `settings.yaml`
  - Prefer `dtype: float16` on GPU; use `float32` on CPU
- Speed: GPU strongly recommended; CPU works but is slow
- Tokenizer warnings and CUDA fragmentations are mitigated automatically in the CLI by environment defaults
- Model download: ensure internet access on first run; set `HF_HOME` to move the cache
