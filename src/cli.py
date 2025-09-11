from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np

try:
    import yaml
except Exception:
    yaml = None  # type: ignore

import torch

from .model_registry import load_model, to_tokens
from .prompts import make_abab_prompts, make_name_prompts, find_copy_ops
from .runner import run_with_cache_all_layers
from .scoring import compute_scores, layer_profile
from .robustness import shuffled_antecedent, permute_heads_within_layer, bootstrap_peak
from .causal import measure_causal_delta
from .plots import heatmap, profile_curve, cross_size_overlay, nulls_plot, causal_bar


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_settings(path: str) -> dict:
    if yaml is None:
        raise RuntimeError("pyyaml not installed; cannot load settings.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _set_seeds(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _gen_prompts(cfg: dict) -> List[str]:
    n = int(cfg["prompts"]["n_sequences"])
    seq_len = int(cfg["seq_len"])
    lag_values = list(cfg["prompts"]["lag_values"]) or [1, 2, 4]
    mix = str(cfg["prompts"].get("mix", "uniform"))
    task_types = list(cfg["prompts"].get("task_types", ["ABAB", "NAMES"]))

    n_abab = n // 2 if "ABAB" in task_types else 0
    n_names = n - n_abab if "NAMES" in task_types else 0

    ababs = make_abab_prompts(n_abab, seq_len, lag_values, mix)
    names = make_name_prompts(n_names, seq_len, lag_values, mix)
    prompts = ababs + names
    return prompts


def _save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def cmd_profile(args: argparse.Namespace):
    # Load settings from repo root
    settings = _load_settings(os.path.join(ROOT, "settings.yaml"))
    _set_seeds(int(settings.get("seed", 123)))

    model_id = args.model
    device = settings.get("device", "cuda")
    dtype = settings.get("dtype", "float16")
    seq_len = int(settings.get("seq_len", 256))

    prompts = _gen_prompts(settings)

    model = load_model(model_id, device=device, dtype=dtype)
    toks = to_tokens(model, prompts, seq_len=seq_len, prepend_bos=True)

    # Copy opportunities
    copy_ops = find_copy_ops(toks)

    # Compute scores
    memory_saver = bool(args.memory_saver)
    scores = compute_scores(model, toks, copy_ops, memory_saver=memory_saver)

    # Profile
    q = float(settings["evaluation"].get("high_score_quantile", 0.9))
    prof = layer_profile(scores, mode="topq", q=q)

    # Outputs
    run_dir = os.path.join(ROOT, "outputs", "runs", _now_stamp())
    _ensure_dir(run_dir)
    np.save(os.path.join(run_dir, f"scores_{model_id}.npy"), scores)
    _save_json(os.path.join(run_dir, f"profile_{model_id}.json"), prof)

    # Save settings and metadata
    if yaml is not None:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(settings, f)
    meta = {
        "model_id": model_id,
        "n_layers": int(getattr(model.cfg, "n_layers")),  # type: ignore[attr-defined]
        "n_heads": int(getattr(model.cfg, "n_heads")),    # type: ignore[attr-defined]
        "seq_len": seq_len,
        "n_prompts": len(prompts),
        "n_copy_ops": len(copy_ops),
    }
    _save_json(os.path.join(run_dir, "metadata.json"), meta)
    with open(os.path.join(run_dir, "prompts.txt"), "w") as f:
        f.write("\n".join(prompts))

    # Figures
    figs_dir = os.path.join(ROOT, "figs")
    _ensure_dir(figs_dir)
    heatmap(scores, os.path.join(figs_dir, f"heatmap_{model_id}.png"), title=f"Head-level induction: {model_id}")
    profile_curve(
        prof["per_layer"],
        os.path.join(figs_dir, f"profile_{model_id}.png"),
        title=f"Per-layer profile: {model_id}",
        annotate_peak=True,
    )

    print(f"Saved run outputs to: {run_dir}")
    print(f"Figures in: {figs_dir}")


def cmd_crosssize(args: argparse.Namespace):
    settings = _load_settings(os.path.join(ROOT, "settings.yaml"))
    _set_seeds(int(settings.get("seed", 123)))

    models = args.models
    if len(models) != 2:
        raise ValueError("--models must provide exactly two model ids")
    device = settings.get("device", "cuda")
    dtype = settings.get("dtype", "float16")
    seq_len = int(settings.get("seq_len", 256))

    # Use same textual prompts for both
    prompts = _gen_prompts(settings)

    profs = []
    for model_id in models:
        model = load_model(model_id, device=device, dtype=dtype)
        toks = to_tokens(model, prompts, seq_len=seq_len, prepend_bos=True)
        copy_ops = find_copy_ops(toks)
        scores = compute_scores(model, toks, copy_ops, memory_saver=args.memory_saver)
        q = float(settings["evaluation"].get("high_score_quantile", 0.9))
        prof = layer_profile(scores, mode="topq", q=q)
        profs.append(prof["per_layer"])  # store per-layer curves

    figs_dir = os.path.join(ROOT, "figs")
    _ensure_dir(figs_dir)
    cross_size_overlay(
        profs[0],
        profs[1],
        os.path.join(figs_dir, "cross_size_overlay.png"),
        title=f"Cross-size overlay: {models[0]} vs {models[1]}",
        rel_depth=True,
    )
    print(f"Saved cross-size overlay to {os.path.join(figs_dir, 'cross_size_overlay.png')}")


def cmd_nulls(args: argparse.Namespace):
    settings = _load_settings(os.path.join(ROOT, "settings.yaml"))
    _set_seeds(int(settings.get("seed", 123)))

    model_id = args.model
    device = settings.get("device", "cuda")
    dtype = settings.get("dtype", "float16")
    seq_len = int(settings.get("seq_len", 256))
    q = float(settings["evaluation"].get("high_score_quantile", 0.9))

    prompts = _gen_prompts(settings)
    model = load_model(model_id, device=device, dtype=dtype)
    toks = to_tokens(model, prompts, seq_len=seq_len, prepend_bos=True)
    copy_ops = find_copy_ops(toks)
    scores = compute_scores(model, toks, copy_ops, memory_saver=args.memory_saver)

    # Observed density (fraction of high heads per layer)
    thresh = float(np.quantile(scores.flatten(), q))
    observed_density = (scores > thresh).mean(axis=1).tolist()

    # Shuffled antecedent null
    null_ops = shuffled_antecedent(copy_ops, toks)
    scores_null = compute_scores(model, toks, null_ops, memory_saver=args.memory_saver)

    # Head permutation null summary
    perm_summary = permute_heads_within_layer(scores, reps=200, q=q)

    # Bootstrap on profile
    prof = layer_profile(scores, mode="topq", q=q)
    boot = bootstrap_peak(prof["per_layer"], n_resamples=int(settings["evaluation"].get("bootstrap_resamples", 1000)))

    run_dir = os.path.join(ROOT, "outputs", "runs", _now_stamp())
    _ensure_dir(run_dir)
    out = {
        "observed_density": observed_density,
        "q": q,
        "thresh": thresh,
        "permute_summary": perm_summary,
        "bootstrap": boot,
    }
    _save_json(os.path.join(run_dir, f"nulls_{model_id}.json"), out)

    figs_dir = os.path.join(ROOT, "figs")
    _ensure_dir(figs_dir)
    nulls_plot(observed_density, perm_summary, os.path.join(figs_dir, f"nulls_{model_id}.png"), title=f"Null comparison: {model_id}")

    print(f"Saved nulls outputs to: {run_dir}")
    print(f"Nulls figure in: {figs_dir}")


def cmd_causal(args: argparse.Namespace):
    settings = _load_settings(os.path.join(ROOT, "settings.yaml"))
    _set_seeds(int(settings.get("seed", 123)))

    model_id = args.model
    device = settings.get("device", "cuda")
    dtype = settings.get("dtype", "float16")
    seq_len = int(settings.get("seq_len", 256))
    top_k = int(args.topk or settings["evaluation"].get("causal_top_k", 3))
    rand_k = int(settings["evaluation"].get("causal_random_k", top_k))

    prompts = _gen_prompts(settings)
    model = load_model(model_id, device=device, dtype=dtype)
    toks = to_tokens(model, prompts, seq_len=seq_len, prepend_bos=True)
    copy_ops = find_copy_ops(toks)
    scores = compute_scores(model, toks, copy_ops, memory_saver=args.memory_saver)

    # Top-k heads by global score
    L, H = scores.shape
    flat_idx = np.argsort(scores.flatten())[::-1]
    top_pairs = []
    for idx in flat_idx[:top_k]:
        l = int(idx // H)
        h = int(idx % H)
        top_pairs.append((l, h))

    deltas = measure_causal_delta(model, toks, copy_ops, top_pairs, k_random=rand_k)

    run_dir = os.path.join(ROOT, "outputs", "runs", _now_stamp())
    _ensure_dir(run_dir)
    _save_json(os.path.join(run_dir, f"causal_{model_id}.json"), deltas)

    figs_dir = os.path.join(ROOT, "figs")
    _ensure_dir(figs_dir)
    causal_bar(deltas["delta_top"], deltas["delta_rand"], os.path.join(figs_dir, "causal_delta.png"), title=f"Causal delta: {model_id}")

    print(f"Saved causal outputs to: {run_dir}")
    print(f"Causal figure in: {figs_dir}")


def main():
    parser = argparse.ArgumentParser(description="Induction topology experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_profile = sub.add_parser("profile", help="Compute head scores and per-layer profile")
    p_profile.add_argument("--model", required=True, help="Model id (e.g., pythia-410m)")
    p_profile.add_argument("--memory_saver", action="store_true", help="Cache per-layer to avoid OOM")
    p_profile.set_defaults(func=cmd_profile)

    p_cross = sub.add_parser("crosssize", help="Cross-size overlay for two models")
    p_cross.add_argument("--models", nargs=2, required=True, help="Two model ids")
    p_cross.add_argument("--memory_saver", action="store_true")
    p_cross.set_defaults(func=cmd_crosssize)

    p_nulls = sub.add_parser("nulls", help="Compute null comparisons and bootstrap")
    p_nulls.add_argument("--model", required=True)
    p_nulls.add_argument("--memory_saver", action="store_true")
    p_nulls.set_defaults(func=cmd_nulls)

    p_causal = sub.add_parser("causal", help="Small causal ablation spot-check")
    p_causal.add_argument("--model", required=True)
    p_causal.add_argument("--topk", type=int, default=None)
    p_causal.add_argument("--memory_saver", action="store_true")
    p_causal.set_defaults(func=cmd_causal)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
