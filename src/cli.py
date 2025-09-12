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

"""
Set default env vars to reduce CUDA memory fragmentation and silence tokenizers
parallelism warnings. Users can override these by setting their own env values.
Must be set before importing torch / creating CUDA context.
"""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from .model_registry import load_model, to_tokens
from .prompts import make_abab_prompts, make_name_prompts, find_copy_ops, split_copy_ops, lag_hist
from .runner import run_with_cache_all_layers
from .scoring import compute_scores, layer_profile
from .robustness import shuffled_antecedent, permute_heads_within_layer, bootstrap_peak
from .causal import measure_causal_delta
from .plots import heatmap, profile_curve, cross_size_overlay, nulls_plot, causal_bar, multi_overlay


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
    batch_size = int(settings.get("batch_size", 8))

    prompts = _gen_prompts(settings)
    if getattr(args, "print_prompts", False):
        print("\n".join(prompts))

    model = load_model(model_id, device=device, dtype=dtype)
    toks = to_tokens(model, prompts, seq_len=seq_len, prepend_bos=True)

    # Copy opportunities
    copy_ops = find_copy_ops(toks)

    # Compute scores
    memory_saver = bool(args.memory_saver)
    scores = compute_scores(model, toks, copy_ops, memory_saver=memory_saver, batch_size=batch_size)

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
    batch_size = int(settings.get("batch_size", 8))

    # Use same textual prompts for both
    prompts = _gen_prompts(settings)

    profs = []
    for model_id in models:
        model = load_model(model_id, device=device, dtype=dtype)
        toks = to_tokens(model, prompts, seq_len=seq_len, prepend_bos=True)
        copy_ops = find_copy_ops(toks)
        scores = compute_scores(model, toks, copy_ops, memory_saver=args.memory_saver, batch_size=batch_size)
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
        label_A=models[0],
        label_B=models[1],
    )
    print(f"Saved cross-size overlay to {os.path.join(figs_dir, 'cross_size_overlay.png')}")


def cmd_multisize(args: argparse.Namespace):
    """
    Compute per-layer profiles for N models using a single shared prompt set,
    then overlay curves on relative depth and save a combined figure and run JSON.
    """
    settings = _load_settings(os.path.join(ROOT, "settings.yaml"))
    _set_seeds(int(settings.get("seed", 123)))

    models: List[str] = list(args.models)
    if len(models) < 2:
        raise ValueError("--models must provide at least two model ids")

    # Optional display labels
    labels_in: List[str] | None = list(args.labels) if getattr(args, "labels", None) else None
    if labels_in is not None and len(labels_in) != len(models):
        raise ValueError("--labels length must match --models length")
    labels: List[str] = labels_in if labels_in is not None else models

    device = settings.get("device", "cuda")
    dtype = settings.get("dtype", "float16")
    seq_len = int(settings.get("seq_len", 256))
    batch_size = int(settings.get("batch_size", 8))

    # Generate one prompt set for all models
    prompts = _gen_prompts(settings)
    if getattr(args, "print_prompts", False):
        print("\n".join(prompts))

    curves: List[List[float]] = []
    n_layers_list: List[int] = []
    peak_rel_map: dict[str, float] = {}
    bandwidth_map: dict[str, float] = {}

    for model_id in models:
        model = load_model(model_id, device=device, dtype=dtype)
        toks = to_tokens(model, prompts, seq_len=seq_len, prepend_bos=True)
        copy_ops = find_copy_ops(toks)
        scores = compute_scores(model, toks, copy_ops, memory_saver=args.memory_saver, batch_size=batch_size)
        q = float(settings["evaluation"].get("high_score_quantile", 0.9))
        prof = layer_profile(scores, mode="topq", q=q)
        curves.append(prof["per_layer"])  # store per-layer curve
        n_layers_list.append(len(prof["per_layer"]))
        peak_rel_map[model_id] = float(prof.get("peak_rel", 0.0))
        bandwidth_map[model_id] = float(prof.get("bandwidth", 0.0))

    # Save combined run JSON and traceability artifacts
    run_dir = os.path.join(ROOT, "outputs", "runs", _now_stamp())
    _ensure_dir(run_dir)

    out = {
        "models": models,
        "labels": labels,
        "n_layers_per_model": n_layers_list,
        "peak_rel": peak_rel_map,
        "bandwidth": bandwidth_map,
        "profiles": {m: curves[i] for i, m in enumerate(models)},
    }
    _save_json(os.path.join(run_dir, "multisize.json"), out)
    # Save prompts and config
    if yaml is not None:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(settings, f)
    with open(os.path.join(run_dir, "prompts.txt"), "w") as f:
        f.write("\n".join(prompts))

    # Figure: multi-curve overlay on relative depth
    figs_dir = os.path.join(ROOT, "figs")
    _ensure_dir(figs_dir)
    fig_path = os.path.join(figs_dir, "cross_size_overlay_all.png")
    multi_overlay(curves, labels, fig_path, title="Cross-size overlay (all)", rel_depth=True)

    # Console summary: peak_rel per model and max delta
    peak_vals = [peak_rel_map[m] for m in models]
    if len(peak_vals) > 0:
        max_delta = max(peak_vals) - min(peak_vals)
    else:
        max_delta = 0.0
    parts = [f"{labels[i]}={peak_vals[i]:.3f}" for i in range(len(models))]
    print("peaks: " + ", ".join(parts) + f"; max Δ={max_delta:.3f}")
    print(f"Saved multisize outputs to: {run_dir}")
    print(f"Figure: {fig_path}")


def cmd_nulls(args: argparse.Namespace):
    settings = _load_settings(os.path.join(ROOT, "settings.yaml"))
    _set_seeds(int(settings.get("seed", 123)))

    model_id = args.model
    device = settings.get("device", "cuda")
    dtype = settings.get("dtype", "float16")
    seq_len = int(settings.get("seq_len", 256))
    batch_size = int(settings.get("batch_size", 8))
    q = float(settings["evaluation"].get("high_score_quantile", 0.9))

    prompts = _gen_prompts(settings)
    if getattr(args, "print_prompts", False):
        print("\n".join(prompts))
    model = load_model(model_id, device=device, dtype=dtype)
    toks = to_tokens(model, prompts, seq_len=seq_len, prepend_bos=True)
    copy_ops = find_copy_ops(toks)
    scores = compute_scores(model, toks, copy_ops, memory_saver=args.memory_saver, batch_size=batch_size)

    # Observed density (fraction of high heads per layer)
    thresh = float(np.quantile(scores.flatten(), q))
    observed_density = (scores > thresh).mean(axis=1).tolist()

    # Shuffled antecedent null
    null_ops = shuffled_antecedent(copy_ops, toks)
    scores_null = compute_scores(model, toks, null_ops, memory_saver=args.memory_saver, batch_size=batch_size)

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
    # Save the prompts for traceability
    with open(os.path.join(run_dir, "prompts.txt"), "w") as f:
        f.write("\n".join(prompts))

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
    batch_size = int(settings.get("batch_size", 8))
    top_k = int(args.topk or settings["evaluation"].get("causal_top_k", 3))
    rand_k = int(settings["evaluation"].get("causal_random_k", top_k))

    prompts = _gen_prompts(settings)
    model = load_model(model_id, device=device, dtype=dtype)
    toks = to_tokens(model, prompts, seq_len=seq_len, prepend_bos=True)
    copy_ops = find_copy_ops(toks)
    scores = compute_scores(model, toks, copy_ops, memory_saver=args.memory_saver, batch_size=batch_size)

    # Top-k heads by global score
    L, H = scores.shape
    flat_idx = np.argsort(scores.flatten())[::-1]
    top_pairs = []
    for idx in flat_idx[:top_k]:
        l = int(idx // H)
        h = int(idx % H)
        top_pairs.append((l, h))

    deltas = measure_causal_delta(model, toks, copy_ops, top_pairs, k_random=rand_k, batch_size=batch_size)

    run_dir = os.path.join(ROOT, "outputs", "runs", _now_stamp())
    _ensure_dir(run_dir)
    _save_json(os.path.join(run_dir, f"causal_{model_id}.json"), deltas)

    figs_dir = os.path.join(ROOT, "figs")
    _ensure_dir(figs_dir)
    causal_bar(deltas["delta_top"], deltas["delta_rand"], os.path.join(figs_dir, "causal_delta.png"), title=f"Causal delta: {model_id}")

    print(f"Saved causal outputs to: {run_dir}")
    print(f"Causal figure in: {figs_dir}")


def cmd_lagbuckets(args: argparse.Namespace):
    settings = _load_settings(os.path.join(ROOT, "settings.yaml"))
    _set_seeds(int(settings.get("seed", 123)))
    
    model_id = args.model
    device = settings.get("device", "cuda")
    dtype = settings.get("dtype", "float16")
    seq_len = int(settings.get("seq_len", 256))
    batch_size = int(settings.get("batch_size", 8))
    short_max = int(args.short_max or 8)
    q = float(settings["evaluation"].get("high_score_quantile", 0.9))

    # Prompts
    prompts = _gen_prompts(settings)
    if getattr(args, "print_prompts", False):
        print("\n".join(prompts))

    # Model + tokens
    model = load_model(model_id, device=device, dtype=dtype)
    toks = to_tokens(model, prompts, seq_len=seq_len, prepend_bos=True)

    # Copy ops and buckets
    ops_all = find_copy_ops(toks)
    ops_short, ops_long = split_copy_ops(ops_all, short_max=short_max)
    hist = lag_hist(ops_all)

    # Compute scores per bucket
    scores_short = compute_scores(model, toks, ops_short, memory_saver=args.memory_saver, batch_size=batch_size)
    scores_long = compute_scores(model, toks, ops_long, memory_saver=args.memory_saver, batch_size=batch_size)

    prof_short = layer_profile(scores_short, mode="topq", q=q)
    prof_long = layer_profile(scores_long, mode="topq", q=q)

    if len(ops_long) == 0:
        print(f"Warning: no long-lag ops found (> {short_max}). Long profile will be near-zero.")
    if len(ops_short) == 0:
        print(f"Warning: no short-lag ops found (<= {short_max}). Short profile will be near-zero.")

    # Outputs
    run_dir = os.path.join(ROOT, "outputs", "runs", _now_stamp())
    _ensure_dir(run_dir)
    out = {
        "model_id": model_id,
        "short_max": short_max,
        "lag_hist_all": hist,
        "n_ops_all": len(ops_all),
        "n_ops_short": len(ops_short),
        "n_ops_long": len(ops_long),
        "profile_short": prof_short,
        "profile_long": prof_long,
    }
    _save_json(os.path.join(run_dir, f"lagbuckets_{model_id}.json"), out)
    # save prompts and config for traceability
    if yaml is not None:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(settings, f)
    with open(os.path.join(run_dir, "prompts.txt"), "w") as f:
        f.write("\n".join(prompts))

    # Figure overlay
    figs_dir = os.path.join(ROOT, "figs")
    _ensure_dir(figs_dir)
    title = f"Lag overlay (≤{short_max} vs >{short_max}): {model_id}"
    out_fig = os.path.join(figs_dir, f"lag_overlay_{model_id}.png")
    cross_size_overlay(
        prof_short["per_layer"],
        prof_long["per_layer"],
        out_fig,
        title,
        rel_depth=True,
        label_A=f"Short (≤{short_max})",
        label_B=f"Long (>{short_max})",
    )

    # Console summary
    ps, pl = prof_short, prof_long
    print(f"Saved lag-bucket outputs to: {run_dir}")
    print(f"Figure in: {out_fig}")
    print(
        "Short: peak_layer=%d (rel=%.3f), bw=%.3f | Long: peak_layer=%d (rel=%.3f), bw=%.3f"
        % (
            ps.get("peak_layer", -1), ps.get("peak_rel", 0.0), ps.get("bandwidth", 0.0),
            pl.get("peak_layer", -1), pl.get("peak_rel", 0.0), pl.get("bandwidth", 0.0),
        )
    )


def _scan_profile_runs(runs_dir: str) -> list[dict]:
    rows: list[dict] = []
    for root, dirs, files in os.walk(runs_dir):
        for fn in files:
            if not fn.startswith("profile_") or not fn.endswith(".json"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, "r") as f:
                    prof = json.load(f)
            except Exception:
                continue
            # derive model from filename: profile_<model>.json
            model_id = fn[len("profile_") : -len(".json")]
            # try metadata
            meta = {}
            try:
                meta_path = os.path.join(root, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as mf:
                        meta = json.load(mf)
            except Exception:
                meta = {}
            # run timestamp is the last directory name under outputs/runs
            parts = os.path.normpath(root).split(os.sep)
            ts = parts[-1] if parts else ""
            rows.append(
                {
                    "ts": ts,
                    "model": str(meta.get("model_id", model_id)),
                    "n_layers": int(meta.get("n_layers", 0)),
                    "n_heads": int(meta.get("n_heads", 0)),
                    "seq_len": int(meta.get("seq_len", 0)),
                    "n_prompts": int(meta.get("n_prompts", 0)),
                    "n_copy_ops": int(meta.get("n_copy_ops", 0)),
                    "mode": str(prof.get("mode", "topq")),
                    "q": float(prof.get("q", 0.9)),
                    "peak_layer": int(prof.get("peak_layer", -1)),
                    "peak_rel": float(prof.get("peak_rel", 0.0)),
                    "bandwidth": float(prof.get("bandwidth", 0.0)),
                    "path": path,
                }
            )
    return rows


def _print_table(rows: list[dict], columns: list[tuple[str, str]]):
    # columns: list of (key, header)
    # compute widths
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    headers = [hdr for (_k, hdr) in columns]
    data = [[fmt(row.get(k, "")) for (k, _hdr) in columns] for row in rows]
    widths = [len(h) for h in headers]
    for r in data:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def print_row(vals):
        parts = []
        for i, v in enumerate(vals):
            parts.append(str(v).ljust(widths[i]))
        print(" | ".join(parts))

    # header
    print_row(headers)
    print("-+-".join("-" * w for w in widths))
    for r in data:
        print_row(r)


def cmd_profiles(args: argparse.Namespace):
    runs_dir = os.path.join(ROOT, "outputs", "runs")
    rows = _scan_profile_runs(runs_dir)
    if not rows:
        print(f"No profile_*.json found in {runs_dir}")
        return
    # sort by model then timestamp
    rows.sort(key=lambda r: (r.get("model", ""), r.get("ts", "")))

    if args.latest:
        latest = {}
        for r in rows:
            latest[r["model"]] = r  # overwritten to keep last by ts order
        rows_to_show = [latest[m] for m in sorted(latest.keys())]
        print("Latest profile per model:\n")
    else:
        rows_to_show = rows
        print("All discovered profiles:\n")

    cols = [
        ("ts", "Run"),
        ("model", "Model"),
        ("n_layers", "L"),
        ("n_heads", "H"),
        ("seq_len", "T"),
        ("n_prompts", "Prompts"),
        ("peak_layer", "PeakL"),
        ("peak_rel", "PeakRel"),
        ("bandwidth", "BW"),
        ("q", "q"),
    ]
    _print_table(rows_to_show, cols)
    print("\nFiles:")
    for r in rows_to_show:
        print(f"- {r['model']} @ {r['ts']}: {r['path']}")

def main():
    parser = argparse.ArgumentParser(description="Induction topology experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_profile = sub.add_parser("profile", help="Compute head scores and per-layer profile")
    p_profile.add_argument("--model", required=True, help="Model id (e.g., pythia-410m)")
    p_profile.add_argument("--memory_saver", action="store_true", help="Cache per-layer to avoid OOM")
    p_profile.add_argument("--print_prompts", action="store_true", help="Print generated prompts to stdout")
    p_profile.set_defaults(func=cmd_profile)

    p_cross = sub.add_parser("crosssize", help="Cross-size overlay for two models")
    p_cross.add_argument("--models", nargs=2, required=True, help="Two model ids")
    p_cross.add_argument("--memory_saver", action="store_true")
    p_cross.set_defaults(func=cmd_crosssize)

    p_nulls = sub.add_parser("nulls", help="Compute null comparisons and bootstrap")
    p_nulls.add_argument("--model", required=True)
    p_nulls.add_argument("--memory_saver", action="store_true")
    p_nulls.add_argument("--print_prompts", action="store_true", help="Print generated prompts to stdout")
    p_nulls.set_defaults(func=cmd_nulls)

    p_causal = sub.add_parser("causal", help="Small causal ablation spot-check")
    p_causal.add_argument("--model", required=True)
    p_causal.add_argument("--topk", type=int, default=None)
    p_causal.add_argument("--memory_saver", action="store_true")
    p_causal.set_defaults(func=cmd_causal)

    p_lag = sub.add_parser("lagbuckets", help="Overlay per-layer profiles for short vs long lags")
    p_lag.add_argument("--model", required=True)
    p_lag.add_argument("--short_max", type=int, default=8)
    p_lag.add_argument("--memory_saver", action="store_true")
    p_lag.add_argument("--print_prompts", action="store_true", help="Print generated prompts to stdout")
    p_lag.set_defaults(func=cmd_lagbuckets)

    p_profiles = sub.add_parser("profiles", help="Summarize saved profile_* JSONs as console tables")
    p_profiles.add_argument("--latest", action="store_true", help="Show only the latest run per model")
    p_profiles.set_defaults(func=cmd_profiles)

    p_multi = sub.add_parser("multisize", help="Overlay per-layer profiles for N models with shared prompts")
    p_multi.add_argument("--models", nargs="+", required=True, help="One or more model ids")
    p_multi.add_argument("--labels", nargs="+", default=None, help="Optional pretty labels for legend")
    p_multi.add_argument("--memory_saver", action="store_true")
    p_multi.add_argument("--print_prompts", action="store_true", help="Print generated prompts to stdout")
    p_multi.set_defaults(func=cmd_multisize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
