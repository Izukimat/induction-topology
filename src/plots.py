from __future__ import annotations

import os
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def heatmap(scores: np.ndarray, out_path: str, title: str):
    _ensure_dir(out_path)
    plt.figure(figsize=(8, 5))
    plt.imshow(scores, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(label="Induction score")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def profile_curve(per_layer: List[float], out_path: str, title: str, annotate_peak: bool = True):
    _ensure_dir(out_path)
    y = np.array(per_layer)
    x = np.arange(len(y))
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker="o", lw=1.5)
    if annotate_peak and len(y) > 0:
        p = int(np.argmax(y))
        plt.axvline(p, color="red", ls="--", lw=1, alpha=0.7)
        plt.text(p, y[p], f" peak={p}", color="red", va="bottom")
    plt.xlabel("Layer index")
    plt.ylabel("Layer profile (mean)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def cross_size_overlay(
    profile_A: List[float],
    profile_B: List[float],
    out_path: str,
    title: str,
    rel_depth: bool = True,
    label_A: str = "Model A",
    label_B: str = "Model B",
):
    _ensure_dir(out_path)
    yA = np.array(profile_A)
    yB = np.array(profile_B)
    if rel_depth:
        xA = np.arange(len(yA)) / max(len(yA), 1)
        xB = np.arange(len(yB)) / max(len(yB), 1)
        xlabel = "Relative depth (layer / n_layers)"
    else:
        xA = np.arange(len(yA))
        xB = np.arange(len(yB))
        xlabel = "Layer index"
    plt.figure(figsize=(8, 4))
    plt.plot(xA, yA, label=label_A, marker="o", lw=1.5)
    plt.plot(xB, yB, label=label_B, marker="s", lw=1.5)
    plt.xlabel(xlabel)
    plt.ylabel("Layer profile")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def multi_overlay(
    curves: List[List[float]],
    labels: List[str],
    out_path: str,
    title: str,
    rel_depth: bool = True,
):
    """
    Plot K>=2 per-layer curves in one overlay.
    If rel_depth=True, x-axis is layer_index / n_layers for each curve.
    """
    _ensure_dir(out_path)
    if len(curves) == 0:
        # nothing to plot
        return
    plt.figure(figsize=(8, 4))
    # Distinct markers to aid reading; cycle if needed
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "+"]
    for i, y_list in enumerate(curves):
        y = np.array(y_list)
        if rel_depth:
            x = np.arange(len(y)) / max(len(y), 1)
            xlabel = "Relative depth (layer / n_layers)"
        else:
            x = np.arange(len(y))
            xlabel = "Layer index"
        label = labels[i] if i < len(labels) else f"Model {i+1}"
        marker = markers[i % len(markers)]
        plt.plot(x, y, label=label, marker=marker, lw=1.5)
    plt.xlabel(xlabel)
    plt.ylabel("Layer profile")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def nulls_plot(observed: List[float], null_summary: Dict, out_path: str, title: str):
    _ensure_dir(out_path)
    y_obs = np.array(observed)
    L = len(y_obs)
    x = np.arange(L)

    mean = np.array(null_summary.get("mean", [0] * L))
    lo = np.array(null_summary.get("lo", [0] * L))
    hi = np.array(null_summary.get("hi", [0] * L))

    plt.figure(figsize=(8, 4))
    plt.plot(x, y_obs, label="Observed", color="blue")
    plt.fill_between(x, lo, hi, color="gray", alpha=0.3, label="Null 90% band")
    plt.plot(x, mean, color="black", ls="--", label="Null mean")
    plt.xlabel("Layer index")
    plt.ylabel("Density / Profile")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def causal_bar(delta_top: float, delta_rand: float, out_path: str, title: str):
    _ensure_dir(out_path)
    plt.figure(figsize=(5, 4))
    xs = ["Top-k ablation", "Random-k ablation"]
    ys = [delta_top, delta_rand]
    colors = ["#d95f02", "#1b9e77"]
    plt.bar(xs, ys, color=colors)
    plt.axhline(0.0, color="black", lw=1)
    plt.ylabel("Δ Copy accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
