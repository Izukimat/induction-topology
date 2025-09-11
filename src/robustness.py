from __future__ import annotations

import random
from typing import List, Tuple, Dict

import numpy as np


def shuffled_antecedent(copy_ops: List[tuple[int, int, int]], toks) -> List[tuple[int, int, int]]:
    """
    For each (b,t,s), replace s with a random earlier index s' in [0, t-1], s' != s.
    This breaks the true match while preserving the marginal lag distribution roughly.
    """
    rng = random.Random()
    out: list[tuple[int, int, int]] = []
    for (b, t, s) in copy_ops:
        if t <= 0:
            out.append((b, t, s))
            continue
        choices = list(range(0, t))
        if s in choices and len(choices) > 1:
            choices.remove(s)
        s_prime = rng.choice(choices) if choices else s
        out.append((b, t, s_prime))
    return out


def permute_heads_within_layer(scores: np.ndarray, reps: int = 100, q: float = 0.9) -> Dict:
    """
    Build a null by randomly reassigning head scores across layers.
    For each repetition, sample H scores for each layer from the pooled distribution
    (breaks any layer-specific concentration). Compute per-layer density of 'high' heads
    (above the global q-quantile threshold). Return summary stats.
    """
    L, H = scores.shape
    flat = scores.flatten()
    thresh = float(np.quantile(flat, q))

    def density(v: np.ndarray) -> np.ndarray:
        return (v > thresh).mean(axis=1)

    observed_density = density(scores)

    rng = np.random.default_rng()
    sims = []
    for _ in range(reps):
        sampled = rng.choice(flat, size=(L, H), replace=True)
        sims.append(density(sampled))
    sims = np.stack(sims, axis=0)  # [reps, L]

    mean = sims.mean(axis=0)
    lo = np.quantile(sims, 0.05, axis=0)
    hi = np.quantile(sims, 0.95, axis=0)

    return {
        "q": q,
        "threshold": thresh,
        "observed_density": observed_density.tolist(),
        "mean": mean.tolist(),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
    }


def bootstrap_peak(profile_values: List[float], n_resamples: int = 1000) -> Dict:
    """
    Nonparametric bootstrap on per-layer values to estimate CI for peak relative depth and bandwidth.
    We resample layer indices with replacement to build synthetic profiles, then compute statistics.
    """
    vals = np.array(profile_values, dtype=np.float64)
    L = len(vals)
    if L == 0 or (vals <= 0).all():
        return {
            "peak_rel_mean": 0.0,
            "peak_rel_ci": [0.0, 0.0],
            "bandwidth_mean": 0.0,
            "bandwidth_ci": [0.0, 0.0],
        }

    def compute_bandwidth(v: np.ndarray) -> float:
        w = v / (v.sum() + 1e-12)
        cdf = np.cumsum(w)
        left = int(np.searchsorted(cdf, 0.16, side="left"))
        right = int(np.searchsorted(cdf, 0.84, side="left"))
        right = max(right, left)
        return float((right - left + 1) / L)

    peak_rels = []
    bws = []
    rng = np.random.default_rng()
    for _ in range(n_resamples):
        idx = rng.integers(0, L, size=L, dtype=np.int64)
        v_boot = np.zeros(L, dtype=np.float64)
        # accumulate resampled layers back onto their original indices
        for i in idx:
            v_boot[i] += vals[i]
        peak_layer = int(np.argmax(v_boot))
        peak_rels.append(float(peak_layer / L))
        bws.append(compute_bandwidth(v_boot))

    peak_rels = np.array(peak_rels)
    bws = np.array(bws)

    return {
        "peak_rel_mean": float(peak_rels.mean()),
        "peak_rel_ci": [float(np.quantile(peak_rels, 0.025)), float(np.quantile(peak_rels, 0.975))],
        "bandwidth_mean": float(bws.mean()),
        "bandwidth_ci": [float(np.quantile(bws, 0.025)), float(np.quantile(bws, 0.975))],
    }

