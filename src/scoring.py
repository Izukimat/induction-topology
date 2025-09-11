from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np
import torch

from .runner import run_with_cache_all_layers, run_layerwise_cache, extract_attn_from_cache_layer


def head_scores_from_attn(attn: torch.Tensor, copy_ops: list[tuple[int, int, int]]) -> np.ndarray:
    """
    attn: [B, H, Q, K] for a single layer. For each head h, average attn[b,h,t,s] over copy_ops.
    Returns np.array shape [H].
    """
    if len(copy_ops) == 0:
        H = attn.shape[1]
        return np.zeros(H, dtype=np.float32)

    if not isinstance(attn, torch.Tensor):
        attn = torch.tensor(attn)

    B, H, Q, K = attn.shape
    device = attn.device

    bs = torch.tensor([b for (b, t, s) in copy_ops], dtype=torch.long, device=device)
    ts = torch.tensor([t for (b, t, s) in copy_ops], dtype=torch.long, device=device)
    ss = torch.tensor([s for (b, t, s) in copy_ops], dtype=torch.long, device=device)

    scores = []
    for h in range(H):
        vals = attn[bs, h, ts, ss]
        scores.append(vals.mean().item())

    return np.array(scores, dtype=np.float32)


def compute_scores(model, toks, copy_ops: list[tuple[int, int, int]], memory_saver: bool = False) -> np.ndarray:
    """
    Returns scores: [L, H]. If memory_saver=True, loop layers and call run_layerwise_cache;
    else run once with run_with_cache_all_layers.
    """
    L = model.cfg.n_layers  # type: ignore[attr-defined]
    H = model.cfg.n_heads   # type: ignore[attr-defined]
    scores = np.zeros((L, H), dtype=np.float32)

    if memory_saver:
        for l in range(L):
            attn = run_layerwise_cache(model, toks, l)
            scores[l] = head_scores_from_attn(attn, copy_ops)
        return scores

    # Single pass cache
    logits, cache = run_with_cache_all_layers(model, toks)
    for l in range(L):
        attn = extract_attn_from_cache_layer(cache, l)
        scores[l] = head_scores_from_attn(attn, copy_ops)
    return scores


def layer_profile(scores: np.ndarray, mode: str = "mean", q: float = 0.9) -> dict:
    """
    Compute per-layer means or top-quantile means.
    Returns dict with: 'per_layer': list[float], 'peak_layer': int, 'peak_rel': float, 'bandwidth': float
    Bandwidth: fraction of layers in the central 68% mass of layer scores.
    """
    L, H = scores.shape

    per_layer = []
    if mode == "mean":
        per_layer = scores.mean(axis=1)
    else:
        # top-quantile mean within each layer
        per_layer = np.array([
            s[np.argsort(s)][int(np.floor((1 - q) * len(s))):].mean() if len(s) > 0 else 0.0
            for s in scores
        ], dtype=np.float32)

    # Peak
    peak_layer = int(np.argmax(per_layer))
    peak_rel = float(peak_layer / max(L, 1))

    # Bandwidth: central 68% mass width
    vals = per_layer.astype(np.float64)
    total = float(vals.sum())
    if total <= 0:
        bandwidth = 0.0
    else:
        w = vals / total
        cdf = np.cumsum(w)
        lo = 0.16
        hi = 0.84
        left_idx = int(np.searchsorted(cdf, lo, side="left"))
        right_idx = int(np.searchsorted(cdf, hi, side="left"))
        right_idx = max(right_idx, left_idx)
        bandwidth = float((right_idx - left_idx + 1) / L)

    return {
        "mode": mode,
        "q": q,
        "per_layer": per_layer.tolist(),
        "peak_layer": peak_layer,
        "peak_rel": peak_rel,
        "bandwidth": bandwidth,
    }

