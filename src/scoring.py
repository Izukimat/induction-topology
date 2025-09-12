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


def compute_scores(
    model,
    toks,
    copy_ops: list[tuple[int, int, int]],
    memory_saver: bool = False,
    batch_size: int | None = None,
) -> np.ndarray:
    """
    Returns scores: [L, H]. If memory_saver=True, loop layers and call run_layerwise_cache;
    else run once with run_with_cache_all_layers.
    """
    L = model.cfg.n_layers  # type: ignore[attr-defined]
    H = model.cfg.n_heads   # type: ignore[attr-defined]
    scores = np.zeros((L, H), dtype=np.float32)
    # Number of sequences in the batch
    B = int(toks.shape[0]) if hasattr(toks, "shape") else len(toks)
    bsz = int(batch_size) if batch_size and batch_size > 0 else B

    if memory_saver:
        # Process one layer at a time and micro-batch the sequences to avoid OOM.
        for l in range(L):
            # Weighted accumulation over chunks
            sum_h = np.zeros((H,), dtype=np.float64)
            total_ops = 0
            for start in range(0, B, bsz):
                end = min(start + bsz, B)
                # Filter and reindex copy ops for this chunk
                ops_chunk = [(b - start, t, s) for (b, t, s) in copy_ops if start <= b < end]
                if len(ops_chunk) == 0:
                    continue
                toks_chunk = toks[start:end]
                attn = run_layerwise_cache(model, toks_chunk, l)
                s_chunk = head_scores_from_attn(attn, ops_chunk).astype(np.float64)
                sum_h += s_chunk * float(len(ops_chunk))
                total_ops += len(ops_chunk)
            if total_ops > 0:
                scores[l] = (sum_h / float(total_ops)).astype(np.float32)
            else:
                scores[l] = np.zeros((H,), dtype=np.float32)
        return scores

    # Single pass cache across all layers. If batch is large, micro-batch and accumulate.
    if bsz >= B:
        # No need to chunk
        logits, cache = run_with_cache_all_layers(model, toks)
        for l in range(L):
            attn = extract_attn_from_cache_layer(cache, l)
            scores[l] = head_scores_from_attn(attn, copy_ops)
        return scores
    else:
        # Micro-batch: run all layers per chunk and combine with weighted means.
        sum_lh = np.zeros((L, H), dtype=np.float64)
        total_ops = 0
        for start in range(0, B, bsz):
            end = min(start + bsz, B)
            ops_chunk = [(b - start, t, s) for (b, t, s) in copy_ops if start <= b < end]
            if len(ops_chunk) == 0:
                continue
            toks_chunk = toks[start:end]
            logits, cache = run_with_cache_all_layers(model, toks_chunk)
            for l in range(L):
                attn = extract_attn_from_cache_layer(cache, l)
                s_chunk = head_scores_from_attn(attn, ops_chunk).astype(np.float64)
                sum_lh[l] += s_chunk * float(len(ops_chunk))
            total_ops += len(ops_chunk)
        if total_ops > 0:
            scores = (sum_lh / float(total_ops)).astype(np.float32)
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
        # Top-quantile mean within each layer.
        # Here q is the quantile threshold (e.g., q=0.9 keeps the top 10%).
        per_layer = np.array([
            (s[np.argsort(s)][int(np.floor(q * len(s))):].mean() if len(s) > 0 else 0.0)
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
