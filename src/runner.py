from __future__ import annotations

from typing import Tuple
import torch


def _pattern_key(layer_idx: int) -> str:
    return f"blocks.{layer_idx}.attn.hook_pattern"


def run_with_cache_all_layers(model, toks) -> tuple:
    """
    Run model once and return (logits, cache). Cache includes attention per layer.
    Uses names_filter to retain only attention patterns to reduce memory.
    """
    logits, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: "attn" in n and "hook_pattern" in n,
        return_type="both",
    )
    return logits, cache


def run_layerwise_cache(model, toks, layer_idx: int):
    """
    Run model focusing on a single layer's attention (names_filter for that layer only),
    to avoid OOM. Returns attention tensor for that layer only: [B, H, Q, K].
    """
    key = _pattern_key(layer_idx)
    logits, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: n == key,
        return_type="both",
    )
    attn = cache[key]
    # Ensure on CPU for downstream numpy ops
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().to("cpu")
    return attn


def extract_attn_from_cache_layer(cache, layer_idx: int):
    key = _pattern_key(layer_idx)
    attn = cache[key]
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().to("cpu")
    return attn

