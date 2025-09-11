from __future__ import annotations

import random
from contextlib import contextmanager
from typing import List, Tuple, Dict

import torch


def ablate_heads_context(model, heads: List[tuple[int, int]]):
    """
    Register temporary hooks to zero selected heads' 'z' (value stream) during forward.
    Returns a context manager that applies the ablation within the block.
    heads: list of (layer_idx, head_idx)
    """
    layer_to_heads: dict[int, list[int]] = {}
    for (l, h) in heads:
        layer_to_heads.setdefault(int(l), []).append(int(h))

    handles = []

    def make_hook(layer_idx: int, head_ids: list[int]):
        def hook_fn(z, hook):  # z expected shape [B, pos, n_heads, d_head] or [B, n_heads, pos, d_head]
            n_heads = getattr(model.cfg, "n_heads")  # type: ignore[attr-defined]
            if z.ndim != 4:
                return z
            # Identify head dimension
            if z.shape[1] == n_heads:
                # [B, H, P, d]
                z[:, head_ids, :, :] = 0.0
            elif z.shape[2] == n_heads:
                # [B, P, H, d]
                z[:, :, head_ids, :] = 0.0
            else:
                # Unexpected shape; no-op
                return z
            return z

        return hook_fn

    @contextmanager
    def ctx():
        try:
            for l, hs in layer_to_heads.items():
                name = f"blocks.{l}.attn.hook_z"
                h = model.add_hook(name, make_hook(l, hs))
                handles.append(h)
            yield
        finally:
            for h in handles:
                h.remove()

    return ctx()


def copy_accuracy(logits, toks, copy_ops: List[tuple[int, int, int]]) -> float:
    """
    At positions t in copy_ops, compute fraction where argmax(logits[b,t]) equals token at (b,t).
    """
    if len(copy_ops) == 0:
        return 0.0
    with torch.no_grad():
        pred = logits.argmax(dim=-1)  # [B, T]
        correct = 0
        total = 0
        seen = set()
        for (b, t, s) in copy_ops:
            key = (int(b), int(t))
            if key in seen:
                continue
            seen.add(key)
            correct += int(pred[b, t].item() == toks[b, t].item())
            total += 1
        return float(correct / max(total, 1))


def measure_causal_delta(model, toks, copy_ops: List[tuple[int, int, int]], top_heads: List[tuple[int, int]], k_random: int = 3) -> Dict:
    """
    Measure copy accuracy baseline; then with top_heads ablated; then with k_random heads ablated.
    Return deltas.
    """
    with torch.no_grad():
        base_logits = model(toks)
    base_acc = copy_accuracy(base_logits, toks, copy_ops)

    # Top heads ablation
    with ablate_heads_context(model, top_heads):
        with torch.no_grad():
            logits_top = model(toks)
    top_acc = copy_accuracy(logits_top, toks, copy_ops)

    # Random heads ablation
    L = int(getattr(model.cfg, "n_layers"))  # type: ignore[attr-defined]
    H = int(getattr(model.cfg, "n_heads"))   # type: ignore[attr-defined]
    all_heads = [(l, h) for l in range(L) for h in range(H)]
    rng = random.Random(123)
    rand_heads = rng.sample(all_heads, k=min(k_random, len(all_heads)))
    with ablate_heads_context(model, rand_heads):
        with torch.no_grad():
            logits_rand = model(toks)
    rand_acc = copy_accuracy(logits_rand, toks, copy_ops)

    return {
        "baseline_acc": base_acc,
        "ablated_top_acc": top_acc,
        "ablated_rand_acc": rand_acc,
        "delta_top": float(top_acc - base_acc),
        "delta_rand": float(rand_acc - base_acc),
        "random_heads": rand_heads,
    }

