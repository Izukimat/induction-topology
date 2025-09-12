from __future__ import annotations

import random
from contextlib import contextmanager, nullcontext
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
                if h is not None:
                    handles.append(h)
            yield
        finally:
            for h in handles:
                if hasattr(h, "remove"):
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

def _copy_accuracy_batched(
    model,
    toks,
    copy_ops: List[tuple[int, int, int]],
    batch_size: int = 8,
    ablate_heads: List[tuple[int, int]] | None = None,
) -> float:
    """
    Compute copy accuracy without building full logits for the whole batch.
    Runs in micro-batches and accumulates correct/total counts.
    If ablate_heads is provided, applies ablation during the forward passes.
    """
    if len(copy_ops) == 0:
        return 0.0
    B = int(toks.shape[0])
    bsz = max(1, int(batch_size))
    correct_total = 0
    seen_global = set()
    ctx = ablate_heads_context(model, ablate_heads) if ablate_heads else nullcontext()
    with ctx:
        for start in range(0, B, bsz):
            end = min(start + bsz, B)
            cur_ops = [(b - start, t, s) for (b, t, s) in copy_ops if start <= b < end]
            if len(cur_ops) == 0:
                continue
            with torch.no_grad():
                preds = model(toks[start:end]).argmax(dim=-1)
            for (b, t, s) in cur_ops:
                key = (int(b + start), int(t))
                if key in seen_global:
                    continue
                seen_global.add(key)
                correct_total += int(preds[b, t].item() == toks[start + b, t].item())
    total_positions = len({(int(b), int(t)) for (b, t, _s) in copy_ops})
    return float(correct_total / max(total_positions, 1))


def measure_causal_delta(
    model,
    toks,
    copy_ops: List[tuple[int, int, int]],
    top_heads: List[tuple[int, int]],
    k_random: int = 3,
    batch_size: int = 8,
) -> Dict:
    """
    Measure copy accuracy baseline; then with top_heads ablated; then with k_random heads ablated.
    Return deltas.
    """
    base_acc = _copy_accuracy_batched(model, toks, copy_ops, batch_size=batch_size, ablate_heads=None)

    # Top heads ablation
    top_acc = _copy_accuracy_batched(model, toks, copy_ops, batch_size=batch_size, ablate_heads=top_heads)

    # Random heads ablation
    L = int(getattr(model.cfg, "n_layers"))  # type: ignore[attr-defined]
    H = int(getattr(model.cfg, "n_heads"))   # type: ignore[attr-defined]
    all_heads = [(l, h) for l in range(L) for h in range(H)]
    rng = random.Random(123)
    rand_heads = rng.sample(all_heads, k=min(k_random, len(all_heads)))
    rand_acc = _copy_accuracy_batched(model, toks, copy_ops, batch_size=batch_size, ablate_heads=rand_heads)

    return {
        "baseline_acc": base_acc,
        "ablated_top_acc": top_acc,
        "ablated_rand_acc": rand_acc,
        "delta_top": float(top_acc - base_acc),
        "delta_rand": float(rand_acc - base_acc),
        "random_heads": rand_heads,
    }
