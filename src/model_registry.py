import os
from typing import List

import torch

try:
    from transformer_lens import HookedTransformer
except Exception as e:  # pragma: no cover - allows repo to load without deps installed
    HookedTransformer = None  # type: ignore


PYTHIA_MAP = {
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
}


def _to_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str.lower(), torch.float16)


def load_model(model_id: str, device: str = "cuda", dtype: str = "float16"):
    """
    Load a HookedTransformer by model_id (e.g., 'pythia-410m', 'pythia-1b', 'gpt2').
    Set dtype and device. Return model ready for inference.
    """
    if HookedTransformer is None:
        raise RuntimeError(
            "transformer_lens is not installed. Please install per the setup instructions."
        )

    hf_id = PYTHIA_MAP.get(model_id, model_id)
    dt = _to_dtype(dtype)

    model = HookedTransformer.from_pretrained(
        hf_id,
        device=device,
        dtype=dt,
    )
    model.eval()
    return model


def to_tokens(model, texts: List[str], seq_len: int, prepend_bos: bool = True):
    """
    Tokenize a list of strings into a padded tensor of shape [B, T] (T=seq_len),
    truncating or padding with BOS/space as needed.
    """
    if not hasattr(model, "to_tokens"):
        raise ValueError("Model does not expose to_tokens method (expected HookedTransformer)")

    toks = model.to_tokens(texts, prepend_bos=prepend_bos)
    # Ensure batch dimension exists
    if toks.ndim == 1:
        toks = toks.unsqueeze(0)

    B, T = toks.shape[0], toks.shape[-1]
    if T > seq_len:
        toks = toks[:, :seq_len]
    elif T < seq_len:
        # pad with space token (ensuring it's a single token id)
        try:
            pad_id = int(model.to_single_token(" "))
        except Exception:
            pad_id = int(model.tokenizer.encode(" ")[0])
        pad = torch.full((B, seq_len - T), pad_id, dtype=toks.dtype, device=toks.device)
        toks = torch.cat([toks, pad], dim=-1)

    return toks

