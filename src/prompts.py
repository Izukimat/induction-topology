import random
import string
import typing as T


def _sample_lags(n: int, lag_values: list[int], mix: str) -> list[int]:
    if mix == "uniform" or not lag_values:
        return [random.choice(lag_values) for _ in range(n)]
    # Fallback uniform if unknown mix strategy
    return [random.choice(lag_values) for _ in range(n)]


def make_abab_prompts(n: int, seq_len: int, lag_values: list[int], mix: str) -> list[str]:
    """
    Create ABAB-like patterns with controlled lags. Example: "A B A B A B ..." where
    A/B are sampled single-character tokens. Ensures repeats at target lags.

    Strategy:
    - Build a base stream of uppercase letters separated by spaces to encourage single-tokenization in GPT-2-like BPEs.
    - For each sequence, insert repeated tokens at positions offset by chosen lags.
    """
    letters = list(string.ascii_uppercase)
    prompts: list[str] = []
    lags = _sample_lags(n, lag_values, mix)

    for i in range(n):
        lag = lags[i]
        # initialize with random letters
        toks = [random.choice(letters) for _ in range(max(seq_len // 2, lag + 2))]
        # inject ABAB behavior: ensure token at t repeats token at (t - lag)
        for t in range(lag, len(toks), lag):
            toks[t] = toks[t - lag]
        # render with spaces, pad/truncate to approx length
        s = " ".join(toks)
        s = s[: max(1, seq_len * 2)]  # generous, tokenizer will cut/pad later
        prompts.append(s)
    return prompts


def make_name_prompts(n: int, seq_len: int, lag_values: list[int], mix: str) -> list[str]:
    """
    Create prompts with repeated single-token-like names.
    We use short capitalized names that are commonly single tokens in GPT-2 BPE.

    Example: "Alice ... Bob ... Alice ... Bob ..." with controlled lags.
    """
    # Short names likely to be single tokens; leading spaces handled by tokenizer
    names = [
        "Alice",
        "Bob",
        "Carol",
        "Dave",
        "Erin",
        "Frank",
        "Grace",
        "Heidi",
        "Ivan",
        "Judy",
        "Kathy",
        "Larry",
        "Mona",
        "Nina",
        "Owen",
        "Paul",
        "Quinn",
        "Ruth",
        "Sara",
        "Tina",
        "Uma",
        "Vera",
        "Walt",
        "Xena",
        "Yuri",
        "Zoe",
    ]

    prompts: list[str] = []
    lags = _sample_lags(n, lag_values, mix)

    for i in range(n):
        lag = lags[i]
        L = max(seq_len // 2, lag + 2)
        seq = [random.choice(names) for _ in range(L)]
        for t in range(lag, L, lag):
            seq[t] = seq[t - lag]
        s = " ".join(seq)
        s = s[: max(1, seq_len * 3)]
        prompts.append(s)
    return prompts


def find_copy_ops(token_tensor) -> list[tuple[int, int, int]]:
    """
    Given tokens [B, T], return list of (b, t, s) pairs where tokens[b,t]==tokens[b,s], s<t.
    """
    import torch

    toks = token_tensor
    assert toks.ndim == 2, "Expected [B, T] token tensor"
    B, T = toks.shape
    ops: list[tuple[int, int, int]] = []
    for b in range(B):
        seen: dict[int, list[int]] = {}
        for t in range(T):
            tok = int(toks[b, t].item())
            if tok in seen:
                for s in seen[tok]:
                    if s < t:
                        ops.append((b, t, s))
                seen[tok].append(t)
            else:
                seen[tok] = [t]
    return ops

