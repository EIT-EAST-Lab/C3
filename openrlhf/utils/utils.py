# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

from typing import List

import torch
import math
import torch.nn.functional as F
from transformers import AutoTokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        no_padding_batch.append((ids[mask.bool()]))
    return no_padding_batch


def safe_get_global_grad_norm(model) -> float:
    """Best-effort grad-norm for logging across plain nn.Module / DeepSpeedEngine.

    - If DeepSpeedEngine provides get_global_grad_norm(), use it.
    - Else compute from .grad (local) as a fallback.
    This is intended for *logging*; it does not clip or modify grads.
    """
    # ---- unwrap common wrappers ----
    # OpenRLHF Actor wrapper stores DeepSpeedEngine at actor.model
    # DDP stores real module at .module
    obj = model
    seen = set()
    for _ in range(8):
        if obj is None or id(obj) in seen:
            break
        seen.add(id(obj))

        fn = getattr(obj, "get_global_grad_norm", None)
        if callable(fn):
            try:
                v = fn()
                if v is None:
                    return 0.0
                if isinstance(v, torch.Tensor):
                    v = v.detach().float().item()
                return float(v)
            except Exception:
                # if DS exists but errors, fall through to unwrap/fallback
                pass

        nxt = None
        for attr in ("model", "engine", "module"):
            cand = getattr(obj, attr, None)
            if cand is not None and cand is not obj:
                nxt = cand
                break
        if nxt is None:
            break
        obj = nxt

    # ---- local fallback: sum ||grad||^2 over parameters we can see ----
    total_sq = 0.0
    params_fn = getattr(obj, "parameters", None)
    if not callable(params_fn):
        return 0.0

    for p in params_fn():
        g = getattr(p, "grad", None)
        if g is None:
            continue
        try:
            # L2 norm squared accumulation
            gn = float(g.detach().float().norm(2).item())
            total_sq += gn * gn
        except Exception:
            continue

    return float(math.sqrt(total_sq))
