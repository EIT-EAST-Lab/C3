# -*- coding: utf-8 -*-
"""c3.credit.c3.scoring

Scoring utilities used by Rule-B C3.

Only batched Q-critic scoring is kept on the main path.
All legacy C3 credit computations (e.g., K×K Q-matrix / regenerate / variance gating)
have been removed from the main training path.
"""

from __future__ import annotations

from typing import List

import torch


@torch.no_grad()
def score_texts_batched(
    q_critic,
    texts: List[str],
    device: torch.device,
    max_len: int,
    forward_bs: int,
) -> torch.Tensor:
    """Batched Q scoring with C3 semantics.

    Args:
      q_critic:
        - must expose `.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=...)`
        - forward signature: q_critic(input_ids, attention_mask, apply_sigmoid=True) -> [B] or [B,1]
      texts: list of input strings
      device: torch.device for forward
      max_len: tokenizer truncation length
      forward_bs: micro-batch size

    Returns:
      torch.Tensor on CPU, shape [len(texts)], dtype float32
    """
    if not texts:
        return torch.empty(0, dtype=torch.float32)

    bs = max(1, int(forward_bs))

    toks = q_critic.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_len),
    )
    ids_all = toks.input_ids
    msk_all = toks.attention_mask

    outs_cpu: List[torch.Tensor] = []
    nb = (device.type == "cuda")

    for s in range(0, ids_all.size(0), bs):
        e = min(s + bs, ids_all.size(0))
        ids = ids_all[s:e].to(device, non_blocking=nb)
        msk = msk_all[s:e].to(device, non_blocking=nb)

        q = q_critic(ids, msk, apply_sigmoid=True).view(-1)
        outs_cpu.append(q.detach().float().cpu())

        # match C3's memory-friendly behavior
        del ids, msk, q

    return torch.cat(outs_cpu, dim=0)
