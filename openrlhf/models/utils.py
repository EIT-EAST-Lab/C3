# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/40_upstream.md and docs/41_changes_from_openrlhf.md for provenance.

import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    log_ratio = log_ratio.clamp(min=-10, max=10)
    return log_ratio


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    kl_reward = -kl_coef * kl

    if action_mask is None:
        # Fallback: apply scalar reward everywhere (should be rare; keep behavior explicit).
        # Most RLHF paths provide action_mask.
        if isinstance(kl, list):
            return [(-kl_coef * k) + (torch.as_tensor(r, device=k.device, dtype=k.dtype)) for k in kl]
        else:
            return kl_reward + torch.as_tensor(r, device=kl.device, dtype=kl.dtype)

    # The following code is equivalent to:
    #
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break
    #
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    reward = last_reward + kl_reward
    return reward


def _logsumexp_by_chunk(logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    # logits: [N, V]
    seq_len = logits.shape[0]
    logsumexp_values = torch.zeros((seq_len,), device=logits.device, dtype=logits.dtype)
    for s_idx in range(0, seq_len, chunk_size):
        end_idx = min(s_idx + chunk_size, seq_len)
        logsumexp_values[s_idx:end_idx] = torch.logsumexp(logits[s_idx:end_idx], dim=-1)
    return logsumexp_values


def log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    inplace_backward: bool | None = None,
) -> torch.Tensor:
    """
    Compute log p(label_t | logits_t) for each position, returning shape logits.shape[:-1].

    logits: [..., V]
    labels: [...]
    """
    if temperature != 1.0:
        # In-place scaling to avoid allocating another logits-sized tensor.
        # This mirrors previous behavior, but without forcing fp32 copies.
        logits.div_(temperature)

    batch_dim = logits.shape[:-1]
    last_dim = logits.shape[-1]

    if labels.shape != batch_dim:
        raise ValueError(f"labels shape {tuple(labels.shape)} must match logits.shape[:-1] {tuple(batch_dim)}")

    # Prefer flash-attn CE if available (fast + memory-friendly).
    if inplace_backward is None:
        inplace_backward = str(os.environ.get("FLASH_ATTN_CE_INPLACE_BWD", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
        }

    # Flatten to [N, V] and [N]
    logits_2d = logits.reshape(-1, last_dim)
    labels_1d = labels.reshape(-1)

    # If labels contain ignore_index (-100), keep it safe.
    ignore_index = -100

    # Try flash-attn kernel (works best on CUDA; if unavailable, fallback to torch CE).
    # NOTE: Unlike the old code, we do NOT require logits to be fp32 here.
    try:
        from flash_attn.ops.triton.cross_entropy import cross_entropy_loss  # type: ignore

        # flash-attn expects CUDA tensor
        if logits_2d.is_cuda:
            try:
                output = cross_entropy_loss(
                    logits_2d,
                    labels_1d,
                    inplace_backward=bool(inplace_backward),
                )
            except TypeError:
                # older signature
                output = cross_entropy_loss(logits_2d, labels_1d)

            # output[0] is per-token loss (N,)
            loss_1d = output[0]
            log_probs_labels = (-loss_1d).view(*batch_dim)
            return log_probs_labels
    except Exception:
        # ImportError or runtime incompatibility -> fallback
        pass

    # Torch fallback (robust; typically does not materialize full log_softmax)
    loss_1d = F.cross_entropy(
        logits_2d,
        labels_1d,
        reduction="none",
        ignore_index=ignore_index,
    )
    log_probs_labels = (-loss_1d).view(*batch_dim)
    return log_probs_labels


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Memory-safe token entropy for logits[..., V].

    The old implementation:
        pd = softmax(logits)  # allocates logits-sized tensor
        entropy = logsumexp(logits) - sum(pd * logits)
    can be extremely memory-hungry for large vocab.

    This implementation computes:
        Z = sum(exp(x))
        E[x] = sum(exp(x) * x) / Z
        H = log(Z) - E[x]
    using vocab-chunking to avoid allocating full softmax.

    Chunk size can be controlled by env:
        OPENRLHF_ENTROPY_VOCAB_CHUNK (default 2048)
    """
    chunk = int(str(os.environ.get("OPENRLHF_ENTROPY_VOCAB_CHUNK", "2048")).strip() or "2048")
    if chunk <= 0:
        # Fallback to the original (may OOM on large logits)
        pd = torch.nn.functional.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
        return entropy

    # logits: [..., V]
    V = logits.shape[-1]
    # max for numerical stability (float32, but only [...])
    m = logits.max(dim=-1).values.float()  # [...]

    exp_sum = torch.zeros_like(m)          # [...]
    exp_x_sum = torch.zeros_like(m)        # [...]

    # Iterate over vocab in chunks
    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        x = logits[..., s:e].float()  # [..., c]
        ex = torch.exp(x - m.unsqueeze(-1))  # [..., c]
        exp_sum = exp_sum + ex.sum(dim=-1)
        exp_x_sum = exp_x_sum + (ex * x).sum(dim=-1)

    # logZ = log(exp_sum) + m
    logZ = torch.log(exp_sum.clamp_min(1e-20)) + m
    # E[x] = exp_x_sum / exp_sum
    Ex = exp_x_sum / exp_sum.clamp_min(1e-20)
    entropy = logZ - Ex
    return entropy
