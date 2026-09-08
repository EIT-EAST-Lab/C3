# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/40_upstream.md and docs/41_changes_from_openrlhf.md for provenance.

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean


def _reduce_masked(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    token_level_loss: bool,
) -> torch.Tensor:
    """Reduce masked loss: token-level (global mean) or sequence-level (mean over seq, then batch)."""
    if token_level_loss:
        return masked_mean(x, mask, dim=None)
    return masked_mean(x, mask, dim=-1).mean()


class GPTLMLoss(nn.Module):
    """GPT LM cross-entropy loss with optional RingAttention label sharding."""

    IGNORE_INDEX = -100

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group is not None:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, V], labels: [B, T]
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start = self.ring_attn_rank * seq_len_per_process
            end = min(start + seq_len_per_process, total_seq_len)
            labels = labels[..., start:end]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # If all labels are ignored, CE would be NaN; keep grads alive with a 0-scaled term.
        if torch.all(shift_labels == self.IGNORE_INDEX):
            loss = shift_logits.mean() * 0
        else:
            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if self.ring_attn_group is not None:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size

        return loss


class SFTLoss(nn.Module):
    """Supervised fine-tuning loss from per-token log-probs."""

    def __init__(self, token_level_loss: bool = True):
        super().__init__()
        self.token_level_loss = token_level_loss

    def forward(self, per_token_logps: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        return _reduce_masked(-per_token_logps, loss_mask, token_level_loss=self.token_level_loss)


class PolicyLoss(nn.Module):
    """Policy loss for PPO/GSPO/REINFORCE.

    Returns:
      (loss, clip_ratio, ppo_kl, vllm_kl)
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        dual_clip: float = None,
        token_level_loss: bool = True,
        policy_loss_type: str = "ppo",
        enable_vllm_is_correction: bool = False,
        vllm_is_truncated_threshold: list = None,
        use_icepop: bool = False,
    ) -> None:
        super().__init__()
        self.clip_eps_low = float(clip_eps_low)
        self.clip_eps_high = float(clip_eps_high)
        self.dual_clip = dual_clip
        self.policy_loss_type = str(policy_loss_type)
        self.enable_vllm_is_correction = bool(enable_vllm_is_correction)
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.use_icepop = bool(use_icepop)

        # GSPO is sequence-level.
        self.token_level_loss = bool(token_level_loss)
        if self.policy_loss_type == "gspo":
            self.token_level_loss = False

        # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
        if self.dual_clip is not None:
            assert self.dual_clip > 1.0, f"dual_clip must be > 1.0, got {self.dual_clip}"

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        rollout_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if action_mask is None:
            raise ValueError("action_mask is required for policy loss")

        ptype = str(self.policy_loss_type).lower().strip()

        # --------------------
        # REINFORCE (no IS/clip)
        # --------------------
        if ptype == "reinforce":
            loss_tok = -(advantages * log_probs)
            loss = _reduce_masked(loss_tok, action_mask, token_level_loss=self.token_level_loss)
            z = torch.zeros((), dtype=loss.dtype, device=loss.device)
            return loss, z, z, None

        # --------------------
        # PPO / GSPO ratios
        # --------------------
        if ptype == "ppo":
            log_ratio = log_probs - old_log_probs
            ratio = log_ratio.exp()

        elif ptype == "gspo":
            # GSPO: https://arxiv.org/pdf/2507.18071
            if self.enable_vllm_is_correction:
                if rollout_log_probs is None:
                    raise ValueError("rollout_log_probs is required when enable_vllm_is_correction=True")
                log_ratio = log_probs - rollout_log_probs
            else:
                log_ratio = log_probs - old_log_probs

            denom = action_mask.sum(dim=-1).clamp_min(1)
            ratio_seq = (log_ratio * action_mask).sum(dim=-1) / denom
            ratio = ratio_seq.exp().unsqueeze(-1) * action_mask

        else:
            raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")

        # --------------------
        # PPO surrogate (also used by GSPO after seq-ratio broadcast)
        # --------------------
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

        if self.dual_clip is None:
            loss_tok = -torch.min(surr1, surr2)
        else:
            clip1 = torch.min(surr1, surr2)
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            loss_tok = -torch.where(advantages < 0, clip2, clip1)

        # --------------------
        # vLLM IS-correction (PPO only)
        # --------------------
        vllm_kl = None
        if self.enable_vllm_is_correction and ptype == "ppo":
            if rollout_log_probs is None:
                raise ValueError("rollout_log_probs is required when enable_vllm_is_correction=True")
            if not self.vllm_is_truncated_threshold or len(self.vllm_is_truncated_threshold) != 2:
                raise ValueError("vllm_is_truncated_threshold must be [low, high] when enable_vllm_is_correction=True")

            low_th, high_th = self.vllm_is_truncated_threshold
            if self.use_icepop:
                vllm_is = torch.exp(old_log_probs - rollout_log_probs).detach()
                keep = (vllm_is >= low_th) & (vllm_is <= high_th)
                vllm_is = vllm_is * keep
            else:
                vllm_is = torch.exp(old_log_probs - rollout_log_probs).clamp(min=low_th, max=high_th).detach()

            loss_tok = vllm_is * loss_tok
            vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)

        # --------------------
        # Metrics
        # --------------------
        loss = _reduce_masked(loss_tok, action_mask, token_level_loss=self.token_level_loss)
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        ppo_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
        return loss, clip_ratio, ppo_kl, vllm_kl


class ValueLoss(nn.Module):
    """Value loss for PPO-style critic training."""

    def __init__(self, clip_eps: float = None, token_level_loss: bool = True) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.token_level_loss = token_level_loss

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if action_mask is None:
            raise ValueError("action_mask is required for value loss")

        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss_tok = torch.max(surr1, surr2)
        else:
            loss_tok = (values - returns) ** 2

        loss = _reduce_masked(loss_tok, action_mask, token_level_loss=self.token_level_loss)
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """Pairwise loss for reward model."""

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """Pairwise loss for reward model: https://arxiv.org/abs/2204.05862"""

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        return torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()


class DPOLoss(nn.Module):
    """DPO loss."""

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; smoothing=0 gives original DPO.
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return loss, chosen_rewards, rejected_rewards


# Adapted from:
# https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """KTO loss for even sampling."""

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from:
# https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """KTO loss for uneven sampling."""

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat((self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from:
# https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """Language model knowledge distillation loss."""

    IGNORE_INDEX = -100

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)

        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class PRMLoss(nn.Module):
    """Process reward model loss."""

    IGNORE_INDEX = -100

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask].squeeze(1)
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # Soft label.
            assert self.reward_token_ids is not None and len(self.reward_token_ids) == 2
            logits = logits[..., self.reward_token_ids]
            pos = labels.to(logits.dtype)
            neg = 1 - pos
            neg[pos != -100] = 1 - pos[pos != -100]
            labels = torch.stack([pos, neg], dim=-1)

        elif self.reward_token_ids is not None:
            # Hard label mapped into [0..len(reward_token_ids)-1].
            logits = logits[..., self.reward_token_ids]
            for i, tok in enumerate(self.reward_token_ids):
                labels = torch.where(labels == tok, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc
