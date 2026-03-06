"""
Replay buffer utilities.

Key ideas:
- Store rollout Experience as per-sample BufferItem (unpadded where possible).
- Sample/Collate BufferItems back into a padded batched Experience.
- IMPORTANT: Never silently drop optional fields (esp. centralized critic fields).
  Mixed presence inside one batch is treated as a bug and fails fast.
"""

# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

from __future__ import annotations

import random
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import distributed as dist

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions
from openrlhf.utils.utils import zero_pad_sequences


# =============================================================================
# Buffer schema
# =============================================================================

_CRITIC_KEYS: Tuple[str, ...] = (
    "critic_input_ids",
    "critic_attention_mask",
    "critic_action_mask",
    "critic_values",
    "critic_returns",
)


@dataclass
class BufferItem:
    """One unbatched sample stored in replay buffer."""

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    rollout_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    action_mask: Optional[torch.Tensor]

    # Optional centralized critic fields
    critic_input_ids: Optional[torch.Tensor] = None
    critic_attention_mask: Optional[torch.Tensor] = None
    critic_action_mask: Optional[torch.Tensor] = None
    critic_values: Optional[torch.Tensor] = None
    critic_returns: Optional[torch.Tensor] = None

    info: Optional[dict] = None


def _bufferitem_tensor_keys() -> Tuple[str, ...]:
    """All BufferItem dataclass fields except 'info'."""
    return tuple(f.name for f in fields(BufferItem) if f.name != "info")


# =============================================================================
# Experience <-> BufferItem
# =============================================================================

def split_experience_batch(experience: Experience) -> List[BufferItem]:
    """Split a batched Experience into per-sample BufferItems."""
    batch_size = len(experience.sequences)
    keys = _bufferitem_tensor_keys()

    # Some pipelines attach a transient index; clear it before storing.
    if hasattr(experience, "index"):
        experience.index = None

    # Validate batch-aligned fields.
    for k in keys:
        v = getattr(experience, k, None)
        if v is None:
            continue
        if isinstance(v, (torch.Tensor, list)) and len(v) != batch_size:
            raise ValueError(f"Size of {k} ({len(v)}) != batch_size ({batch_size})")

    info_dict = getattr(experience, "info", None) or {}
    if not isinstance(info_dict, dict):
        raise TypeError(f"Experience.info must be dict, got {type(info_dict)}")

    def _slice_mapping(obj, i: int):
        """Slice nested Mapping values if batch-aligned; otherwise keep as constant."""
        if isinstance(obj, torch.Tensor):
            try:
                return obj[i] if obj.dim() >= 1 and obj.size(0) == batch_size else obj
            except Exception:
                return obj
        if isinstance(obj, list):
            return obj[i] if len(obj) == batch_size else obj
        if isinstance(obj, Mapping):
            return {kk: _slice_mapping(vv, i) for kk, vv in obj.items()}
        return obj

    items: List[BufferItem] = []
    for i in range(batch_size):
        item_kwargs = {}
        for k in keys:
            v = getattr(experience, k, None)
            item_kwargs[k] = (v[i] if v is not None else None)

        info_out: dict = {}
        for k, v in info_dict.items():
            if v is None:
                info_out[k] = None
            elif isinstance(v, (torch.Tensor, list)):
                if len(v) != batch_size:
                    raise ValueError(f"Size of info[{k}] ({len(v)}) != batch_size ({batch_size})")
                info_out[k] = v[i]
            elif isinstance(v, Mapping):
                info_out[k] = _slice_mapping(v, i)
            else:
                # Keep strict: scalars/strings must be batch-aligned lists in Experience.info.
                raise TypeError(f"Unsupported type for info[{k}]: {type(v)}")

        item_kwargs["info"] = info_out
        items.append(BufferItem(**item_kwargs))

    return items


def make_experience_batch(items: List[BufferItem], packing_samples: bool = False) -> Experience:
    """Combine per-sample BufferItems into a batched Experience."""
    if not items:
        raise ValueError("Empty items list")

    keys = _bufferitem_tensor_keys()

    def _any_present(k: str) -> bool:
        return any(getattr(it, k, None) is not None for it in items)

    def _all_present(k: str) -> bool:
        return all(getattr(it, k, None) is not None for it in items)

    # Main tensor fields: never let a single sample decide presence for the whole batch.
    kwargs = {}
    for k in keys:
        if not _any_present(k):
            kwargs[k] = None
            continue
        if not _all_present(k):
            raise RuntimeError(
                f"[ReplayBuffer][FAIL-FAST] Mixed presence for field {k!r} in one batch. "
                "This would silently drop fields and can break centralized critic routing."
            )
        kwargs[k] = zero_pad_sequences([getattr(it, k) for it in items], "right", stack=True)

    def _merge_mapping(dicts: List[Mapping]):
        """Merge list[dict] -> dict-of-batched-values (recursive)."""
        keys_union = set()
        for d in dicts:
            if d is None:
                continue
            keys_union |= set(d.keys())

        out = {}
        for kk in keys_union:
            sub = [(d.get(kk) if d is not None else None) for d in dicts]
            non_none = [x for x in sub if x is not None]

            if non_none and all(isinstance(x, Mapping) for x in non_none):
                out[kk] = _merge_mapping([x if isinstance(x, Mapping) else {} for x in sub])
                continue

            if non_none and all(isinstance(x, (int, float)) for x in non_none) and len(non_none) == len(sub):
                out[kk] = torch.tensor(sub)
                continue

            out[kk] = sub
        return out

    # Info dict: preserve existing behavior (keys sourced from first item only).
    kwargs["info"] = {}
    first_info = items[0].info or {}
    if not isinstance(first_info, dict):
        raise TypeError(f"BufferItem.info must be dict, got {type(first_info)}")

    for key in first_info.keys():
        values = [((it.info or {}).get(key)) for it in items]
        non_none = [v for v in values if v is not None]

        if non_none and all(isinstance(v, Mapping) for v in non_none):
            dicts = [v if isinstance(v, Mapping) else {} for v in values]
            kwargs["info"][key] = _merge_mapping(dicts)
            continue

        if non_none:
            first_type = type(non_none[0])
            if not all(isinstance(v, first_type) for v in non_none):
                raise TypeError(f"Inconsistent types in info[{key}]")

        if values and all(isinstance(v, (int, float)) for v in values):
            kwargs["info"][key] = torch.tensor(values)
        else:
            kwargs["info"][key] = values

    return Experience(**kwargs)


# =============================================================================
# Padding trimming
# =============================================================================

def remove_padding_in_sequences(items: List[BufferItem]) -> List[BufferItem]:
    """Trim right padding for each BufferItem based on attention masks."""
    # Fields trimmed using the main attention_mask (exclude info and critic fields).
    main_keys: Tuple[str, ...] = tuple(
        f.name for f in fields(BufferItem) if f.name not in ("info",) + _CRITIC_KEYS
    )

    for item in items:
        # ---- main tensors ----
        if item.attention_mask is not None:
            right_pad = item.attention_mask.flip(0).argmax()
            right_pad = None if right_pad == 0 else -right_pad
            for k in main_keys:
                v = getattr(item, k, None)
                if v is not None:
                    setattr(item, k, v[:right_pad])

        # ---- critic tensors ----
        cam = getattr(item, "critic_attention_mask", None)
        if cam is not None:
            c_right_pad = cam.flip(0).argmax()
            c_right_pad = None if c_right_pad == 0 else -c_right_pad

            # Only trim sequence-aligned critic tensors.
            # Do NOT trim width-1 critic_action_mask / scalar critic values/returns.
            c_ids = getattr(item, "critic_input_ids", None)
            if c_ids is not None:
                item.critic_input_ids = c_ids[:c_right_pad]
            item.critic_attention_mask = cam[:c_right_pad]

            # Legacy compatibility: if critic_action_mask is sequence-aligned, trim it too.
            c_am = getattr(item, "critic_action_mask", None)
            if c_am is not None and isinstance(c_am, torch.Tensor):
                try:
                    if c_am.dim() >= 1 and cam.dim() >= 1 and int(c_am.shape[0]) == int(cam.shape[0]):
                        item.critic_action_mask = c_am[:c_right_pad]
                except Exception:
                    pass

    return items


# =============================================================================
# DP balancing (length-based)
# =============================================================================

def balance_experiences(experiences: Sequence[Experience], args) -> List[Experience]:
    """Balance experience across DP ranks by length; reorder only."""
    items_all: List[BufferItem] = []
    for exp in experiences:
        items_all.extend(split_experience_batch(exp))

    if not items_all:
        return []

    def _to_int(x) -> int:
        if isinstance(x, torch.Tensor):
            if x.numel() <= 0:
                return 0
            return int(x.view(-1)[0].item())
        try:
            return int(x)
        except Exception:
            return 0

    items_all.sort(key=lambda it: _to_int((it.info or {}).get("total_length", 0)), reverse=True)

    effective_num = (
        args.actor_num_nodes
        * args.actor_num_gpus_per_node
        // args.ring_attn_size
        // args.ds_tensor_parallel_size
    )
    effective_num = max(1, int(effective_num))

    split_items: List[List[BufferItem]] = [
        items_all[i : i + effective_num] for i in range(0, len(items_all), effective_num)
    ]

    half = len(split_items) // 2
    first_half = split_items[:half]
    last_half = [blk[::-1] for blk in split_items[half:]]

    interval_items: List[List[BufferItem]] = []
    for i in range(half):
        interval_items.append(first_half[i])
        interval_items.append(last_half[-(i + 1)])
    if len(last_half) > len(first_half):
        interval_items.append(last_half[0])

    slot_groups: List[List[BufferItem]] = [[] for _ in range(effective_num)]
    for blk in interval_items:
        for j, it in enumerate(blk):
            if j >= effective_num:
                break
            slot_groups[j].append(it)

    in_count = len(items_all)
    out_count = sum(len(g) for g in slot_groups)
    if out_count != in_count:
        raise RuntimeError(
            f"[DynamicBatch][BUG] balance_experiences changed item count: "
            f"in={in_count} out={out_count} effective_num={effective_num}"
        )

    return [make_experience_batch(g) for g in slot_groups if g]


# =============================================================================
# Replay buffer
# =============================================================================

class NaiveReplayBuffer(ABC):
    """Naive replay buffer storing BufferItems."""

    def __init__(
        self,
        sample_batch_size: int,
        limit: int = 0,
        cpu_offload: bool = True,
        packing_samples: bool = False,
        dynamic_batch: bool = False,
    ) -> None:
        super().__init__()
        self.sample_batch_size = int(sample_batch_size)
        self.limit = int(limit)  # <=0 means unlimited
        self.cpu_offload = bool(cpu_offload)
        self.packing_samples = bool(packing_samples)
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")

        self.items: List[BufferItem] = []

        self.dynamic_batch = bool(dynamic_batch)
        self.dynamic_indices: List[List[int]] = []
        self.dynamic_loss_scale: List[float] = []
        self.dynamic_optimizer_step: List[int] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))

        items = remove_padding_in_sequences(split_experience_batch(experience))
        self.items.extend(items)

        if self.limit > 0:
            overflow = len(self.items) - self.limit
            if overflow > 0:
                self.items = self.items[overflow:]

    @torch.no_grad()
    def extend(self, experiences: Iterable[Experience]) -> None:
        for exp in experiences:
            self.append(exp)

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        exp = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload:
            exp.to_device(self.target_device)
        return exp

    def __len__(self) -> int:
        return len(self.dynamic_indices) if self.dynamic_batch else len(self.items)

    def __getitem__(self, idx: int) -> Union[BufferItem, List[BufferItem]]:
        if self.dynamic_batch:
            return [self.items[i] for i in self.dynamic_indices[idx]]
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        if self.dynamic_batch:
            batch = batch[0]  # DataLoader batch_size=1; __getitem__ returns list[BufferItem]
        return make_experience_batch(batch, self.packing_samples)

    def setup_dynamic_batch(self, strategy) -> None:
        """Prepare indices/loss scaling for dynamic sequence-length batching."""
        args = strategy.args

        def _infer_total_length(sample) -> int:
            """
            Robust length inference for dynamic batching.
            Prefer tensor-backed fields (attention_mask / sequences) over info['total_length'].
            This avoids fragile dependence on warmup-cache / info slimming.
            """
            am = getattr(sample, "attention_mask", None)
            if isinstance(am, torch.Tensor):
                # per-sample BufferItem usually stores 1D tensors
                if am.dim() == 1:
                    return int(am.numel())
                # batched (rare here) -> use T
                if am.dim() >= 2:
                    return int(am.size(-1))

            seq = getattr(sample, "sequences", None)
            if isinstance(seq, torch.Tensor):
                if seq.dim() == 1:
                    return int(seq.numel())
                if seq.dim() >= 2:
                    return int(seq.size(-1))

            info = getattr(sample, "info", None)
            if isinstance(info, dict) and ("total_length" in info):
                v = info.get("total_length", 0)
                if isinstance(v, torch.Tensor):
                    return int(v.view(-1)[0].item()) if v.numel() > 0 else 0
                try:
                    return int(v)
                except Exception:
                    return 0

            raise RuntimeError(
                "[ReplayBuffer][FAIL-FAST] Cannot infer sequence length for dynamic batching: "
                "missing attention_mask/sequences and info['total_length']."
            )

        sample_lengths = [_infer_total_length(sample) for sample in self.items]

        if len(sample_lengths) == 0:
            raise RuntimeError(
                "Replay buffer is empty when setting up dynamic batch. "
                "This usually means rollout generation produced zero samples (or they were filtered out)."
            )

        world_size = dist.get_world_size()
        dp_size = world_size // args.ring_attn_size // args.ds_tensor_parallel_size
        dp_size = max(1, int(dp_size))
        local_train_batch_size = max(1, int(args.train_batch_size) // dp_size)

        num_steps = args.rollout_batch_size * args.n_samples_per_prompt // max(1, args.train_batch_size)
        if num_steps <= 0:
            num_steps = 1

        num_microbatches: List[int] = []
        for i in range(num_steps):
            start = i * local_train_batch_size
            if start >= len(sample_lengths):
                num_microbatches.append(1)
                continue

            end = min((i + 1) * local_train_batch_size, len(sample_lengths))
            step_lengths = sample_lengths[start:end]

            nmb = get_minimum_num_micro_batch_size(
                step_lengths,
                args.train_max_tokens_per_gpu,
                args.ring_attn_size,
                args.ds_tensor_parallel_size,
            )
            nmb = max(1, min(int(nmb), len(step_lengths)))
            num_microbatches.append(nmb)

        num_microbatches_t = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        num_microbatches_t = strategy.all_reduce(num_microbatches_t, op="max")
        num_microbatches = [max(1, int(x)) for x in num_microbatches_t.tolist()]

        micro_batch_indices: List[List[int]] = []
        data_partitions: List[List[List[int]]] = []

        for i, num_mbs in enumerate(num_microbatches):
            start = i * local_train_batch_size
            if start >= len(sample_lengths):
                break

            end = min((i + 1) * local_train_batch_size, len(sample_lengths))
            samples = sample_lengths[start:end]

            num_mbs = max(1, min(int(num_mbs), len(samples)))
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)

            for part in partitions:
                for k in range(len(part)):
                    part[k] += start

            micro_batch_indices.extend(partitions)
            data_partitions.append(partitions)

        self.dynamic_indices = micro_batch_indices
        self.sample_batch_size = 1

        loss_scales: List[float] = []
        optimizer_steps: List[int] = []

        for partitions in data_partitions:
            sample_num = sum(len(partition) for partition in partitions)
            loss_scale = [len(partition) / sample_num for partition in partitions]
            optimizer_step = [0] * (len(partitions) - 1) + [1]
            loss_scales.extend(loss_scale)
            optimizer_steps.extend(optimizer_step)

        self.dynamic_loss_scale = loss_scales
        self.dynamic_optimizer_step = optimizer_steps
