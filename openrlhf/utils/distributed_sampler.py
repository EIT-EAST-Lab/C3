# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

import math
from typing import Iterator, Optional

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

__all__ = ["DistributedSampler"]


class DistributedSampler(Sampler[int]):
    """
    Distributed sampler (PyTorch DistributedSampler-style) with resume support.

    - Splits dataset indices across ranks (num_replicas, rank).
    - Optional deterministic shuffle per-epoch via (seed + epoch).
    - Supports resuming within an epoch via `consumed_samples`.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        consumed_samples: int = 0,
    ) -> None:
        self.dataset = dataset

        self.num_replicas, self.rank = self._resolve_dist_args(num_replicas, rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self.epoch = 0

        # Compute per-rank sample count (same semantics as upstream).
        ds_len = len(self.dataset)  # type: ignore[arg-type]
        if self.drop_last and (ds_len % self.num_replicas != 0):
            # nearest length evenly divisible by num_replicas
            self.num_samples = math.ceil((ds_len - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(ds_len / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

        # NOTE: keep the original misspelling for backward compat.
        self.consumed_indicies = int(consumed_samples) // self.num_replicas

    @staticmethod
    def _resolve_dist_args(
        num_replicas: Optional[int],
        rank: Optional[int],
    ) -> tuple[int, int]:
        if num_replicas is None or rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            # Keep behavior close to original; if dist isn't initialized, these may throw.
            if num_replicas is None:
                num_replicas = dist.get_world_size()
            if rank is None:
                rank = dist.get_rank()

        num_replicas_i = int(num_replicas)
        rank_i = int(rank)
        if rank_i < 0 or rank_i >= num_replicas_i:
            raise ValueError(f"Invalid rank {rank_i}, rank should be in the interval [0, {num_replicas_i - 1}]")
        return num_replicas_i, rank_i

    def _base_indices(self) -> list[int]:
        ds_len = len(self.dataset)  # type: ignore[arg-type]
        if not self.shuffle:
            return list(range(ds_len))

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return torch.randperm(ds_len, generator=g).tolist()

    def _evenly_divisible_indices(self, indices: list[int]) -> list[int]:
        # Pad or truncate to `total_size` (same as upstream semantics).
        if self.drop_last:
            return indices[: self.total_size]

        padding_size = self.total_size - len(indices)
        if padding_size <= 0:
            return indices

        if padding_size <= len(indices):
            return indices + indices[:padding_size]

        # If dataset is tiny, repeat as needed.
        repeat = math.ceil(padding_size / len(indices))
        return indices + (indices * repeat)[:padding_size]

    def __iter__(self) -> Iterator[int]:
        indices = self._evenly_divisible_indices(self._base_indices())
        assert len(indices) == self.total_size

        # Subsample for this rank.
        indices = indices[self.rank : self.total_size : self.num_replicas]

        # Skip already-consumed samples (per-rank).
        indices = indices[self.consumed_indicies :]
        assert len(indices) == self.num_samples - self.consumed_indicies

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.consumed_indicies

    def set_epoch(self, epoch: int, consumed_samples: int = 0) -> None:
        """Set epoch (affects shuffle order) and consumed progress within the epoch."""
        self.epoch = int(epoch)
        self.consumed_indicies = int(consumed_samples) // self.num_replicas

    # ------------------------------------------------------------------
    # Make sampler stateful + lightweight for torchdata StatefulDataLoader
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """
        IMPORTANT: Do NOT serialize self.dataset.
        Store only minimal info needed to reproduce the index stream.
        """
        return {
            "epoch": int(self.epoch),
            "seed": int(self.seed),
            "shuffle": bool(self.shuffle),
            "drop_last": bool(self.drop_last),
            "num_replicas": int(self.num_replicas),
            "rank": int(self.rank),
            # keep the original misspelling for backward compat
            "consumed_indicies": int(self.consumed_indicies),
        }

    def load_state_dict(self, state: dict) -> None:
        """Tolerant loader: ignore unknown keys and mismatched topology."""
        if not isinstance(state, dict):
            return

        if "epoch" in state:
            try:
                self.epoch = int(state["epoch"])
            except Exception:
                pass
        if "seed" in state:
            try:
                self.seed = int(state["seed"])
            except Exception:
                pass
        if "shuffle" in state:
            try:
                self.shuffle = bool(state["shuffle"])
            except Exception:
                pass
        if "drop_last" in state:
            try:
                self.drop_last = bool(state["drop_last"])
            except Exception:
                pass

        # consumed index (per-rank)
        ci = state.get("consumed_indicies", None)
        if ci is None:
            ci = state.get("consumed_indices", None)  # tolerate common spelling

        if ci is None:
            # If checkpoint stored global consumed_samples, convert to per-rank.
            cs = state.get("consumed_samples", None)
            if cs is not None:
                try:
                    ci = int(cs) // int(self.num_replicas)
                except Exception:
                    ci = None

        if ci is not None:
            try:
                self.consumed_indicies = max(0, int(ci))
            except Exception:
                pass
