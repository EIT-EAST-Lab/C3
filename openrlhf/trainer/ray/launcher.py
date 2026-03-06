# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

import logging
import os
import math
import socket
from typing import Dict, Optional, Type

import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm

from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.trainer.ray.utils import ray_noset_visible_devices
from openrlhf.utils.deepspeed import DeepspeedStrategy


class BaseDistributedActor:
    def __init__(self, world_size, rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # Ray reports GPU ids as floats sometimes (e.g. [0.0]); normalize to an int-ish string.
        gpu_ids = ray.get_gpu_ids()
        if ray_noset_visible_devices() and gpu_ids:
            try:
                os.environ["LOCAL_RANK"] = str(int(gpu_ids[0]))
            except Exception:
                os.environ["LOCAL_RANK"] = "0"
        else:
            os.environ["LOCAL_RANK"] = "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        return address.strip("[]")

    @staticmethod
    def _is_port_available(port: int, addr: str = "0.0.0.0") -> bool:
        # NOTE: this is only a best-effort check; we still reduce collisions dramatically
        # by not relying on ephemeral ports chosen via bind(0) during concurrent startups.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((addr, port))
            except OSError:
                return False
        return True

    @staticmethod
    def _get_free_port():
        """Pick a rendezvous port for torch.distributed.

        Why not the usual "bind(0) -> close -> use"?
        ------------------------------------------------
        When multiple Ray actors start at the same time (e.g. actor/critic/q-critic
        singleton groups), the classic pattern can pick the same ephemeral port
        concurrently, and then all rank-0 processes race to create the TCPStore,
        leading to:

          DistNetworkError: EADDRINUSE (address already in use)

        Instead, we pick a deterministic starting point (based on PID) and probe
        sequentially for an available port. This makes collisions between actors
        extremely unlikely even under high concurrency.

        You can override the probe range via env vars:
          - OPENRLHF_MASTER_PORT_BASE (default: 29500)
          - OPENRLHF_MASTER_PORT_SPAN (default: 20000)
        """

        base = int(os.environ.get("OPENRLHF_MASTER_PORT_BASE", "29500"))
        if not (1024 <= base <= 65000):
            base = 29500

        span = int(os.environ.get("OPENRLHF_MASTER_PORT_SPAN", "20000"))
        span = max(1024, min(span, 30000))

        seed = (os.getpid() * 1103515245 + os.getppid() * 12345) & 0x7FFFFFFF
        start = base + (seed % span)

        # Probe forward from the deterministic start.
        for i in range(256):
            port = start + i
            if port > 65535:
                port = 1024 + (port % (65535 - 1024))
            if BaseDistributedActor._is_port_available(port):
                return port

        # Final fallback: use an OS-chosen ephemeral port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BaseModelActor(BaseDistributedActor):
    def _setup_distributed(self, strategy: DeepspeedStrategy):
        self.strategy = strategy
        # Ray can bring up multiple independent distributed groups concurrently
        # (e.g. policy / critic / q-critic). Even with careful port picking, a
        # singleton group (world_size=1) may still rarely hit EADDRINUSE if some
        # other process grabs the chosen port between "check" and "listen".
        # For world_size=1 we can safely retry with a new port.
        retries = int(os.environ.get("OPENRLHF_DIST_INIT_RETRIES", "20"))
        for attempt in range(max(1, retries)):
            try:
                strategy.setup_distributed()
                return
            except Exception as e:
                msg = str(e)
                is_addr_in_use = ("EADDRINUSE" in msg) or ("address already in use" in msg)
                if is_addr_in_use and self._world_size == 1 and attempt < retries - 1:
                    # Pick a new port and retry.
                    self._master_port = self._get_free_port()
                    os.environ["MASTER_PORT"] = str(self._master_port)
                    continue
                raise

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def execute_batch(self, method_name: str, all_data, start_idx, end_idx):
        kwargs = {key: value[start_idx:end_idx] for key, value in all_data.items()}
        first_param = next(iter(kwargs.values()))
        list_length = len(first_param)

        for param_name, param_value in kwargs.items():
            if len(param_value) != list_length:
                raise ValueError(f"Parameter {param_name} has length {len(param_value)}, expected {list_length}")

        func = getattr(self, method_name)
        if not callable(func):
            raise ValueError(f"Function {method_name} is not callable")

        # -------------------------
        # Generic batched fast-path
        # -------------------------
        # Any actor method can declare it supports batched list inputs by setting:
        #   func.__openrlhf_batched__ = True
        #
        # Contract:
        #   - execute_batch will call func(**kwargs) ONCE with list slices.
        #   - func may return:
        #       * None                -> treated as [None] * N
        #       * list/tuple len==N   -> returned directly
        #       * any other object    -> broadcast to [obj] * N
        try:
            is_batched = bool(getattr(func, "__openrlhf_batched__", False))
        except Exception:
            is_batched = False

        if is_batched:
            try:
                out = func(**kwargs)
                if out is None:
                    return [None] * list_length
                if isinstance(out, (list, tuple)):
                    if len(out) != list_length:
                        raise ValueError(
                            f"Batched method {method_name} returned len={len(out)} but expected {list_length}"
                        )
                    return list(out)
                return [out] * list_length
            except Exception:
                # Safe fallback: if batched execution fails for any reason, fall back to per-item calls.
                pass

        results = []
        for i in tqdm(range(list_length), desc=f"{method_name}", disable=not self.strategy.is_rank_0()):
            sample_kwargs = {param_name: param_value[i] for param_name, param_value in kwargs.items()}
            result = func(**sample_kwargs)
            results.append(result)

        return results


@ray.remote(num_gpus=1)
class ReferenceModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, role_name: Optional[str] = None):
        # role_name is accepted for per-role policy mode (kept for parity; reference is static)
        self.role_name = role_name or "shared"

        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                packed_seq_lens=packed_seq_lens,
            )
        return log_probs.to("cpu")


@ray.remote(num_gpus=1)
class RewardModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = get_llm_for_sequence_regression(
            pretrain,
            "reward",
            normalize_reward=strategy.args.normalize_reward,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            value_head_prefix=strategy.args.value_head_prefix,
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(model)
        strategy.print("reward normalization status: {}".format(strategy.args.normalize_reward))
        strategy.print("mean: {}, std {}".format(model.mean, model.std))

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
        pad_sequence=False,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            reward = self.model(
                sequences.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                pad_sequence=True,
                packed_seq_lens=packed_seq_lens,
            )
        return reward.to("cpu")


class RayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BaseModelActor],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        duplicate_actors: int = 1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        self.duplicate_actors = duplicate_actors

        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(self._num_nodes * self._num_gpus_per_node)]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(world_size, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, None, None)
        self._actor_handlers = [master_actor]

        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank,
                        ),
                    ).remote(world_size, rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(self, *args, **kwargs):
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_save_model(self):
        return [actor.save_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs

    def async_run_method_batch(self, method_name, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(value, "__len__"):
                raise ValueError(f"Parameter {key} must be iterable")

        first_param = next(iter(kwargs.values()))
        total_length = len(first_param)
        if total_length == 0:
            raise ValueError(f"[RayActorGroup.async_run_method_batch] method={method_name} returned an empty batch.")

        for key, value in kwargs.items():
            if len(value) != total_length:
                raise ValueError(
                    f"All parameters must have the same length. {key} has length {len(value)}, expected {total_length}"
                )

        # Effective actors are data-parallel partitions (duplicate_actors are TP/ring replicas).
        num_actors = len(self._actor_handlers)
        effective_actors = num_actors // self.duplicate_actors

        # In distributed training, all data-parallel ranks must participate.
        # If the produced micro-batch list is shorter than DP world size (common near epoch end
        # or after filtering), pad by repeating items so each rank gets >= 1 item.
        if total_length < effective_actors:
            pad_to = effective_actors
            for k, v in list(kwargs.items()):
                if not isinstance(v, list):
                    # execute_batch assumes list-like params; keep behavior strict.
                    continue
                if len(v) != total_length:
                    raise ValueError(
                        f"[RayActorGroup.async_run_method_batch] inconsistent batch lens: "
                        f"{k} has {len(v)} != {total_length}"
                    )
                if len(v) == 0:
                    raise ValueError(
                        f"[RayActorGroup.async_run_method_batch] method={method_name} has empty list param: {k}"
                    )
                need = pad_to - len(v)
                if need > 0:
                    # cycle-pad (minimizes bias vs repeating only last item)
                    v2 = list(v) + [v[i % len(v)] for i in range(need)]
                    kwargs[k] = v2
            total_length = pad_to

        all_data_ref = ray.put(kwargs)

        refs = []
        # Distribute tasks to actors
        for chunk_idx in range(effective_actors):
            # balanced partition: guarantees each rank gets >=1 item when total_length >= effective_actors
            start_idx = (chunk_idx * total_length) // effective_actors
            end_idx = ((chunk_idx + 1) * total_length) // effective_actors

            for j in range(self.duplicate_actors):
                actor_idx = chunk_idx * self.duplicate_actors + j
                actor = self._actor_handlers[actor_idx]
                refs.append(actor.execute_batch.remote(method_name, all_data_ref, start_idx, end_idx))

        return refs
