# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

import math
import os
import re
import socket
from abc import ABC
from typing import Dict, List, Optional, Union

import deepspeed
import ray
import torch
import torch.distributed
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.models import Actor, PolicyLoss
from openrlhf.models.utils import compute_approx_kl, masked_mean
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.distributed_util import stateless_init_process_group, torch_dist_barrier_and_cuda_sync
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import safe_get_global_grad_norm

from ..ppo_utils import NaiveReplayBuffer
from .launcher import BaseModelActor
from .utils import get_physical_gpu_id

logger = init_logger(__name__)


# =============================================================================
# Small helpers
# =============================================================================
def _sanitize_role_name(role_name: Optional[str]) -> str:
    rn = (role_name or "shared").strip() or "shared"
    return re.sub(r"[^0-9A-Za-z_]+", "_", rn)


def _group_name_for_role(role_name: str) -> str:
    return "openrlhf" if role_name == "shared" else f"openrlhf_{role_name}"


def _dist_is_init() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


def _dist_rank() -> int:
    return int(torch.distributed.get_rank()) if _dist_is_init() else 0


def _dist_world_size() -> int:
    return int(torch.distributed.get_world_size()) if _dist_is_init() else 1


# =============================================================================
# Actor-side PPO trainer (runs inside PolicyModelActor)
# =============================================================================
class ActorPPOTrainer(ABC):
    def __init__(
        self,
        strategy: DeepspeedStrategy,
        actor: Actor,
        ema_model: Optional[Actor],
        actor_optim: Optional[Optimizer],
        actor_scheduler: Optional[object],
        ema_beta: float = 0.992,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        tokenizer=None,
        dataloader_pin_memory: bool = True,
        vllm_engines: Optional[List] = None,
        role_name: Optional[str] = None,
        **kwargs,
    ):
        """PPO trainer (Ray actor side)."""
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.generate_kwargs = kwargs

        self.dataloader_pin_memory = dataloader_pin_memory
        self.micro_train_batch_size = int(micro_train_batch_size)
        self.ema_beta = float(ema_beta)
        self.role_name = _sanitize_role_name(role_name)

        self.actor = actor
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler

        self.vllm_engines = vllm_engines
        self.max_epochs = int(getattr(self.args, "max_epochs", 1) or 1)

        # REINFORCE: avoid reusing the same batch across epochs (off-policy drift).
        if str(getattr(self.args, "policy_loss_type", "ppo") or "ppo").lower().strip() == "reinforce":
            self.max_epochs = 1

        self.actor_loss_fn = PolicyLoss(
            clip_eps_low=self.args.eps_clip_low_high[0],
            clip_eps_high=self.args.eps_clip_low_high[1],
            dual_clip=self.args.dual_clip,
            policy_loss_type=self.args.policy_loss_type,
            enable_vllm_is_correction=self.args.enable_vllm_is_correction,
            vllm_is_truncated_threshold=(
                self.args.vllm_is_truncated_threshold if self.args.enable_vllm_is_correction else None
            ),
            use_icepop=self.args.use_icepop,
        )

        self.aux_loss = bool(self.args.aux_loss_coef > 1e-8)

        self.replay_buffer = NaiveReplayBuffer(
            self.micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            getattr(self.args, "packing_samples", False),
            self.args.use_dynamic_batch,
        )

        self._model_update_group = None
        self.use_cuda_ipc = self._should_use_cuda_ipc()
        self._maybe_init_vllm_sync_group()

        torch_dist_barrier_and_cuda_sync()

    # -------------------------------------------------------------------------
    # vLLM weight sync setup
    # -------------------------------------------------------------------------
    def _should_use_cuda_ipc(self) -> bool:
        """Prefer CUDA-IPC in colocate mode to avoid NCCL invalid usage."""
        if str(os.getenv("OPENRLHF_DISABLE_CUDA_IPC", "0")).lower() in {"1", "true", "yes"}:
            return False

        backend = str(getattr(self.strategy.args, "vllm_sync_backend", "nccl") or "nccl").lower().strip()
        return bool(backend == "nccl" and self.args.colocate_all_models and not self.args.async_train)

    def _maybe_init_vllm_sync_group(self) -> None:
        # Rank0 creates the sync group for collective weight broadcast (non-IPC).
        if self.vllm_engines is None or self.use_cuda_ipc:
            return
        if _dist_rank() != 0:
            return

        backend = str(getattr(self.strategy.args, "vllm_sync_backend", "nccl") or "nccl").lower().strip()
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        tp = int(getattr(self.strategy.args, "vllm_tensor_parallel_size", 1) or 1)
        num_engines = int(len(self.vllm_engines))
        world_size = num_engines * tp + 1

        use_ray = bool(getattr(self.strategy.args, "vllm_sync_with_ray", False))
        group_name = _group_name_for_role(self.role_name)

        refs = []
        for i, engine in enumerate(self.vllm_engines):
            refs.append(
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * tp + 1,
                    world_size,
                    group_name,
                    backend=backend,
                    use_ray=use_ray,
                )
            )

        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = stateless_init_process_group(
                master_address, master_port, 0, world_size, torch.cuda.current_device()
            )

        ray.get(refs)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def ppo_train(self, kl_ctl: float):
        if self.actor_optim is None or self.actor_scheduler is None:
            raise RuntimeError(
                "ActorPPOTrainer.ppo_train() called without an optimizer/scheduler. "
                "This is expected in eval_only runs; do not call training methods."
            )

        if self.args.use_dynamic_batch:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        not_shuffle = bool(
            self.strategy.ring_attn_group is not None
            or self.args.ds_tensor_parallel_size > 1
            or self.args.use_dynamic_batch
        )

        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=not not_shuffle,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list: List[Dict[str, float]] = []
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for step, experience in enumerate(pbar):
                experience.to_device(device)
                status = self.training_step(experience, kl_ctl, step)

                # Normalize KL by generated tokens (for logging).
                status["kl"] *= status["response_length"]
                status = self.strategy.all_reduce(status)
                status["kl"] /= status["response_length"]

                short_status = {
                    "act_loss": status.get("policy_loss", 0.0),
                    "reward": status.get("reward", 0.0),
                    "return": status.get("return", 0.0),
                    "gen_len": status.get("response_length", 0.0),
                    "tot_len": status.get("total_length", 0.0),
                    "kl": status.get("kl", 0.0),
                    "act_lr": status.get("actor_lr", 0.0),
                }
                if "entropy_loss" in status:
                    short_status["ent_loss"] = status["entropy_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if not status_list:
            return {}

        status_mean = dict(status_list[0])
        for m in status_list[1:]:
            for k, v in m.items():
                status_mean[k] += v
        for k in list(status_mean.keys()):
            status_mean[k] /= float(len(status_list))
        return status_mean

    def training_step(self, experience: Experience, kl_ctl: float, step: int) -> Dict[str, float]:
        self.actor.train()

        sequences = experience.sequences
        action_mask = experience.action_mask
        attention_mask = experience.attention_mask
        packed_seq_lens = None

        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs

        action_log_probs, output = self.actor(
            sequences,
            action_mask,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
            return_entropy=self.args.entropy_loss_coef is not None,
        )

        actor_loss, clip_ratio, ppo_kl, vllm_kl = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            rollout_log_probs=experience.rollout_log_probs,
        )
        experience.info["ppo_clip_ratio"] = clip_ratio.detach()
        experience.info["ppo_kl"] = ppo_kl.detach()
        if vllm_kl is not None:
            experience.info["vllm_kl"] = vllm_kl.detach()

        if self.args.use_kl_loss:
            if self.args.init_kl_coef > 0:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)
            kl_loss = masked_mean(kl, experience.action_mask)
            experience.info["kl"] = kl_loss.detach()
        else:
            kl_loss = 0

        loss = actor_loss + kl_loss * float(kl_ctl)

        if self.aux_loss:
            loss += output.aux_loss * self.args.aux_loss_coef

        entropy_loss = None
        if self.args.entropy_loss_coef is not None:
            entropy_tail = output.entropy[:, -experience.action_mask.shape[1] :]
            entropy_loss = masked_mean(entropy_tail, experience.action_mask)
            if self.args.entropy_loss_coef != 0:
                loss -= entropy_loss * self.args.entropy_loss_coef

        if self.args.use_dynamic_batch:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]

        self.strategy.backward(loss, self.actor, self.actor_optim)

        actor_gn = safe_get_global_grad_norm(self.actor)

        if self.args.use_dynamic_batch:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        else:
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            if self.args.use_dynamic_batch:
                if self.replay_buffer.dynamic_optimizer_step[step]:
                    self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
            else:
                self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        status: Dict[str, float] = {
            "policy_loss": float(actor_loss.detach().item()),
            "actor_lr": float(self.actor_scheduler.get_last_lr()[0]),
            "actor/grad_norm": float(actor_gn),
        }
        if entropy_loss is not None:
            status["entropy_loss"] = float(entropy_loss.detach().item())

        # Reduce numeric info only (C3 metadata may include dict/list/strings).
        info = experience.info or {}
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                try:
                    status[k] = float(v.float().mean().item())
                except Exception:
                    pass
                continue
            if isinstance(v, (int, float, bool)):
                status[k] = float(v)
                continue
            if isinstance(v, list) and v:
                try:
                    t = torch.as_tensor(v, dtype=torch.float32)
                    status[k] = float(t.mean().item())
                    continue
                except Exception:
                    pass
                try:
                    if all(isinstance(x, torch.Tensor) for x in v):
                        t = torch.stack([x.float().mean() for x in v])
                        status[k] = float(t.mean().item())
                except Exception:
                    pass

        return status

    # -------------------------------------------------------------------------
    # vLLM weight broadcast
    # -------------------------------------------------------------------------
    def _broadcast_to_vllm(self, weights_version=None) -> None:
        """Sync policy weights into vLLM workers.

        Modes:
          - CUDA-IPC (colocate): share CUDA storages, no trainer+vLLM NCCL group.
          - Collective (non-colocate): rank0 broadcasts params; vLLM engines receive in one task/engine.
        """
        if self.vllm_engines is None:
            torch_dist_barrier_and_cuda_sync()
            return

        use_prefix_cache = bool(getattr(self.strategy.args, "enable_prefix_caching", False))
        vllm_enable_sleep = bool(getattr(self.strategy.args, "vllm_enable_sleep", False))
        timeout_s = float(getattr(self.strategy.args, "vllm_update_weight_timeout_s", 600.0))
        rank = _dist_rank()

        def _wait_refs(refs, *, name: str, idx: int, total: int) -> None:
            if not refs:
                return
            ready, not_ready = ray.wait(refs, num_returns=len(refs), timeout=timeout_s)
            if ready:
                ray.get(ready)
            if not_ready:
                raise TimeoutError(
                    f"[vLLM][weights] timeout after {timeout_s:.1f}s at {idx}/{total}: {name} "
                    f"(not_ready={len(not_ready)})"
                )

        def _wake_up_engines_best_effort() -> None:
            if not (vllm_enable_sleep and rank == 0):
                return
            try:
                from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

                batch_vllm_engine_call(self.vllm_engines, "wake_up")
            except Exception:
                pass

        _wake_up_engines_best_effort()
        torch.cuda.empty_cache()

        model = self.actor.model.module
        named_params = list(model.named_parameters())
        num_params = int(len(named_params))
        if num_params == 0:
            torch_dist_barrier_and_cuda_sync()
            return

        # -----------------------
        # Collective (non-IPC)
        # -----------------------
        if not self.use_cuda_ipc:
            if rank != 0:
                # Only rank0 participates in the trainer<->vLLM collective group.
                torch_dist_barrier_and_cuda_sync()
                return
            if self._model_update_group is None:
                raise RuntimeError("vLLM sync group not initialized (expected non-IPC mode).")

            use_ray_collective = bool(getattr(self.strategy.args, "vllm_sync_with_ray", False))

            # One long receiver loop per engine (NOT per-parameter tasks).
            meta_names, meta_dtypes, meta_shapes = [], [], []
            for n, p in named_params:
                sh = (
                    p.ds_shape
                    if (getattr(self.strategy.args, "zero_stage", 0) == 3 and hasattr(p, "ds_shape"))
                    else p.shape
                )
                meta_names.append(n)
                meta_dtypes.append(p.dtype)
                meta_shapes.append(sh)

            update_refs = [
                engine.update_weights.remote(meta_names, meta_dtypes, meta_shapes, empty_cache_last=True)
                for engine in self.vllm_engines
            ]

            def _broadcast_tensor_(t: torch.Tensor) -> None:
                if use_ray_collective:
                    import ray.util.collective as collective

                    collective.broadcast(t, src_rank=0, group_name=self._model_update_group)
                else:
                    self._model_update_group.broadcast(t, src=0, stream=torch.cuda.current_stream())

            for idx, (_name, param) in enumerate(named_params, start=1):
                if self.strategy.args.ds_tensor_parallel_size > 1:
                    gather_ctx = deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True)
                else:
                    gather_ctx = deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3)

                with gather_ctx:
                    _broadcast_tensor_(param.data)

            _wait_refs(update_refs, name="update_weights", idx=num_params, total=num_params)

        # -----------------------
        # CUDA-IPC (colocate)
        # -----------------------
        else:
            from torch.multiprocessing.reductions import reduce_tensor

            chunk_size = int(getattr(self.strategy.args, "vllm_ipc_chunk_size", 32) or 32)
            world_size = _dist_world_size()

            def _gather_object_to_rank0(obj):
                if not _dist_is_init() or world_size <= 1:
                    return [obj]
                # Prefer gather_object (dst only stores list), fallback to all_gather_object.
                if hasattr(torch.distributed, "gather_object"):
                    gathered = [None] * world_size if rank == 0 else None
                    try:
                        torch.distributed.gather_object(obj, gathered, dst=0)
                        return gathered
                    except Exception:
                        pass
                gathered2 = [None] * world_size
                try:
                    torch.distributed.all_gather_object(gathered2, obj)
                    return gathered2
                except Exception:
                    return [obj]

            for start in range(0, num_params, chunk_size):
                chunk = named_params[start : start + chunk_size]
                last_chunk = (start + len(chunk)) >= num_params

                chunk_names: List[str] = []
                chunk_dtypes: List[torch.dtype] = []
                chunk_shapes: List[object] = []
                chunk_params: List[torch.nn.Parameter] = []
                for n, p in chunk:
                    chunk_names.append(n)
                    chunk_dtypes.append(p.dtype)
                    sh = p.ds_shape if (self.strategy.args.zero_stage == 3 and hasattr(p, "ds_shape")) else p.shape
                    chunk_shapes.append(sh)
                    chunk_params.append(p)

                if self.strategy.args.ds_tensor_parallel_size > 1:
                    gather_ctx = deepspeed.module_inject.layers.GatherReplacedLayerParams(
                        chunk_params, model, enabled=True
                    )
                else:
                    gather_ctx = deepspeed.zero.GatheredParameters(chunk_params, enabled=self.strategy.args.zero_stage == 3)

                with gather_ctx:
                    local_gpu_id = get_physical_gpu_id()
                    local_handles: List[Dict[int, object]] = []
                    tmp_tensors: List[torch.Tensor] = []  # keep contiguous copies alive until update completes

                    for _n, p in chunk:
                        w = p.data.detach()
                        if not w.is_contiguous():
                            w = w.contiguous()
                            tmp_tensors.append(w)
                        local_handles.append({local_gpu_id: reduce_tensor(w)})

                    gathered = _gather_object_to_rank0(local_handles)

                    if rank == 0:
                        merged_per_param: List[Dict[int, object]] = []
                        for j in range(len(chunk)):
                            merged: Dict[int, object] = {}
                            for per_rank_list in gathered:
                                if not per_rank_list:
                                    continue
                                merged.update(per_rank_list[j])
                            merged_per_param.append(merged)

                        refs = [
                            engine.update_weights_cuda_ipc.remote(
                                chunk_names,
                                chunk_dtypes,
                                chunk_shapes,
                                merged_per_param,
                                empty_cache_last=last_chunk,
                            )
                            for engine in self.vllm_engines
                        ]
                        _wait_refs(
                            refs,
                            name=f"update_weights_cuda_ipc[{start}:{start+len(chunk)}]",
                            idx=start + len(chunk),
                            total=num_params,
                        )

                torch_dist_barrier_and_cuda_sync()

        # Prefix cache depends on weights.
        if use_prefix_cache and rank == 0:
            try:
                from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

                batch_vllm_engine_call(self.vllm_engines, "reset_prefix_cache")
            except Exception:
                pass

        torch.cuda.empty_cache()

        # Version bookkeeping (optional).
        if weights_version is not None and rank == 0:
            version = int(weights_version)
            ray.get([engine.set_weights_version.remote(version) for engine in self.vllm_engines])

            strict = bool(getattr(self.args, "strict_weights_version_check", False))
            if strict:
                got_versions = ray.get([engine.get_weights_version.remote() for engine in self.vllm_engines])
                if any(int(v) != version for v in got_versions):
                    raise RuntimeError(
                        f"[role={self.role_name}] vLLM weights_version mismatch: expected={version}, got={got_versions}."
                    )
            else:
                try:
                    got_versions = ray.get([engine.get_weights_version.remote() for engine in self.vllm_engines])
                    if any(int(v) != version for v in got_versions):
                        logger.warning(
                            "[role=%s] vLLM weights_version mismatch (strict disabled): expected=%d got=%s",
                            self.role_name,
                            version,
                            got_versions,
                        )
                except Exception:
                    pass

        torch_dist_barrier_and_cuda_sync()


# =============================================================================
# Ray actor: policy model owner (one per policy replica / per role)
# =============================================================================
@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    # ---- Path helpers (per-role isolation) ----
    def _actor_ckpt_subdir(self) -> str:
        rn = getattr(self, "role_name", "shared")
        return "_actor" if rn == "shared" else f"_actor_{rn}"

    def _hf_ckpt_subdir(self, tag: str) -> str:
        rn = getattr(self, "role_name", "shared")
        return f"{tag}_hf" if rn == "shared" else f"{tag}_hf_{rn}"

    def _final_save_dir(self, base_dir: str) -> str:
        if getattr(self.strategy.args, "policy_sharing_mode", "shared") == "per_role" and self.role_name != "shared":
            return os.path.join(base_dir, f"actor_{self.role_name}")
        return base_dir

    # ---- Lifecycle ----
    def init_model_from_pretrained(
        self,
        strategy: DeepspeedStrategy,
        pretrain,
        max_steps=None,
        vllm_engines=None,
        role_name: Optional[str] = None,
    ):
        args = strategy.args
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps
        self.role_name = _sanitize_role_name(role_name)

        if getattr(args, "vllm_num_engines", 0) > 0:
            # Avoid NCCL hang in certain weight sync topologies.
            if str(getattr(args, "vllm_sync_backend", "nccl") or "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        eval_only = bool(getattr(args, "eval_only", False))

        actor_ds_config = strategy.get_ds_eval_config(offload=False) if eval_only else strategy.get_ds_train_config(is_actor=True)
        actor = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=actor_ds_config,
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(actor)

        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )

        # In eval_only, do NOT build optimizer/LR-scheduler/EMA. They allocate large states (esp. ZeRO)
        # and can prevent colocated vLLM from starting.
        ema_model = None
        actor_optim = None
        actor_scheduler = None

        if not eval_only:
            if args.enable_ema:
                ema_model = Actor(
                    pretrain,
                    attn_implementation=strategy.args.attn_implementation,
                    bf16=strategy.args.bf16,
                    load_in_4bit=strategy.args.load_in_4bit,
                    ds_config=strategy.get_ds_eval_config(offload=True),
                    packing_samples=strategy.args.packing_samples,
                )

            actor_optim = strategy.create_optimizer(
                actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
            )
            actor_scheduler = get_scheduler(
                args.lr_scheduler,
                actor_optim,
                num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
                num_training_steps=max_steps,
                scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
            )

            if args.gradient_checkpointing:
                actor.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
                )

            self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
                (actor, actor_optim, actor_scheduler),
                is_rlhf=True,
            )

            if ema_model:
                ema_model._offload = True
                self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
            else:
                self.ema_model = None
        else:
            # eval_only: deepspeed eval init only (no optimizer states).
            self.actor = strategy.prepare(actor, is_rlhf=True)
            self.actor_optim = None
            self.actor_scheduler = None
            self.ema_model = None

        # Load checkpoint (per-role isolation).
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt_path, self._actor_ckpt_subdir())
        if args.load_checkpoint and os.path.exists(ckpt_path):
            strategy.print(f"[role={self.role_name}] Loading the checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states["global_step"] = states["global_step"]
            self.checkpoint_states["episode"] = states["episode"]
            self.checkpoint_states["data_loader_state_dict"] = states["data_loader_state_dict"]

        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            tokenizer=self.tokenizer,
            eps_clip=args.eps_clip,
            ema_beta=args.ema_beta,
            vllm_engines=self.vllm_engines,
            role_name=self.role_name,
        )

    # ---- Train / IO ----
    def fit(self, kl_ctl: float = 0):
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.ppo_train(kl_ctl)
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args
        save_dir = self._final_save_dir(args.save_path)
        model_to_save = self.ema_model if bool(getattr(args, "enable_ema", False)) else self.actor
        # eval_only normalization may disable EMA after the actor is built.
        if model_to_save is None:
            model_to_save = self.actor
        self.strategy.save_model(
            model_to_save,
            self.tokenizer,
            save_dir,
        )

    # ---- RPC methods used by ExperienceMaker / rollout sync ----
    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        self.actor.eval()
        with torch.no_grad():
            action_log_probs = self.actor(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
            )
        self.actor.train()
        return action_log_probs.to("cpu")

    def broadcast_to_vllm(self, weights_version=None):
        self.trainer._broadcast_to_vllm(weights_version=weights_version)

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def append(self, experience: Experience):
        """Append experience(s) to replay buffer (single or list)."""
        rb = self.trainer.replay_buffer
        if isinstance(experience, list):
            try:
                rb.extend(experience)
            except Exception:
                for e in experience:
                    rb.append(e)
        else:
            rb.append(experience)

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def save_checkpoint(self, tag, client_states):
        args = self.strategy.args
        ckpt_dir = os.path.join(args.ckpt_path, self._actor_ckpt_subdir())

        self.strategy.save_ckpt(
            self.actor.model,
            ckpt_dir,
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )

        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, self._hf_ckpt_subdir(tag))
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )

        torch_dist_barrier_and_cuda_sync()


# Mark append as batched-capable for BaseModelActor.execute_batch.
try:
    PolicyModelActor.append.__openrlhf_batched__ = True
except Exception:
    pass
