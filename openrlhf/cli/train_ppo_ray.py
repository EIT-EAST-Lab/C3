# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import ray
from ray.util.placement_group import placement_group

from c3.integration.marl_specs import load_task
from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.trainer.ray.launcher import RayActorGroup, ReferenceModelActor, RewardModelActor
from openrlhf.trainer.ray.ppo_actor import PolicyModelActor
from openrlhf.trainer.ray.ppo_critic import CriticModelActor, QCriticModelActor
from openrlhf.utils import get_strategy
from openrlhf.utils.run_metadata import init_run_artifacts

# Tooling extracted into a sibling module.
from openrlhf.cli.train_ppo_ray_tooling import (
    _VLLM_AUTO_ENGINES_SENTINEL,  # noqa: F401 (kept for compatibility)
    _detect_total_gpu_count,
    _ensure_ray_initialized,
    _normalize_choice,
    _resolve_pretrain_by_role,
    _setup_local_run_logging,
    build_parser,
    normalize_args,
)


# =========================
# Plan data structures
# =========================


@dataclass(frozen=True)
class VLLMPlan:
    enabled: bool
    num_engines: int
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: Optional[float]
    use_hybrid_engine: bool
    internal_pg_created_by_vllm: bool
    executor_backend: str
    ray_actor_num_gpus_request: float
    request_batch_size: int


@dataclass(frozen=True)
class ModelGroupsPlan:
    create_actor: bool
    create_ref: bool
    create_critic: bool
    create_reward: bool
    create_q_critic: bool


@dataclass(frozen=True)
class PlacementPlan:
    colocate_all_models: bool
    colocate_actor_ref: bool
    colocate_critic_reward: bool
    pg_all_bundles: Optional[int]
    pg_actor_ref_bundles: Optional[int]
    pg_critic_reward_bundles: Optional[int]
    pg_per_role_bundles: Optional[int]
    pg_shared_extra_bundles: Optional[int]


@dataclass(frozen=True)
class ExecutionPlan:
    policy_sharing_mode: str  # shared | per_role
    role_names: Optional[List[str]]
    pretrain_by_role: Optional[Dict[str, str]]
    role_count: int

    eval_only: bool
    async_train: bool

    marl_algorithm: str
    mappo_normalize_scope: str
    c3_task: Optional[str]
    is_c3_mas: bool

    use_env_reward: bool
    reward_source: str

    advantage_estimator: str
    init_kl_coef: float

    shared_gpu_fraction_models: float
    placement: PlacementPlan
    vllm: VLLMPlan
    models: ModelGroupsPlan

    run_dir: Optional[str]
    run_id: Optional[str]

    dump_first_c3_batch: Optional[str]
    dump_first_c3_batch_fields: str
    dump_first_c3_batch_any: bool
    dump_rollouts_jsonl_path: Optional[str]
    dump_rollouts_every: int
    dump_c3_batch_data_path: Optional[str]
    eval_dump_path: Optional[str]
    eval_dump_mode: str

    notes: List[str] = field(default_factory=list)


# =========================
# Args validation
# =========================


def validate_args(args, ctx: Dict[str, object]) -> None:
    # Required args
    if args.pretrain is None:
        raise SystemExit("ERROR: missing required argument: --pretrain <HF_MODEL_OR_PATH>")

    # Project constraint: vLLM must always be enabled (shared and per_role).
    if not (getattr(args, "vllm_num_engines", 0) and int(args.vllm_num_engines) > 0):
        raise SystemExit("ERROR: vLLM must be enabled: please set --vllm_num_engines > 0.")

    policy_mode = getattr(args, "policy_sharing_mode", "shared")

    # per_role guards
    if policy_mode == "per_role":
        if getattr(args, "async_train", False):
            raise ValueError("per_role policy暂不支持 --async_train（请先关闭）。")
        if not getattr(args, "c3_task", None):
            raise ValueError("per_role policy requires --c3_task to infer role topo order.")

    # async + sleep is unsupported
    if args.async_train:
        assert not args.vllm_enable_sleep, "Async RLHF is not supported with --vllm_enable_sleep."

    # packing_samples requires vLLM
    if args.packing_samples:
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."

    if args.ring_attn_size > 1:
        assert args.packing_samples, "--ring_attn_size > 1 requires --packing_samples."

    # RewardProvider + reward source fail-fast
    is_c3_mas = bool(getattr(args, "rollout_generator_cls", None) and getattr(args, "c3_task", None))
    provider_raw = (getattr(args, "reward_provider_cls", "auto") or "auto").strip()
    provider_builtin = provider_raw.lower() if ":" not in provider_raw else None
    builtin = {"auto", "chain", "env", "remote_rm", "none"}

    if is_c3_mas and provider_builtin in builtin:
        from c3.envs import SUPPORTED_ENVS

        env_ok = bool(getattr(args, "use_env_reward", False))
        rm_ok = bool(getattr(args, "remote_rm_url", None) or getattr(args, "agent_func_path", None))

        if provider_builtin == "none":
            raise SystemExit("ERROR: C3 MAS rollouts require rewards; --reward_provider_cls=none will always fail.")

        if provider_builtin == "env" and not env_ok:
            raise SystemExit(
                f"ERROR: reward_provider_cls=env requires env reward, but env_name={args.c3_env_name!r} "
                f"is not supported (supported={SUPPORTED_ENVS})."
            )

        if provider_builtin == "remote_rm" and not rm_ok:
            raise SystemExit("ERROR: reward_provider_cls=remote_rm requires --remote_rm_url or --agent_func_path.")

        if provider_builtin in {"auto", "chain"} and not (env_ok or rm_ok):
            raise SystemExit(
                "ERROR: reward_provider_cls=auto/chain requires either env reward (supported C3 env) "
                "or --remote_rm_url/--agent_func_path. "
                "Note: --reward_pretrain does NOT provide rewards for C3 RewardProvider."
            )

    if is_c3_mas and provider_builtin in builtin:
        has_reward_source = bool(args.remote_rm_url or args.agent_func_path)
    else:
        has_reward_source = bool(args.reward_pretrain or args.remote_rm_url or args.agent_func_path)

    if not has_reward_source and not args.use_env_reward:
        if getattr(args, "c3_env_name", None) is not None and is_c3_mas and provider_builtin in builtin:
            raise SystemExit(
                "ERROR: C3 MAS requires RewardProvider to produce rewards. "
                "Provide env reward (supported env) or --remote_rm_url/--agent_func_path. "
                "Note: --reward_pretrain does NOT provide rewards for C3 RewardProvider."
            )
        if getattr(args, "c3_env_name", None) is not None:
            raise SystemExit(
                "ERROR: missing required reward source: set --reward_pretrain <HF_MODEL_OR_PATH> "
                f"(or --remote_rm_url / --agent_func_path). Note: C3 task env_name={args.c3_env_name!r} "
                "is not in the supported env-reward list, so we cannot skip reward_pretrain."
            )
        raise SystemExit(
            "ERROR: missing required reward source: set --reward_pretrain <HF_MODEL_OR_PATH> "
            "(or --remote_rm_url / --agent_func_path)."
        )

    if args.use_env_reward and not has_reward_source:
        print(
            f"[Info] C3 env-reward mode enabled (env_name={args.c3_env_name}). "
            "Reward model will be disabled; rollout generator must provide Experience.rewards."
        )

    # K-dependent MARL algorithms (only enforce when MAS rollouts are enabled)
    if is_c3_mas and args.marl_algorithm in {"c3", "magrpo"}:
        k = int(getattr(args, "n_samples_per_prompt", 1) or 1)
        if k <= 1:
            raise SystemExit(
                f"ERROR: marl_algorithm={args.marl_algorithm} requires --n_samples_per_prompt >= 2 (got {k}). "
                "Reason: GRPO-family/group-based credit & advantage need multiple samples per prompt."
            )

    if args.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm"]:
        assert args.n_samples_per_prompt > 1, f"{args.advantage_estimator} requires n_samples_per_prompt > 1"

    if args.eval_dataset and not bool(getattr(args, "use_env_reward", False)):
        assert args.remote_rm_url, "`--eval_dataset` is only supported with `--remote_rm_url` unless env reward is enabled."

    # colocation parity checks
    use_env_reward = bool(getattr(args, "use_env_reward", False))

    if bool(getattr(args, "colocate_all_models", False)):
        if args.init_kl_coef > 0:
            assert (
                args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
            ), "colocate_all_models: actor/ref must match (num_nodes, gpus_per_node) when KL is enabled."

        # ---- critic / q_critic colocate checks ----
        marl_alg = str(getattr(args, "marl_algorithm", "auto") or "auto").lower().strip()
        eval_only = bool(getattr(args, "eval_only", False))

        # C3: Q-critic is only needed for value_assisted/value_only.
        c3_cv = str(getattr(args, "c3_credit_variant", "value_assisted") or "value_assisted").lower().strip()
        need_q_critic = bool(marl_alg == "c3" and c3_cv in {"value_assisted", "value_only"})

        create_v_critic = bool(
            (not eval_only)
            and bool(getattr(args, "critic_pretrain", None))
            and ((not is_c3_mas) or (is_c3_mas and marl_alg == "mappo"))
        )
        create_q_critic = bool((not eval_only) and is_c3_mas and need_q_critic)

        if create_v_critic or create_q_critic:
            actor_total = int(args.actor_num_nodes * args.actor_num_gpus_per_node)
            critic_total = int(args.critic_num_nodes * args.critic_num_gpus_per_node)
            assert critic_total > 0, "colocate_all_models: critic_num_gpus_per_node must be > 0 when critic is enabled."
            assert critic_total <= actor_total, (
                "colocate_all_models: critic world_size must be <= actor world_size. "
                f"got critic_total={critic_total}, actor_total={actor_total}. "
                "Tip: in C3/MAPPO, critic is small and can be 1 GPU while actor uses more GPUs."
            )

        if (
            (not use_env_reward)
            and (not args.remote_rm_url)
            and (not getattr(args, "agent_func_path", None))
            and getattr(args, "reward_pretrain", None)
        ):
            assert (
                args.actor_num_nodes == args.reward_num_nodes
                and args.actor_num_gpus_per_node == args.reward_num_gpus_per_node
            ), "colocate_all_models: actor/reward must match (num_nodes, gpus_per_node) when reward model is enabled."

        # vLLM parity
        vllm_enabled = bool(getattr(args, "vllm_num_engines", 0) and int(args.vllm_num_engines) > 0)
        if vllm_enabled and (not bool(getattr(args, "async_train", False))):
            tp = int(getattr(args, "vllm_tensor_parallel_size", 1) or 1)
            role_cnt = 1
            if policy_mode == "per_role":
                try:
                    from c3.mas.role_graph import RoleGraph

                    task_spec = ctx.get("task_spec") or load_task(args.c3_task)
                    role_cnt = len(RoleGraph(task_spec.roles).topo_order())
                except Exception:
                    role_cnt = 3

            # If vLLM is forced to create separate PGs, actor pool does not need to cover vLLM GPUs.
            force_sep = getattr(args, "vllm_force_separate_pg", None)
            if policy_mode == "per_role" and force_sep is True:
                pass
            else:
                if policy_mode == "per_role":
                    actor_total = int(args.actor_num_nodes * args.actor_num_gpus_per_node)
                    need = int(args.vllm_num_engines) * tp
                    have = actor_total

                    if bool(getattr(args, "colocate_all_models", False)):
                        override_pg = getattr(args, "per_role_pg_bundles", None)
                        if override_pg is not None:
                            have = int(override_pg)
                        else:
                            create_ref = bool(getattr(args, "init_kl_coef", 0) and float(getattr(args, "init_kl_coef", 0)) > 0)
                            ref_total = int(args.ref_num_nodes * args.ref_num_gpus_per_node) if create_ref else 0
                            if create_ref and ref_total > 0:
                                mode = _normalize_choice(
                                    getattr(args, "per_role_resource_mode", None),
                                    ("auto", "compact", "balanced", "expanded"),
                                    "auto",
                                )
                                if mode in ("balanced", "expanded"):
                                    have = actor_total + ref_total
                                elif mode == "auto":
                                    total = _detect_total_gpu_count()
                                    if total is not None:
                                        if int(total) >= int(role_cnt) * int(actor_total + ref_total):
                                            have = actor_total + ref_total

                    assert have >= need, (
                        "colocate_all_models(per_role): per-role PG GPUs must be >= vllm_num_engines * TP, got "
                        f"have={have}, need={need}. "
                        "Tip: reduce vllm_num_engines/TP, or increase per_role PG bundles (balanced/expanded / --per_role_pg_bundles)."
                    )
                    if have != need:
                        print(
                            f"[Warn] colocate_all_models(per_role): per-role PG GPUs({have}) > vllm_required_gpus({need}); "
                            "vLLM will use a subset of available bundles."
                        )
                else:
                    have = int(args.actor_num_nodes * args.actor_num_gpus_per_node)
                    need = int(args.vllm_num_engines) * tp
                    assert have >= need, (
                        "colocate_all_models: actor_total_gpus must be >= vllm_num_engines * vllm_tensor_parallel_size, got "
                        f"{have} and {need}"
                    )
                    if have != need:
                        print(
                            f"[Warn] colocate_all_models: actor_total_gpus({have}) > vllm_required_gpus({need}); "
                            "vLLM will use a subset of available bundles."
                        )

    if bool(getattr(args, "colocate_actor_ref", False)) and (not bool(getattr(args, "colocate_all_models", False))):
        if args.init_kl_coef > 0:
            assert (
                args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
            ), "colocate_actor_ref: actor/ref must match (num_nodes, gpus_per_node) when KL is enabled."

    if (not args.colocate_all_models) and args.critic_pretrain and bool(getattr(args, "colocate_critic_reward", False)):
        assert (
            args.critic_num_nodes == args.reward_num_nodes and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), "colocate_critic_reward: critic/reward must match (num_nodes, gpus_per_node)."

    # Perf / kl estimator hints
    if args.use_kl_loss:
        if args.kl_estimator not in ["k2", "k3"]:
            print(f"Recommend setting {args.kl_estimator} to 'k2' or 'k3' when using KL as a loss")
    else:
        if args.kl_estimator not in ["k1"]:
            print(f"Recommend setting {args.kl_estimator} to 'k1' when not using KL as a loss.")

    if args.dynamic_filtering:
        assert args.dynamic_filtering_reward_range[0] < args.dynamic_filtering_reward_range[1], (
            "reward_clip_range[0] must be less than reward_clip_range[1]"
        )
        assert (args.remote_rm_url or args.agent_func_path), (
            "remote_rm_url or agent_func_path must be specified when using dynamic filtering"
        )
        assert args.n_samples_per_prompt > 1, "n_samples_per_prompt must be greater than 1 when using dynamic filtering"

    assert (
        args.n_samples_per_prompt * args.rollout_batch_size // args.micro_rollout_batch_size
        >= args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size // args.ds_tensor_parallel_size
    ), "The number of sample batches must be greater than or equal to the effective number of actor processes."


# =========================
# Plan computation & printing
# =========================


def _derive_vllm_ray_actor_num_gpus_request(tp: int, use_hybrid_engine: bool) -> float:
    """
    MUST align with create_vllm_engines():
    - TP==1: num_gpus=1; if hybrid(shared_pg): num_gpus=0.2
    - TP>1: num_gpus=0 (GPU usage managed via PG + child tasks; do not change)
    """
    if int(tp) == 1:
        return 0.2 if use_hybrid_engine else 1.0
    return 0.0


def compute_execution_plan(args, ctx: Dict[str, object]) -> ExecutionPlan:
    policy_mode = getattr(args, "policy_sharing_mode", "shared")
    # ---- per_role guardrails ----
    if policy_mode == "per_role" and bool(getattr(args, "async_train", False)):
        raise SystemExit(
            "ERROR: policy_sharing_mode=per_role is not supported with --async_train. Please disable --async_train."
        )
    role_names: Optional[List[str]] = None
    pretrain_by_role: Optional[Dict[str, str]] = None
    role_count: int = 1

    if policy_mode == "per_role":
        from c3.mas.role_graph import RoleGraph

        task_spec = ctx.get("task_spec") or load_task(args.c3_task)
        role_names = RoleGraph(task_spec.roles).topo_order()
        role_count = len(role_names)
        pretrain_by_role = _resolve_pretrain_by_role(args, role_names)

    is_c3_mas = bool(getattr(args, "rollout_generator_cls", None) and getattr(args, "c3_task", None))
    use_env_reward = bool(getattr(args, "use_env_reward", False))
    eval_only = bool(getattr(args, "eval_only", False))

    # C3: Q-critic is only needed for value_assisted/value_only.
    c3_cv = str(getattr(args, "c3_credit_variant", "value_assisted") or "value_assisted").lower().strip()
    need_q_critic = bool(args.marl_algorithm == "c3" and c3_cv in {"value_assisted", "value_only"})
    create_q_critic = bool((not eval_only) and is_c3_mas and need_q_critic)
    create_reward = bool(
        (not use_env_reward)
        and (not getattr(args, "remote_rm_url", None))
        and (not getattr(args, "agent_func_path", None))
    )

    create_critic = bool(
        (not eval_only)
        and (
            bool(getattr(args, "critic_pretrain", None))
            if (not is_c3_mas)
            else (args.marl_algorithm == "mappo" and bool(getattr(args, "critic_pretrain", None)))
        )
    )

    if use_env_reward:
        reward_source = "env (reward_model disabled)"
    elif getattr(args, "remote_rm_url", None) or getattr(args, "agent_func_path", None):
        reward_source = "remote_rm/agent (reward_model disabled)"
    elif getattr(args, "reward_pretrain", None):
        reward_source = "reward_model"
    else:
        reward_source = "none"

    actor_total_bundles = int(args.actor_num_nodes * args.actor_num_gpus_per_node)
    critic_total_bundles = int(args.critic_num_nodes * args.critic_num_gpus_per_node)
    reward_total_bundles = int(args.reward_num_nodes * args.reward_num_gpus_per_node)

    shared_frac = 0.2
    try:
        if getattr(args, "shared_gpu_fraction_models", None) is not None:
            shared_frac = float(args.shared_gpu_fraction_models)
    except Exception:
        shared_frac = 0.2

    pg_all_bundles: Optional[int] = None
    pg_per_role_bundles: Optional[int] = None
    pg_shared_extra_bundles: Optional[int] = None

    if bool(getattr(args, "colocate_all_models", False)):
        if policy_mode != "per_role":
            pg_all_bundles = actor_total_bundles
        else:
            base_per_role = actor_total_bundles
            total_gpus = _detect_total_gpu_count()

            mode = _normalize_choice(
                getattr(args, "per_role_resource_mode", None),
                ("auto", "compact", "balanced", "expanded"),
                "auto",
            )

            override_pg = getattr(args, "per_role_pg_bundles", None)
            override_frac = getattr(args, "shared_gpu_fraction_models", None)

            create_ref = bool(getattr(args, "init_kl_coef", 0) and float(getattr(args, "init_kl_coef", 0)) > 0)
            ref_total_bundles = int(args.ref_num_nodes * args.ref_num_gpus_per_node) if create_ref else 0

            vllm_enabled = bool(getattr(args, "vllm_num_engines", 0) and int(getattr(args, "vllm_num_engines", 0)) > 0)
            vllm_tp = int(getattr(args, "vllm_tensor_parallel_size", 1) or 1)
            vllm_engines = int(getattr(args, "vllm_num_engines", 0) or 0)
            need_vllm_total = int(role_count) * int(vllm_engines) * int(max(vllm_tp, 1)) if vllm_enabled else 0

            dedicate_ref = False
            if override_pg is not None:
                pg_per_role_bundles = int(override_pg)
            else:
                if mode in ("balanced", "expanded"):
                    dedicate_ref = create_ref
                elif mode == "auto":
                    if create_ref and (total_gpus is not None):
                        if int(total_gpus) >= int(role_count) * int(base_per_role + ref_total_bundles):
                            dedicate_ref = True

                pg_per_role_bundles = int(base_per_role + (ref_total_bundles if dedicate_ref else 0))

            force_sep = getattr(args, "vllm_force_separate_pg", None)
            if force_sep is None:
                vllm_separate = False
                if vllm_enabled:
                    if mode == "expanded":
                        vllm_separate = True
                    elif mode == "auto" and (total_gpus is not None):
                        reserved_policy = int(role_count) * int(pg_per_role_bundles)
                        if int(total_gpus) >= reserved_policy + need_vllm_total:
                            vllm_separate = True
            else:
                vllm_separate = bool(force_sep)

            if total_gpus is not None:
                reserved_policy = int(role_count) * int(pg_per_role_bundles)
                reserved_vllm = int(need_vllm_total) if vllm_separate else 0
                left = int(total_gpus) - reserved_policy - reserved_vllm
                if left > 0:
                    pg_shared_extra_bundles = left

            if override_frac is not None:
                shared_frac = float(override_frac)
            else:
                shared_frac = 1.0 if dedicate_ref else 0.2

            ctx["__per_role_total_gpus__"] = total_gpus
            ctx["__per_role_vllm_separate__"] = vllm_separate
            ctx["__per_role_resource_profile__"] = (
                "expanded" if vllm_separate and dedicate_ref else ("balanced" if dedicate_ref else "compact")
            )
            ctx["__per_role_shared_frac__"] = shared_frac

            try:
                shared_frac = float(ctx.get("__per_role_shared_frac__", shared_frac))
            except Exception:
                pass

    pg_actor_ref_bundles = None
    if (policy_mode != "per_role") and (pg_all_bundles is None) and bool(getattr(args, "colocate_actor_ref", False)):
        pg_actor_ref_bundles = actor_total_bundles

    pg_critic_reward_bundles = None
    if (pg_all_bundles is None) and bool(getattr(args, "colocate_critic_reward", False)) and (
        bool(create_critic) or bool(create_reward) or bool(create_q_critic)
    ):
        if not (policy_mode == "per_role" and bool(getattr(args, "colocate_all_models", False))):
            need = max(
                critic_total_bundles if (create_critic or create_q_critic) else 0,
                reward_total_bundles if create_reward else 0,
            )
            pg_critic_reward_bundles = int(need) if int(need) > 0 else None

    placement = PlacementPlan(
        colocate_all_models=bool(getattr(args, "colocate_all_models", False)),
        colocate_actor_ref=bool(getattr(args, "colocate_actor_ref", False)),
        colocate_critic_reward=bool(getattr(args, "colocate_critic_reward", False)),
        pg_all_bundles=pg_all_bundles,
        pg_actor_ref_bundles=pg_actor_ref_bundles,
        pg_critic_reward_bundles=pg_critic_reward_bundles,
        pg_per_role_bundles=pg_per_role_bundles,
        pg_shared_extra_bundles=pg_shared_extra_bundles,
    )

    vllm_enabled = bool(getattr(args, "vllm_num_engines", 0) and int(args.vllm_num_engines) > 0)
    vllm_tp = int(getattr(args, "vllm_tensor_parallel_size", 1) or 1)
    per_role_vllm_separate = bool(ctx.get("__per_role_vllm_separate__", False)) if policy_mode == "per_role" else False
    use_hybrid_engine = bool(
        vllm_enabled
        and bool(getattr(args, "colocate_all_models", False))
        and (not bool(getattr(args, "async_train", False)))
        and (not per_role_vllm_separate)
    )
    internal_pg_created_by_vllm = bool(vllm_enabled and not use_hybrid_engine)
    executor_backend = "uni" if vllm_tp == 1 else "ray"
    ray_actor_num_gpus_req = _derive_vllm_ray_actor_num_gpus_request(vllm_tp, use_hybrid_engine)

    vllm_plan = VLLMPlan(
        enabled=vllm_enabled,
        num_engines=int(getattr(args, "vllm_num_engines", 0) or 0),
        tensor_parallel_size=vllm_tp,
        max_model_len=int(getattr(args, "vllm_max_model_len", args.vllm_max_model_len)),
        gpu_memory_utilization=getattr(args, "vllm_gpu_memory_utilization", None),
        use_hybrid_engine=use_hybrid_engine,
        internal_pg_created_by_vllm=internal_pg_created_by_vllm,
        executor_backend=executor_backend,
        ray_actor_num_gpus_request=float(ray_actor_num_gpus_req),
        request_batch_size=int(getattr(args, "vllm_generate_batch_size", 0) or 0),
    )

    create_ref = bool(getattr(args, "init_kl_coef", 0) and float(getattr(args, "init_kl_coef", 0)) > 0)

    models_plan = ModelGroupsPlan(
        create_actor=True,
        create_ref=create_ref,
        create_critic=create_critic,
        create_reward=create_reward,
        create_q_critic=create_q_critic,
    )

    notes: List[str] = []
    if bool(getattr(args, "colocate_all_models", False)) and bool(getattr(args, "async_train", False)) and vllm_enabled:
        notes.append("async_train=True => colocate_all_models does NOT share PG with vLLM (vLLM creates its own PG).")
    if bool(getattr(args, "colocate_all_models", False)) and (not bool(getattr(args, "async_train", False))) and vllm_enabled:
        if policy_mode == "per_role":
            notes.append("per_role + colocate_all_models: vLLM shares per-role PGs; small shared models may use pg_shared_extra.")
        else:
            notes.append("async_train=False => vLLM can share pg_all (hybrid engine).")
    if policy_mode == "per_role":
        notes.append("per_role policy does not support async_train (fail-fast).")
    if eval_only:
        notes.append("eval_only=True => ref/critic/remote_rm disabled; vLLM util/batch_size may be overridden for stability.")

    return ExecutionPlan(
        policy_sharing_mode=policy_mode,
        role_names=role_names,
        pretrain_by_role=pretrain_by_role,
        role_count=role_count,
        eval_only=eval_only,
        async_train=bool(getattr(args, "async_train", False)),
        marl_algorithm=str(getattr(args, "marl_algorithm", "auto")),
        mappo_normalize_scope=str(getattr(args, "mappo_normalize_scope", "global")),
        c3_task=getattr(args, "c3_task", None),
        is_c3_mas=is_c3_mas,
        use_env_reward=use_env_reward,
        reward_source=reward_source,
        advantage_estimator=str(getattr(args, "advantage_estimator", "")),
        init_kl_coef=float(getattr(args, "init_kl_coef", 0.0) or 0.0),
        shared_gpu_fraction_models=shared_frac,
        placement=placement,
        vllm=vllm_plan,
        models=models_plan,
        run_dir=getattr(args, "run_dir", None),
        run_id=getattr(args, "run_id", None),
        dump_first_c3_batch=getattr(args, "dump_first_c3_batch", None),
        dump_first_c3_batch_fields=str(getattr(args, "dump_first_c3_batch_fields", "prompts,rewards,info")),
        dump_first_c3_batch_any=bool(getattr(args, "dump_first_c3_batch_any", False)),
        dump_rollouts_jsonl_path=getattr(args, "dump_rollouts_jsonl_path", None),
        dump_rollouts_every=int(getattr(args, "dump_rollouts_every", 0) or 0),
        dump_c3_batch_data_path=getattr(args, "dump_c3_batch_data_path", None),
        eval_dump_path=getattr(args, "eval_dump_path", None),
        eval_dump_mode=str(getattr(args, "eval_dump_mode", "append")),
        notes=notes,
    )


def _print_plan(plan: ExecutionPlan) -> None:
    def _b(v) -> str:
        return "True" if bool(v) else "False"

    print("\n================ Execution Plan ================")
    print(f"mode.eval_only                : {_b(plan.eval_only)}")
    print(f"mode.async_train              : {_b(plan.async_train)}")
    print(f"policy_sharing_mode           : {plan.policy_sharing_mode}")
    if plan.policy_sharing_mode == "per_role":
        print(f"roles                         : {plan.role_names}")
        print(f"role_count                    : {plan.role_count}")
    print(f"marl_algorithm                : {plan.marl_algorithm}")
    print(f"mappo_normalize_scope         : {plan.mappo_normalize_scope}")
    print(f"c3_task                     : {plan.c3_task}")
    print(f"is_c3_mas                   : {_b(plan.is_c3_mas)}")
    print(f"use_env_reward                : {_b(plan.use_env_reward)}")
    print(f"reward_source                 : {plan.reward_source}")

    print("\n---- Colocation / PG ----")
    print(f"colocate_all_models           : {_b(plan.placement.colocate_all_models)}")
    print(f"colocate_actor_ref            : {_b(plan.placement.colocate_actor_ref)}")
    print(f"colocate_critic_reward        : {_b(plan.placement.colocate_critic_reward)}")
    print(f"shared_gpu_fraction(models)   : {plan.shared_gpu_fraction_models}")
    print(f"pg_all.bundles                : {plan.placement.pg_all_bundles}")
    print(f"pg_actor_ref.bundles          : {plan.placement.pg_actor_ref_bundles}")
    print(f"pg_critic_reward.bundles      : {plan.placement.pg_critic_reward_bundles}")
    print(f"pg_per_role.bundles_per_role  : {plan.placement.pg_per_role_bundles}")
    print(f"pg_shared_extra.bundles       : {plan.placement.pg_shared_extra_bundles}")

    print("\n---- vLLM ----")
    print(f"enabled                       : {_b(plan.vllm.enabled)}")
    if plan.vllm.enabled:
        print(f"num_engines x TP              : {plan.vllm.num_engines} x {plan.vllm.tensor_parallel_size}")
        print(f"executor_backend              : {plan.vllm.executor_backend}")
        print(
            f"use_hybrid_engine             : {_b(plan.vllm.use_hybrid_engine)} "
            f"(shared_pg={'pg_all/per_role_pg' if plan.vllm.use_hybrid_engine else 'None'})"
        )
        print(f"ray_actor_num_gpus_request    : {plan.vllm.ray_actor_num_gpus_request}")
        print(f"pg_created_by_vllm            : {_b(plan.vllm.internal_pg_created_by_vllm)}")
        print(f"max_model_len                 : {plan.vllm.max_model_len}")
        print(f"gpu_memory_utilization        : {plan.vllm.gpu_memory_utilization}")
        print(f"request_batch_size            : {plan.vllm.request_batch_size} (driver batching; not engine concurrency)")
    else:
        print("(disabled)")

    print("\n---- Models ----")
    print("actor                         : enabled")
    print(f"ref_model                     : {_b(plan.models.create_ref)} (init_kl_coef={plan.init_kl_coef})")
    print(f"critic                        : {_b(plan.models.create_critic)} (advantage_estimator={plan.advantage_estimator})")
    print(f"reward_model                  : {_b(plan.models.create_reward)}")
    print(f"q_critic                      : {_b(plan.models.create_q_critic)}")

    print("\n---- Dumps ----")
    print(f"dump_first_c3_batch          : {plan.dump_first_c3_batch}")
    print(f"dump_rollouts_jsonl_path       : {plan.dump_rollouts_jsonl_path}")
    print(f"dump_rollouts_every            : {plan.dump_rollouts_every}")
    print(f"dump_c3_batch_data_path      : {plan.dump_c3_batch_data_path}")
    print(f"eval_dump_path                 : {plan.eval_dump_path}")
    print(f"eval_dump_mode                 : {plan.eval_dump_mode}")

    if plan.run_dir or plan.run_id:
        print("\n---- Run ----")
        print(f"run_dir                       : {plan.run_dir}")
        print(f"run_id                        : {plan.run_id}")

    if plan.notes:
        print("\n---- Notes ----")
        for n in plan.notes:
            print(f"- {n}")

    print("===============================================\n")


# =========================
# Builders & execution
# =========================


# Per-bundle CPU is critical for per_role colocate: multiple Ray actors may share the same GPU bundle.
# Allow overriding via env for extreme CPU-constrained environments.
_PG_CPU_PER_BUNDLE = 4
try:
    _PG_CPU_PER_BUNDLE = int(os.environ.get("OPENRLHF_PG_CPU_PER_BUNDLE", "4") or "4")
except Exception:
    _PG_CPU_PER_BUNDLE = 4
_PG_CPU_PER_BUNDLE = max(1, _PG_CPU_PER_BUNDLE)


def _make_pg(num_bundles: int):
    bundles = [{"GPU": 1, "CPU": _PG_CPU_PER_BUNDLE} for _ in range(int(num_bundles))]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())
    return pg


def build_placement_groups(plan: ExecutionPlan):
    pg_all = _make_pg(plan.placement.pg_all_bundles) if plan.placement.pg_all_bundles else None

    pg_by_role = None
    pg_shared = None

    if plan.placement.pg_all_bundles:
        pg_actor_ref = pg_all
    elif plan.placement.pg_actor_ref_bundles:
        pg_actor_ref = _make_pg(plan.placement.pg_actor_ref_bundles)
    else:
        pg_actor_ref = None

    if plan.placement.pg_all_bundles:
        pg_critic_reward = pg_all
    elif plan.placement.pg_critic_reward_bundles:
        pg_critic_reward = _make_pg(plan.placement.pg_critic_reward_bundles)
    else:
        pg_critic_reward = None

    if plan.policy_sharing_mode == "per_role" and plan.placement.pg_per_role_bundles:
        assert plan.role_names, "per_role pg partitioning requires role_names"
        pg_by_role = {}
        for rn in plan.role_names:
            pg_by_role[rn] = _make_pg(plan.placement.pg_per_role_bundles)
        if plan.placement.pg_shared_extra_bundles:
            pg_shared = _make_pg(plan.placement.pg_shared_extra_bundles)

    return pg_all, pg_actor_ref, pg_critic_reward, pg_by_role, pg_shared


def build_vllm_engines(plan: ExecutionPlan, args, pg_all, pg_by_role):
    if not plan.vllm.enabled:
        return None

    if args.agent_func_path:
        from openrlhf.trainer.ray.vllm_engine_async import LLMRayActorAsync as LLMRayActor
    else:
        from openrlhf.trainer.ray.vllm_engine import LLMRayActor

    if plan.policy_sharing_mode == "per_role":
        assert plan.role_names and plan.pretrain_by_role
        vllm_engines_by_role: Dict[str, object] = {}
        for rn in plan.role_names:
            shared_pg = pg_by_role.get(rn) if (plan.vllm.use_hybrid_engine and pg_by_role) else None
            vllm_engines_by_role[rn] = create_vllm_engines(
                plan.vllm.num_engines,
                plan.vllm.tensor_parallel_size,
                plan.pretrain_by_role[rn],
                args.seed,
                args.full_determinism,
                args.enable_prefix_caching,
                args.enforce_eager,
                plan.vllm.max_model_len,
                shared_pg,
                args.vllm_gpu_memory_utilization,
                args.vllm_enable_sleep,
                LLMRayActor,
                "processed_logprobs" if args.enable_vllm_is_correction else None,
                args.agent_func_path,
            )
        return vllm_engines_by_role

    shared_pg = pg_all if plan.vllm.use_hybrid_engine else None
    return create_vllm_engines(
        plan.vllm.num_engines,
        plan.vllm.tensor_parallel_size,
        args.pretrain,
        args.seed,
        args.full_determinism,
        args.enable_prefix_caching,
        args.enforce_eager,
        plan.vllm.max_model_len,
        shared_pg,
        args.vllm_gpu_memory_utilization,
        args.vllm_enable_sleep,
        LLMRayActor,
        "processed_logprobs" if args.enable_vllm_is_correction else None,
        args.agent_func_path,
    )


def _mk_group(num_nodes, num_gpus_per_node, actor_cls, pg, frac, dup):
    return RayActorGroup(
        num_nodes,
        num_gpus_per_node,
        actor_cls,
        pg=pg,
        num_gpus_per_actor=(frac if pg else 1),
        duplicate_actors=dup,
    )


def build_actor_ref_groups(plan: ExecutionPlan, args, pg_actor_ref, pg_by_role):
    dup = int(args.ring_attn_size * args.ds_tensor_parallel_size)
    frac = plan.shared_gpu_fraction_models

    if plan.policy_sharing_mode == "per_role":
        assert plan.role_names is not None
        actor_model: Dict[str, RayActorGroup] = {}
        for rn in plan.role_names:
            pg = pg_by_role.get(rn) if (pg_by_role and bool(getattr(args, "colocate_all_models", False))) else None
            actor_model[rn] = _mk_group(
                args.actor_num_nodes, args.actor_num_gpus_per_node, PolicyModelActor, pg, frac, dup
            )

        if not plan.models.create_ref:
            ref_model = None
        else:
            ref_model = {}
            for rn in plan.role_names:
                pg = pg_by_role.get(rn) if (pg_by_role and bool(getattr(args, "colocate_all_models", False))) else None
                ref_model[rn] = _mk_group(
                    args.ref_num_nodes, args.ref_num_gpus_per_node, ReferenceModelActor, pg, frac, dup
                )
        return actor_model, ref_model

    actor_model = _mk_group(args.actor_num_nodes, args.actor_num_gpus_per_node, PolicyModelActor, pg_actor_ref, frac, dup)
    if not plan.models.create_ref:
        ref_model = None
    else:
        ref_model = _mk_group(args.ref_num_nodes, args.ref_num_gpus_per_node, ReferenceModelActor, pg_actor_ref, frac, dup)
    return actor_model, ref_model


def build_critic_reward_groups(plan: ExecutionPlan, args, pg_all, pg_critic_reward, pg_by_role, pg_shared):
    dup = int(args.ring_attn_size * args.ds_tensor_parallel_size)
    frac = plan.shared_gpu_fraction_models

    per_role_small_pg = None
    if plan.policy_sharing_mode == "per_role" and bool(getattr(args, "colocate_all_models", False)):
        if pg_shared is not None:
            per_role_small_pg = pg_shared
        elif pg_by_role and plan.role_names:
            per_role_small_pg = pg_by_role.get(plan.role_names[0])

    critic_model = None
    if plan.models.create_critic:
        use_pg = per_role_small_pg if per_role_small_pg is not None else pg_critic_reward
        critic_model = _mk_group(
            args.critic_num_nodes, args.critic_num_gpus_per_node, CriticModelActor, use_pg, frac, dup
        )

    q_critic_model = None
    if plan.models.create_q_critic:
        if (
            plan.policy_sharing_mode == "per_role"
            and bool(getattr(args, "colocate_all_models", False))
            and pg_by_role
            and plan.role_names
        ):
            q_pg = pg_by_role.get(plan.role_names[0])
            if pg_shared is not None:
                try:
                    actor_ws = int(args.actor_num_nodes * args.actor_num_gpus_per_node)
                    if int(getattr(plan.placement, "pg_shared_extra_bundles", 0) or 0) >= actor_ws:
                        q_pg = pg_shared
                except Exception:
                    pass
        else:
            q_pg = (pg_all if args.colocate_all_models else (pg_critic_reward if pg_critic_reward else None))

        q_nodes = args.critic_num_nodes
        q_gpus_per_node = args.critic_num_gpus_per_node
        if bool(getattr(args, "colocate_all_models", False)) and q_pg is not None:
            try:
                q_nodes = args.actor_num_nodes
                q_gpus_per_node = args.actor_num_gpus_per_node
            except Exception:
                pass

        q_critic_model = _mk_group(q_nodes, q_gpus_per_node, QCriticModelActor, q_pg, frac, dup)

    reward_pretrain: Optional[str] = None
    reward_model = None
    if plan.models.create_reward:
        reward_pretrain = args.reward_pretrain
        reward_pg = (
            per_role_small_pg
            if per_role_small_pg is not None
            else (pg_critic_reward if pg_critic_reward else (pg_all if args.colocate_all_models else None))
        )
        reward_model = _mk_group(
            args.reward_num_nodes, args.reward_num_gpus_per_node, RewardModelActor, reward_pg, frac, dup
        )

    return critic_model, reward_model, q_critic_model, reward_pretrain


def run_training(
    plan: ExecutionPlan,
    args,
    strategy,
    actor_model,
    critic_model,
    reward_model,
    ref_model,
    vllm_engines,
    q_critic_model,
    reward_pretrain,
):
    if args.async_train:
        from openrlhf.trainer.ppo_trainer_async import PPOTrainerAsync as PPOTrainer
    else:
        from openrlhf.trainer.ppo_trainer import PPOTrainer

    ppo_trainer = PPOTrainer.remote(
        args.pretrain,
        strategy,
        actor_model,
        critic_model,
        reward_model,
        ref_model,
        vllm_engines,
        q_critic_model_group=q_critic_model,
        prompt_split=args.prompt_split,
        eval_split=args.eval_split,
        do_sample=True,
        prompt_max_len=args.prompt_max_len,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    max_steps = ray.get(ppo_trainer.get_max_steps.remote())

    # ---- Eval schedule by progress (percent/ratio) ----
    try:
        p = float(getattr(args, "eval_every_percent", 0.0) or 0.0)
    except Exception:
        p = 0.0
    try:
        r = float(getattr(args, "eval_every_ratio", 0.0) or 0.0)
    except Exception:
        r = 0.0

    ratio = (p / 100.0) if p > 0 else r
    if ratio and ratio > 0 and max_steps and max_steps > 0:
        eval_steps = int(round(float(max_steps) * float(ratio)))
        eval_steps = max(1, min(int(max_steps), int(eval_steps)))
        if int(getattr(args, "eval_steps", -1) or -1) != int(eval_steps):
            print(
                f"[INFO] eval schedule: max_steps={max_steps}, ratio={ratio:.6f} -> eval_steps={eval_steps} (override --eval_steps={getattr(args,'eval_steps',-1)})"
            )
        args.eval_steps = int(eval_steps)

        try:
            eval_steps_offset = int(getattr(args, "eval_steps_offset", 0) or 0)
        except Exception:
            eval_steps_offset = 0
        if eval_steps_offset < 0:
            eval_steps_offset = 0
        args.eval_steps_offset = int(eval_steps_offset)

        if not getattr(args, "eval_dump_path", None) and getattr(args, "run_dir", None):
            args.eval_dump_path = os.path.join(args.run_dir, "eval_during_train.jsonl")

        try:
            ray.get(ppo_trainer.set_eval_steps.remote(int(eval_steps)))
        except Exception:
            pass
        try:
            ray.get(ppo_trainer.set_eval_steps_offset.remote(int(args.eval_steps_offset)))
        except Exception:
            pass

    refs = []
    if plan.policy_sharing_mode == "per_role":
        assert plan.role_names and plan.pretrain_by_role and isinstance(actor_model, dict)
        if isinstance(ref_model, dict):
            for rn in plan.role_names:
                refs.extend(ref_model[rn].async_init_model_from_pretrained(strategy, plan.pretrain_by_role[rn]))
        for rn in plan.role_names:
            rn_vllm = vllm_engines.get(rn) if isinstance(vllm_engines, dict) else None
            refs.extend(
                actor_model[rn].async_init_model_from_pretrained(
                    strategy, plan.pretrain_by_role[rn], max_steps, rn_vllm, role_name=rn
                )
            )
    else:
        if ref_model is not None:
            refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
        refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain, max_steps, vllm_engines))

    if (not args.remote_rm_url) and (reward_model is not None):
        refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))
    ray.get(refs)

    if plan.models.create_critic and args.critic_pretrain and critic_model is not None:
        ray.get(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))

    if q_critic_model is not None:
        q_pretrain = getattr(args, "q_critic_pretrain", None) or args.critic_pretrain or args.pretrain
        ray.get(q_critic_model.async_init_model_from_pretrained(strategy, q_pretrain, max_steps))

    if plan.eval_only:
        if plan.async_train:
            raise SystemExit("ERROR: --eval_only is not supported with --async_train yet.")
        ray.get(ppo_trainer.run_eval_only.remote(global_step=int(getattr(args, "eval_global_step", 0) or 0)))
        return

    ray.get(ppo_trainer.fit.remote())

    if plan.policy_sharing_mode == "per_role":
        assert plan.role_names and isinstance(actor_model, dict)
        save_refs = []
        for rn in plan.role_names:
            save_refs.extend(actor_model[rn].async_save_model())
        ray.get(save_refs)
    else:
        ray.get(actor_model.async_save_model())

    if plan.models.create_critic and args.save_value_network and critic_model is not None:
        ray.get(critic_model.async_save_model())

    if q_critic_model is not None and bool(getattr(args, "save_q_critic", False)):
        ray.get(q_critic_model.async_save_model())


def build_and_run(plan: ExecutionPlan, args) -> None:
    _ensure_ray_initialized(args)

    strategy = get_strategy(args)
    strategy.print(args)

    pg_all, pg_actor_ref, pg_critic_reward, pg_by_role, pg_shared = build_placement_groups(plan)

    vllm_engines = build_vllm_engines(plan, args, pg_all, pg_by_role)
    actor_model, ref_model = build_actor_ref_groups(plan, args, pg_actor_ref, pg_by_role)
    critic_model, reward_model, q_critic_model, reward_pretrain = build_critic_reward_groups(
        plan, args, pg_all, pg_critic_reward, pg_by_role, pg_shared
    )

    run_training(
        plan,
        args,
        strategy,
        actor_model,
        critic_model,
        reward_model,
        ref_model,
        vllm_engines,
        q_critic_model,
        reward_pretrain,
    )


def run_cli(args) -> None:
    ctx = normalize_args(args)
    validate_args(args, ctx)

    # =========================
    # Sync CLI args -> env (critic warmup rollout cache / schedule / slim)
    # =========================
    def _set_env_if(name: str, val):
        if val is None:
            return
        os.environ[name] = str(val)

    if getattr(args, "critic_warmup_rollout_cache_dir", ""):
        _set_env_if("CRITIC_WARMUP_ROLLOUT_CACHE_DIR", args.critic_warmup_rollout_cache_dir)

    if getattr(args, "critic_warmup_rollout_cache_mode", ""):
        _set_env_if("CRITIC_WARMUP_ROLLOUT_CACHE_MODE", args.critic_warmup_rollout_cache_mode)

    slim = getattr(args, "critic_warmup_rollout_cache_slim", -1)
    if isinstance(slim, int) and slim in (0, 1):
        _set_env_if("CRITIC_WARMUP_ROLLOUT_CACHE_SLIM", slim)

    sched = getattr(args, "critic_warmup_rollout_schedule", 1)
    _set_env_if("CRITIC_WARMUP_ROLLOUT_SCHEDULE", int(sched))

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        patch_hub()

    init_run_artifacts(args)
    _setup_local_run_logging(args)

    # For per_role auto placement, we want cluster GPU visibility before planning.
    if getattr(args, "policy_sharing_mode", "shared") == "per_role":
        _ensure_ray_initialized(args)

    plan = None
    try:
        plan = compute_execution_plan(args, ctx)
        _print_plan(plan)
        build_and_run(plan, args)
    finally:
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
        stop_on_exit = os.environ.get("OPENRLHF_RAY_STOP_ON_EXIT", "").strip().lower() in {
            "1","true","yes","y","on",
        }
        if stop_on_exit:
            try:
                subprocess.run(["ray", "stop", "--force"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_cli(args)
