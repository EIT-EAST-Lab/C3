"""
openrlhf/cli/train_ppo_ray_tooling.py

CLI + normalization utilities shared by train_ppo_ray.py.

Responsibilities:
- Build argparse parser.
- Merge C3 task.yaml defaults (CLI > task.yaml > hard defaults).
- Normalize args (MAGRPO/MAPPO/C3 compatibility + safety gates).
- Ray init helpers: runtime_env env-var propagation, object-store sizing cap,
  and an optional first-batch dump hook for C3/MAS debugging.
- Small probes (memory/GPU) for vLLM autosizing.

Notes:
- Keep behavior stable; this module is imported by training entrypoints.
"""

# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

from __future__ import annotations

import argparse
import os
import re
import tempfile
import textwrap
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import ray

from c3.integration.marl_specs import load_task

# --vllm_num_engines=-1 means "auto" (resolved in normalize_args()).
_VLLM_AUTO_ENGINES_SENTINEL = -1

# MAPPO state prompt max length (tokenizer max_length).
_DEFAULT_MAPPO_STATE_MAX_LEN = 2560

# Canonical MAS rollout generator (string compare only; no import needed).
_MAS_ROLLOUT_GENERATOR_CLS = "c3.mas.rollout_generator.MASRolloutGenerator"

# Ray: keep the memory monitor from killing RLHF workers too aggressively.
_RAY_MEMORY_USAGE_THRESHOLD_DEFAULT = "0.99"

# C3 ablation knobs (Step2): baseline mode switch.
_C3_BASELINE_MODES: Tuple[str, ...] = ("loo", "full_mean")

# C3 fanout examples (2-role 2A, K=8) used in error messages/help.
_C3_FANOUT_EXAMPLE_NESTED = "2,4"
_C3_FANOUT_EXAMPLE_FLAT = "8,1"


# ==============================================================================
# Generic helpers
# ==============================================================================


def _is_unset(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _safe_int(v, default: int) -> int:
    try:
        return int(default) if _is_unset(v) else int(v)
    except Exception:
        return int(default)


def _safe_float(v, default: float) -> float:
    try:
        return float(default) if _is_unset(v) else float(v)
    except Exception:
        return float(default)


def _normalize_choice(value: Optional[str], allowed: Tuple[str, ...], default: str) -> str:
    if _is_unset(value):
        return default
    v = str(value).lower().strip()
    return v if v in allowed else default


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _is_mas_rollout_cls(rollout_generator_cls: Optional[str]) -> bool:
    if _is_unset(rollout_generator_cls):
        return False
    s = str(rollout_generator_cls).strip()
    return s == _MAS_ROLLOUT_GENERATOR_CLS or s.endswith("MASRolloutGenerator")


def _is_c3_or_mas_mode(args) -> bool:
    return bool(getattr(args, "c3_task", None)) or _is_mas_rollout_cls(getattr(args, "rollout_generator_cls", None))


def _apply_c3_fanout_list_alias(args) -> None:
    """
    Step6: accept legacy --c3_fanout_list (space-separated ints) as an alias of --c3_fanout.

    Behavior:
      - If provided, set args.c3_fanout to "a,b,c" (string) for consistent logging.
      - Also prefill args.c3_fanout_list (list[int]) for debug visibility.
      - Final args.c3_fanout_list is still produced by _normalize_c3_algorithm_args().
    """
    raw = getattr(args, "c3_fanout_list_arg", None)
    if raw is None:
        return

    try:
        xs = [int(x) for x in list(raw) if str(x).strip() != ""]
    except Exception:
        raise SystemExit(f"[C3][FAIL-FAST] Invalid --c3_fanout_list: {raw!r} (expected ints).")

    if not xs:
        return

    # If both are set, prefer the list-alias: it's explicit intent in older scripts.
    if not _is_unset(getattr(args, "c3_fanout", None)):
        print("[C3][WARN] Both --c3_fanout and --c3_fanout_list provided; using --c3_fanout_list.")

    args.c3_fanout = ",".join(str(x) for x in xs)
    # Prefill for visibility; will be normalized/validated later.
    args.c3_fanout_list = list(xs)


# ==============================================================================
# Ray object store memory helpers
# ==============================================================================


def _detect_cgroup_memory_limit_bytes() -> Optional[int]:
    """Best-effort container memory limit (cgroup v2 then common v1 paths)."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        s = _read_text(path)
        if not s:
            continue
        if path.endswith("memory.max") and s.lower() == "max":
            continue
        try:
            v = int(s)
            if 0 < v < (1 << 60):
                return v
        except Exception:
            pass
    return None


def _detect_system_memory_bytes() -> Optional[int]:
    """Host total memory (may exceed container limit)."""
    s = _read_text("/proc/meminfo")
    if s:
        try:
            for line in s.splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
        except Exception:
            pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except Exception:
        pass
    return None


def _detect_effective_memory_bytes() -> Optional[int]:
    """Effective memory for this job: min(host, cgroup_limit) if a limit exists."""
    host = _detect_system_memory_bytes()
    lim = _detect_cgroup_memory_limit_bytes()
    if host is None:
        return lim
    if lim is None:
        return host
    try:
        return int(min(int(host), int(lim)))
    except Exception:
        return host


def _parse_human_size_to_bytes(s: str) -> Optional[int]:
    """
    Parse sizes like: "4096", "512m", "2g", "1.5gb".
    Returns:
      - None: parse failed / empty
      - 0   : explicit disable/off
      - >0  : bytes
    """
    if s is None:
        return None
    raw = str(s).strip().lower()
    if raw == "":
        return None
    if raw in ("0", "off", "disable", "disabled", "false", "none", "null"):
        return 0

    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([kmgtp]?b?)?$", raw)
    if not m:
        return None

    num = float(m.group(1))
    unit = (m.group(2) or "").strip()
    mul = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "p": 1024**5,
        "pb": 1024**5,
    }.get(unit)
    if mul is None:
        return None
    return max(0, int(num * mul))


def _auto_object_store_memory_bytes(total_mem_bytes: int) -> int:
    """
    Conservative heuristic for RLHF (vLLM/torch/deepspeed are RSS-hungry):

      - target: 5% of effective memory
      - floor : 512MB
      - cap   : 4GB
      - hard cap: <= 10% of total

    This reduces Ray memory-monitor kills caused by object store over-allocation.
    """
    total = max(0, int(total_mem_bytes))
    if total <= 0:
        return 512 * 1024**2

    floor = 512 * 1024**2
    target = max(floor, int(total * 0.05))
    target = min(target, 4 * 1024**3)
    target = min(target, int(total * 0.10))
    return max(256 * 1024**2, int(target))


def _resolve_ray_object_store_memory_bytes() -> Optional[int]:
    """
    Resolve object_store_memory (bytes).

    Priority:
      1) OPENRLHF_RAY_OBJECT_STORE_MEMORY (bytes/human)
      2) OPENRLHF_RAY_OBJECT_STORE_GB (GB float)
      3) RAY_OBJECT_STORE_MEMORY (bytes/human) [compat]
      4) auto heuristic based on effective memory

    Returns:
      - None: do not set (Ray default)
      - 0   : explicit disable (Ray default)
      - >0  : bytes
    """
    for k in ("OPENRLHF_RAY_OBJECT_STORE_MEMORY", "RAY_OBJECT_STORE_MEMORY"):
        v = os.environ.get(k)
        if v is not None and str(v).strip() != "":
            parsed = _parse_human_size_to_bytes(v)
            if parsed is not None:
                return parsed

    v = os.environ.get("OPENRLHF_RAY_OBJECT_STORE_GB")
    if v is not None and str(v).strip() != "":
        parsed = _parse_human_size_to_bytes(str(v).strip() + "gb")
        if parsed is not None:
            return parsed

    mem = _detect_effective_memory_bytes()
    if mem is None:
        return 512 * 1024**2
    return _auto_object_store_memory_bytes(int(mem))


# ==============================================================================
# GPU probes (vLLM autosizing)
# ==============================================================================


def _detect_visible_gpu_count() -> Optional[int]:
    """Single-node GPU count (CUDA_VISIBLE_DEVICES aware)."""
    s = os.environ.get("CUDA_VISIBLE_DEVICES")
    if s is None:
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.device_count())
        except Exception:
            pass
        return None

    s = str(s).strip()
    if not s:
        return 0
    return len([p for p in (x.strip() for x in s.split(",")) if p])


def _detect_total_gpu_count() -> Optional[int]:
    """Best-effort total GPU count for this Ray cluster/job."""
    try:
        if ray.is_initialized():
            return int(ray.cluster_resources().get("GPU", 0))
    except Exception:
        pass
    return _detect_visible_gpu_count()


# ==============================================================================
# Logging + Ray init
# ==============================================================================


def _setup_local_run_logging(args) -> None:
    """Best-effort local logging setup (driver + Ray actors)."""
    run_dir = getattr(args, "run_dir", None)
    if not run_dir:
        return

    log_dir = getattr(args, "log_dir", None) or os.path.join(run_dir, "logs")
    ray_tmpdir = getattr(args, "ray_tmpdir", None) or os.environ.get("RAY_TMPDIR") or os.path.join(run_dir, "ray_tmp")

    try:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ray_tmpdir, exist_ok=True)
    except Exception:
        pass

    os.environ["RAY_TMPDIR"] = str(ray_tmpdir)

    try:
        from openrlhf.utils.logging_utils import setup_run_logging

        log_subdir = "logs"
        try:
            rd = os.path.realpath(run_dir)
            ld = os.path.realpath(log_dir)
            if ld.startswith(rd + os.sep):
                log_subdir = os.path.relpath(ld, rd)
        except Exception:
            pass

        console = bool(getattr(args, "log_console", True))
        redirect_std = bool(getattr(args, "redirect_std_to_log", False))
        setup_run_logging(
            run_dir,
            log_subdir=log_subdir,
            console=console,
            redirect_std=redirect_std,
            tee_std=console,
        )
    except Exception:
        pass

    if getattr(args, "train_metrics_jsonl_path", None) is None:
        args.train_metrics_jsonl_path = os.path.join(run_dir, "train_metrics.jsonl")


def _collect_runtime_env_vars() -> Dict[str, str]:
    """Env vars to propagate into Ray workers."""
    env: Dict[str, str] = {
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONUTF8": os.environ.get("PYTHONUTF8", "1"),
        "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", "utf-8"),
        "WANDB_SILENT": os.environ.get("WANDB_SILENT", "true"),
        "TQDM_DISABLE": os.environ.get("TQDM_DISABLE", "1"),
    }

    forward_if_set = [
        "OPENRLHF_LOG_DIR",
        "OPENRLHF_LOG_PREFIX",
        "OPENRLHF_LOG_LEVEL",
        "OPENRLHF_LOG_CONSOLE",
        "OPENRLHF_REDIRECT_STD",
        "RAY_TMPDIR",
        "CRITIC_WARMUP_ROLLOUT_CACHE_DIR",
        "CRITIC_WARMUP_ROLLOUT_CACHE_MODE",
        "CRITIC_WARMUP_ROLLOUT_CACHE_SLIM",
        "CRITIC_WARMUP_ROLLOUT_SCHEDULE",
        "DS_INPLACE_GRAD_PRESCALE",
        "OPENRLHF_PG_CPU_PER_BUNDLE",
    ]
    for k in forward_if_set:
        v = os.environ.get(k)
        if v is not None and str(v) != "":
            env[k] = str(v)

    return env


def _ensure_ray_initialized(args) -> None:
    """Initialize Ray with runtime_env propagation and optional object-store cap."""
    if ray.is_initialized():
        return

    os.environ.setdefault("RAY_memory_usage_threshold", _RAY_MEMORY_USAGE_THRESHOLD_DEFAULT)

    runtime_env = {"env_vars": _collect_runtime_env_vars()}
    runtime_env = _maybe_enable_first_c3_batch_dump(args, runtime_env)

    ray_init_kwargs = {"runtime_env": runtime_env}

    try:
        obj_store = _resolve_ray_object_store_memory_bytes()
        if obj_store is not None and int(obj_store) > 0:
            ray_init_kwargs["object_store_memory"] = int(obj_store)
            gb = float(int(obj_store)) / (1024**3)
            print(
                f"[Ray] object_store_memory cap enabled: {gb:.2f} GB "
                f"(override: OPENRLHF_RAY_OBJECT_STORE_GB / OPENRLHF_RAY_OBJECT_STORE_MEMORY; "
                f"disable: OPENRLHF_RAY_OBJECT_STORE_MEMORY=0)"
            )
        elif obj_store is not None and int(obj_store) == 0:
            print("[Ray] object_store_memory cap disabled by env (Ray default behavior).")
    except Exception as e:
        print(f"[Ray][Warn] failed to resolve object_store_memory cap: {type(e).__name__}: {e}")

    ray.init(**ray_init_kwargs)


# ==============================================================================
# First batch dump hook (sitecustomize)
# ==============================================================================


def _maybe_enable_first_c3_batch_dump(args, runtime_env: dict) -> dict:
    """
    Dump the first experience batch (post adv/ret) from Ray workers via sitecustomize.

    Worker env vars:
      - OPENRLHF_DUMP_FIRST_C3_BATCH
      - OPENRLHF_DUMP_FIRST_C3_BATCH_FIELDS
      - OPENRLHF_DUMP_FIRST_C3_BATCH_ANY
    """
    dump_path = str(getattr(args, "dump_first_c3_batch", "") or "").strip()
    if not dump_path:
        return runtime_env

    try:
        parent = os.path.dirname(dump_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass

    patch_dir = tempfile.mkdtemp(prefix="openrlhf_sitecustomize_")
    sc_path = os.path.join(patch_dir, "sitecustomize.py")

    fields = str(getattr(args, "dump_first_c3_batch_fields", "prompts,rewards,info") or "prompts,rewards,info")
    dump_any = bool(getattr(args, "dump_first_c3_batch_any", False))

    code = f"""
import os
import torch

from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker, Experience

_ORIG = RemoteExperienceMaker.make_experience_batch

def _should_dump(self):
    any_flag = os.environ.get("OPENRLHF_DUMP_FIRST_C3_BATCH_ANY", "0") == "1"
    if any_flag:
        return True
    alg = getattr(getattr(self, "args", None), "marl_algorithm", None)
    return str(alg or "").lower().strip() == "c3"

def _patched(self, rollout_samples):
    exps = _ORIG(self, rollout_samples)

    path = os.environ.get("OPENRLHF_DUMP_FIRST_C3_BATCH", "").strip()
    if (not path) or (not _should_dump(self)):
        return exps

    done = path + ".done"
    try:
        fd = os.open(done, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return exps

    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        fields = os.environ.get("OPENRLHF_DUMP_FIRST_C3_BATCH_FIELDS", "prompts,rewards,info").strip()
        if fields and fields.lower() != "all":
            flist = [f.strip() for f in fields.split(",") if f.strip()]
            exps_to_save = Experience.select(exps, flist)
        else:
            exps_to_save = exps

        for e in exps_to_save:
            try:
                e.to_device(torch.device("cpu"))
            except Exception:
                pass

        torch.save({{"experiences": exps_to_save}}, path)
        print(f"[OPENRLHF][DUMP] saved first experience batch to {{path}} (fields={{fields}})")
    except Exception as e:
        try:
            os.unlink(done)
        except Exception:
            pass
        print(f"[OPENRLHF][DUMP][ERROR] failed saving first experience batch to {{path}}: {{type(e).__name__}}: {{e}}")

    return exps

RemoteExperienceMaker.make_experience_batch = _patched
"""
    with open(sc_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(code))

    runtime_env = dict(runtime_env or {})
    env_vars = dict(runtime_env.get("env_vars", {}) or {})
    env_vars["OPENRLHF_DUMP_FIRST_C3_BATCH"] = dump_path
    env_vars["OPENRLHF_DUMP_FIRST_C3_BATCH_FIELDS"] = fields
    env_vars["OPENRLHF_DUMP_FIRST_C3_BATCH_ANY"] = "1" if dump_any else "0"
    runtime_env["env_vars"] = env_vars

    # Ship patch_dir so sitecustomize is importable.
    runtime_env["working_dir"] = patch_dir

    print(f"[OPENRLHF][DUMP] enabled first batch dump to: {dump_path} (fields={fields}, any={dump_any})")
    return runtime_env


# ==============================================================================
# per_role pretrain resolve
# ==============================================================================


def _resolve_pretrain_by_role(args, role_names: List[str]) -> Dict[str, str]:
    """Return {role: pretrain_path}. Used only when policy_sharing_mode=per_role."""
    mapping: Dict[str, str] = {}
    json_path = getattr(args, "pretrain_by_role_json", None)
    pattern = getattr(args, "pretrain_by_role_pattern", None)
    base = getattr(args, "pretrain", None)

    if json_path:
        import json
        from pathlib import Path

        p = Path(str(json_path))
        if not p.exists():
            raise FileNotFoundError(f"--pretrain_by_role_json not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("--pretrain_by_role_json must be a JSON object: {role: path}")
        for rn in role_names:
            if rn in data and data[rn]:
                mapping[rn] = str(data[rn])

    elif pattern:
        pat = str(pattern)
        if "{role}" not in pat:
            raise ValueError("--pretrain_by_role_pattern must contain '{role}' placeholder")
        for rn in role_names:
            mapping[rn] = pat.format(role=rn)

    for rn in role_names:
        if not mapping.get(rn):
            mapping[rn] = base

    return mapping


# ==============================================================================
# Task defaults merge + args normalization
# ==============================================================================


_TASK_DEFAULT_KEYS: Tuple[str, ...] = (
    # C3 core
    "c3_fanout",
    "c3_credit_variant",
    "c3_va_alpha",
    "c3_baseline_mode",
    "c3_no_replay",
    # Q-critic controls
    "critic_ctx_limit",
    "critic_forward_bs",
    "critic_preamble_path",
    "q_critic_pretrain",
    "q_critic_learning_rate",
    "q_critic_train_batch_size",
    "c3_critic_loss_type",
    # dumps
    "dump_rollouts_jsonl_path",
    "dump_rollouts_every",
    "dump_rollouts_compact",  # Step2: compact JSONL dump
    "dump_c3_batch_data_path",
    # MARL knobs
    "mappo_normalize_scope",
    "mappo_state_max_len",
    "magrpo_baseline",
    "magrpo_adv_unit",
)



def _merge_task_defaults_into_args(args, task_spec) -> None:
    """Merge task.yaml defaults into args (CLI > task.yaml > hard defaults)."""
    env_cfg = getattr(task_spec, "environment", None)
    env_cfg = env_cfg if isinstance(env_cfg, dict) else {}
    c3_cfg = env_cfg.get("c3", {})
    c3_cfg = c3_cfg if isinstance(c3_cfg, dict) else {}

    def pick_env(key: str, default=None):
        return env_cfg.get(key, c3_cfg.get(key, default))

    for k in _TASK_DEFAULT_KEYS:
        if _is_unset(getattr(args, k, None)):
            v = pick_env(k, None)
            if not _is_unset(v):
                setattr(args, k, v)


def _normalize_train_epochs(args) -> None:
    args.train_epochs = max(1, _safe_int(getattr(args, "train_epochs", None), 1))


def _normalize_dumps(args) -> None:
    # Rollout JSONL dump triggers.
    if _is_unset(getattr(args, "dump_rollouts_every", None)):
        args.dump_rollouts_every = 0
    if _is_unset(getattr(args, "dump_rollouts_jsonl_path", None)):
        args.dump_rollouts_jsonl_path = None
    if _is_unset(getattr(args, "dump_rollouts_compact", None)):
        args.dump_rollouts_compact = False

    # C3 batch dump (debug).
    if _is_unset(getattr(args, "dump_c3_batch_data_path", None)):
        args.dump_c3_batch_data_path = None

    # If a path is provided, default cadence to 1.
    try:
        if args.dump_rollouts_jsonl_path and int(args.dump_rollouts_every) <= 0:
            args.dump_rollouts_every = 1
    except Exception:
        pass

    # Auto policy: keep large texts only when dumping/debugging, unless user overrides.
    if _is_unset(getattr(args, "keep_rollout_texts", None)):
        args.keep_rollout_texts = None
    if args.keep_rollout_texts is None:
        args.keep_rollout_texts = bool(args.dump_rollouts_jsonl_path) or bool(args.dump_c3_batch_data_path)


def _normalize_analysis_out_dir(args) -> None:
    """
    Step2: provide a stable default analysis directory.
    Priority:
      1) --analysis_out_dir (explicit)
      2) <run_dir>/analysis   (preferred if run_dir is used)
      3) <ckpt_path>/analysis (fallback)
      4) <save_path>/analysis (last resort)
    """
    if not _is_unset(getattr(args, "analysis_out_dir", None)):
        return

    run_dir = getattr(args, "run_dir", None)
    if not _is_unset(run_dir):
        args.analysis_out_dir = os.path.join(str(run_dir), "analysis")
        return

    ckpt_path = getattr(args, "ckpt_path", None)
    if not _is_unset(ckpt_path):
        args.analysis_out_dir = os.path.join(str(ckpt_path), "analysis")
        return

    save_path = getattr(args, "save_path", None)
    if not _is_unset(save_path):
        args.analysis_out_dir = os.path.join(str(save_path), "analysis")
        return

    args.analysis_out_dir = None


def _normalize_vllm_max_len(args) -> None:
    if getattr(args, "max_len", None) is not None:
        args.vllm_max_model_len = max(1, int(args.max_len))
    else:
        pm = _safe_int(getattr(args, "prompt_max_len", 0), 0)
        gm = _safe_int(getattr(args, "generate_max_len", 0), 0)
        args.vllm_max_model_len = max(1, int(pm + gm))


def _infer_per_role_role_count(args, ctx: Dict[str, object]) -> int:
    if getattr(args, "policy_sharing_mode", "shared") != "per_role":
        return 1
    try:
        from c3.mas.role_graph import RoleGraph

        task_spec = ctx.get("task_spec") or load_task(args.c3_task)
        return len(RoleGraph(task_spec.roles).topo_order())
    except Exception:
        return 3


def _normalize_vllm_num_engines_auto(args, ctx: Dict[str, object]) -> None:
    """Resolve --vllm_num_engines when set to auto sentinel (-1)."""
    try:
        n = int(getattr(args, "vllm_num_engines", _VLLM_AUTO_ENGINES_SENTINEL))
    except Exception:
        n = _VLLM_AUTO_ENGINES_SENTINEL

    if n == 0:
        raise SystemExit(
            "ERROR: vLLM must be enabled in this project. "
            "Use --vllm_num_engines=-1 for auto, or set a positive integer."
        )
    if n > 0:
        return

    tp = max(1, _safe_int(getattr(args, "vllm_tensor_parallel_size", 1), 1))
    actor_pool = int(getattr(args, "actor_num_nodes", 1) * getattr(args, "actor_num_gpus_per_node", 0))
    if actor_pool <= 0:
        raise SystemExit("ERROR: actor_num_gpus_per_node must be > 0 for vLLM auto sizing.")

    policy_mode = getattr(args, "policy_sharing_mode", "shared")
    colocate_all = bool(getattr(args, "colocate_all_models", False))
    async_train = bool(getattr(args, "async_train", False))

    pool_gpus = actor_pool

    if policy_mode == "per_role":
        role_cnt = _infer_per_role_role_count(args, ctx)
        create_ref = _safe_float(getattr(args, "init_kl_coef", 0.0), 0.0) > 0.0
        ref_pool = int(getattr(args, "ref_num_nodes", 1) * getattr(args, "ref_num_gpus_per_node", 0)) if create_ref else 0

        if colocate_all and (not bool(getattr(args, "vllm_force_separate_pg", False))):
            override_pg = getattr(args, "per_role_pg_bundles", None)
            if override_pg is not None:
                pool_gpus = int(override_pg)
            else:
                mode = _normalize_choice(
                    getattr(args, "per_role_resource_mode", None),
                    ("auto", "compact", "balanced", "expanded"),
                    "auto",
                )
                pool_gpus = actor_pool
                if create_ref and ref_pool > 0:
                    if mode in ("balanced", "expanded"):
                        pool_gpus += ref_pool
                    elif mode == "auto":
                        total = _detect_total_gpu_count()
                        if total is not None:
                            need_for_dedicate_ref = int(role_cnt) * int(actor_pool + ref_pool)
                            if int(total) >= need_for_dedicate_ref:
                                pool_gpus += ref_pool
    else:
        if (not colocate_all) or async_train:
            total = _detect_total_gpu_count()
            if total is not None and int(total) > 0:
                pool_gpus = max(actor_pool, int(total))

    if pool_gpus < tp:
        raise SystemExit(
            f"ERROR: vLLM auto sizing failed: pool_gpus({pool_gpus}) < TP({tp}). "
            "Reduce --vllm_tensor_parallel_size or increase actor/per-role GPU pool."
        )

    args.vllm_num_engines = int(max(1, int(pool_gpus) // int(tp)))
    print(
        f"[vLLM][auto] vllm_num_engines={_VLLM_AUTO_ENGINES_SENTINEL} -> {args.vllm_num_engines} "
        f"(policy_mode={policy_mode}, pool_gpus={pool_gpus}, tp={tp}, "
        f"colocate_all_models={colocate_all}, async_train={async_train})"
    )


def _detect_env_reward(args, task_spec) -> None:
    """Detect whether c3_task uses a supported env-reward environment."""
    args.use_env_reward = False
    args.c3_env_name = None
    if task_spec is None:
        return
    try:
        from c3.envs import SUPPORTED_ENVS

        args.c3_env_name = getattr(task_spec, "env_name", None)
        if getattr(task_spec, "env_name", None) in SUPPORTED_ENVS:
            args.use_env_reward = True
    except Exception as e:
        raise SystemExit(f"ERROR: failed to load --c3_task {getattr(args,'c3_task',None)!r}: {e}")


def _normalize_marl(args) -> None:
    """Resolve marl_algorithm + algorithm-specific knobs."""
    try:
        from c3.algorithms.registry import canonical_name

        args.marl_algorithm = canonical_name(getattr(args, "marl_algorithm", "auto"))
    except Exception:
        args.marl_algorithm = (getattr(args, "marl_algorithm", "auto") or "auto").strip().lower()

    args.magrpo_baseline = _normalize_choice(
        getattr(args, "magrpo_baseline", None),
        allowed=("group_mean", "rloo"),
        default="group_mean",
    )
    args.magrpo_adv_unit = _normalize_choice(
        getattr(args, "magrpo_adv_unit", None),
        allowed=("joint_action", "per_role"),
        default="joint_action",
    )
    args.mappo_normalize_scope = _normalize_choice(
        getattr(args, "mappo_normalize_scope", None),
        allowed=("global", "group", "group_role", "episode", "episode_role"),
        default="global",
    )

    if _is_unset(getattr(args, "mappo_state_max_len", None)):
        args.mappo_state_max_len = int(_DEFAULT_MAPPO_STATE_MAX_LEN)
    else:
        args.mappo_state_max_len = max(256, _safe_int(getattr(args, "mappo_state_max_len", None), _DEFAULT_MAPPO_STATE_MAX_LEN))

    # Auto selection for C3/MAS tasks: K>1 -> MAGRPO, else MAPPO.
    if args.marl_algorithm == "auto" and _is_c3_or_mas_mode(args):
        k = max(1, _safe_int(getattr(args, "n_samples_per_prompt", 1), 1))
        args.marl_algorithm = "magrpo" if k > 1 else "mappo"
        print(f"[Info] marl_algorithm=auto -> {args.marl_algorithm} (K={k})")

    if _is_c3_or_mas_mode(args) or bool(getattr(args, "use_env_reward", False)):
        print(f"[Info] marl_algorithm={args.marl_algorithm}")


def _normalize_c3_task_meta(args, task_spec) -> None:
    """Derive C3 task metadata (roles topo)."""
    if task_spec is None:
        return
    try:
        from c3.mas.role_graph import RoleGraph

        roles_topo = list(RoleGraph(task_spec.roles).topo_order())
    except Exception as e:
        raise SystemExit(f"[C3][FAIL-FAST] Failed to infer topo roles from task_spec: {e}")

    if not roles_topo:
        raise SystemExit("[C3][FAIL-FAST] Empty topo roles in C3 task.")
    args.c3_roles_topo = tuple(roles_topo)

    if not bool(getattr(args, "return_meta_json", False)):
        print("[C3][INFO] Enabling --return_meta_json for C3 task.")
        setattr(args, "return_meta_json", True)


def _normalize_c3_algorithm_args(args) -> None:
    """
    Validate C3-only knobs.
    Fanout is enforced only for C3 training when marl_algorithm=c3 and not eval_only.
    """
    marl_alg = str(getattr(args, "marl_algorithm", "auto") or "auto").lower().strip()
    eval_only = bool(getattr(args, "eval_only", False))
    roles_topo = list(getattr(args, "c3_roles_topo", ()) or ())
    if not roles_topo:
        # Still normalize baseline_mode to a stable value for downstream cfg (no-op otherwise).
        args.c3_baseline_mode = _normalize_choice(getattr(args, "c3_baseline_mode", None), _C3_BASELINE_MODES, "loo")
        return

    def _parse_fanout(v):
        """
        Accept common fanout notations:
          - "2,4"        (comma)
          - "2 4"        (whitespace)
          - "2x4", "2*4" (multiplicative style)
          - "[2,4]" "(2,4)" "{2,4}" (bracketed)
          - list/tuple of ints
        """
        if _is_unset(v):
            return None

        if isinstance(v, (list, tuple)):
            xs = list(v)
        else:
            raw = str(v).strip()
            if raw == "":
                return None
            # Strip common wrappers to be forgiving.
            raw = raw.strip().strip("[](){}")
            # Split by comma / whitespace / x / * (case-insensitive).
            xs = [t for t in re.split(r"[,\s*xX\*]+", raw) if t]

        out: List[int] = []
        for t in xs:
            try:
                out.append(int(str(t).strip()))
            except Exception:
                raise SystemExit(f"[C3][FAIL-FAST] Invalid --c3_fanout element: {t!r}")
        return out


    # Step2: C3 baseline mode (ablation: No LOO).
    args.c3_baseline_mode = _normalize_choice(
        getattr(args, "c3_baseline_mode", None),
        allowed=_C3_BASELINE_MODES,
        default="loo",
    )

    fanout = _parse_fanout(getattr(args, "c3_fanout", None))

    args.c3_credit_variant = _normalize_choice(
        getattr(args, "c3_credit_variant", None),
        allowed=("reward_only", "value_assisted", "value_only"),
        default="value_assisted",
    )
    args.c3_va_alpha = _safe_float(getattr(args, "c3_va_alpha", None), 1.0)
    if args.c3_va_alpha < 0.0 or args.c3_va_alpha > 1.0:
        raise SystemExit(f"[C3][FAIL-FAST] --c3_va_alpha must be in [0,1], got {args.c3_va_alpha}")

    require_fanout = (marl_alg == "c3") and (not eval_only)
    if not require_fanout:
        args.c3_fanout_list = None
        if fanout is not None:
            if len(fanout) != len(roles_topo):
                print(
                    "[C3][WARN] Ignoring --c3_fanout due to length mismatch. "
                    f"roles_topo={roles_topo} (expected len={len(roles_topo)}), fanout={fanout} (got len={len(fanout)}). "
                    f"Example (2A, K=8): --c3_fanout {_C3_FANOUT_EXAMPLE_NESTED} or {_C3_FANOUT_EXAMPLE_FLAT}."
                )
            elif any(int(x) <= 0 for x in fanout):
                print(f"[C3][WARN] Ignoring --c3_fanout with non-positive ints: {fanout}")
            else:
                args.c3_fanout_list = list(fanout)
        return

    k_rollouts = max(1, _safe_int(getattr(args, "n_samples_per_prompt", 8), 8))

    # ---- No-replay knob (can be explicit or inferred from flat fanout) ----
    # Definition: "flat" no-replay fanout means:
    #   fanout[0] == K and fanout[1:] are all 1, e.g. [8,1] for 2A.
    def _is_flat_no_replay_fanout(_fanout, _k):
        try:
            if _fanout is None:
                return False
            _fanout = [int(x) for x in _fanout]
            if len(_fanout) < 2:
                return False
            return int(_fanout[0]) == int(_k) and all(int(x) == 1 for x in _fanout[1:])
        except Exception:
            return False

    if fanout is None:
        # If user explicitly asked no-replay but didn't provide fanout, synthesize a flat one.
        user_no_replay = getattr(args, "c3_no_replay", None)
        if user_no_replay is True:
            fanout = [int(k_rollouts)] + [1] * (len(roles_topo) - 1)
            print(f"[C3][Info] --c3_no_replay set; using flat fanout={fanout} (product={k_rollouts}).")
        elif len(roles_topo) == 2 and k_rollouts == 8:
            fanout = [2, 4]
            print(f"[C3][Info] --c3_fanout not set; using default '{_C3_FANOUT_EXAMPLE_NESTED}' for 2-role K=8.")
        else:
            raise SystemExit(
                "[C3][FAIL-FAST] --c3_fanout is required for C3 training on c3_task.\n"
                f"  roles_topo           : {roles_topo} (len={len(roles_topo)})\n"
                f"  n_samples_per_prompt : {k_rollouts}\n"
                "  Set --c3_fanout as comma-separated ints (one per topo role) and ensure:\n"
                "    product(fanout) == --n_samples_per_prompt\n"
                f"  Example (2A, K=8): --c3_fanout {_C3_FANOUT_EXAMPLE_NESTED}  (nested) "
                f"or --c3_fanout {_C3_FANOUT_EXAMPLE_FLAT}  (flat)\n"
            )

    if len(fanout) != len(roles_topo):
        raise SystemExit(
            "[C3][FAIL-FAST] Invalid --c3_fanout: length mismatch.\n"
            f"  roles_topo  : {roles_topo} (expected len={len(roles_topo)})\n"
            f"  c3_fanout : {fanout} (got len={len(fanout)})\n"
            f"  Example (2A, K=8): --c3_fanout {_C3_FANOUT_EXAMPLE_NESTED}  (nested) "
            f"or --c3_fanout {_C3_FANOUT_EXAMPLE_FLAT}  (flat)\n"
            "  Note: product(fanout) must equal --n_samples_per_prompt."
        )
    if any(int(x) <= 0 for x in fanout):
        raise SystemExit(f"[C3][FAIL-FAST] c3_fanout must be positive ints, got: {fanout}")

    prod = 1
    for x in fanout:
        prod *= int(x)
    if prod != int(k_rollouts):
        raise SystemExit(
            "[C3][FAIL-FAST] Invalid --c3_fanout: product constraint violated.\n"
            f"  roles_topo           : {roles_topo}\n"
            f"  c3_fanout          : {fanout} (product={prod})\n"
            f"  n_samples_per_prompt : {k_rollouts}\n"
            f"  Example (2A, K=8): --c3_fanout {_C3_FANOUT_EXAMPLE_NESTED} or {_C3_FANOUT_EXAMPLE_FLAT}."
        )

    # Finalize c3_no_replay:
    # - If user didn't set it, infer from flat fanout.
    # - If user did set it, keep user choice.
    user_no_replay = getattr(args, "c3_no_replay", None)
    inferred_no_replay = _is_flat_no_replay_fanout(fanout, k_rollouts)
    if user_no_replay is None:
        args.c3_no_replay = bool(inferred_no_replay)
        if args.c3_no_replay:
            print(f"[C3][Info] c3_no_replay inferred from flat fanout={fanout}.")
    else:
        args.c3_no_replay = bool(user_no_replay)

    args.c3_fanout = ",".join(str(x) for x in fanout)
    args.c3_fanout_list = list(fanout)




def _normalize_policy_loss_and_adv(args) -> None:
    """Normalize policy_loss_type and guard incompatible combos."""
    args.policy_loss_type = _normalize_choice(
        getattr(args, "policy_loss_type", None),
        allowed=("ppo", "gspo", "reinforce"),
        default="ppo",
    )
    marl_alg = str(getattr(args, "marl_algorithm", "auto") or "auto").lower().strip()
    if marl_alg == "mappo" and args.policy_loss_type == "reinforce":
        raise SystemExit("ERROR: policy_loss_type=reinforce is not supported for MAPPO. Use policy_loss_type=ppo.")

    if args.policy_loss_type == "reinforce" and bool(getattr(args, "enable_vllm_is_correction", False)):
        print("[Info] policy_loss_type=reinforce: disabling --enable_vllm_is_correction (not used).")
        args.enable_vllm_is_correction = False

    if not _is_unset(getattr(args, "advantage_estimator", None)):
        args.advantage_estimator = str(getattr(args, "advantage_estimator")).lower().strip()


def _normalize_critic_schedule(args) -> None:
    args.critic_target = _normalize_choice(
        getattr(args, "critic_target", None),
        allowed=("auto", "v", "q", "all"),
        default="auto",
    )

    args.critic_warmup_steps = max(0, _safe_int(getattr(args, "critic_warmup_steps", None), 0))
    args.critic_train_steps_per_iter = max(1, _safe_int(getattr(args, "critic_train_steps_per_iter", None), 1))


def _normalize_misc_compat(args) -> None:
    """Compatibility gates and auto-fills across modes."""
    if args.eps_clip_low_high is None:
        args.eps_clip_low_high = (args.eps_clip, args.eps_clip)

    if args.agent_func_path:
        args.remote_rm_url = "agent"

    marl_alg = str(getattr(args, "marl_algorithm", "auto") or "auto").lower().strip()
    is_c3_task = bool(getattr(args, "c3_task", None))

    if is_c3_task:
        if marl_alg == "magrpo":
            if getattr(args, "advantage_estimator", None) == "gae":
                print("[Info] C3 task + MAGRPO: forcing --advantage_estimator=reinforce (no critic).")
                args.advantage_estimator = "reinforce"
            args.critic_pretrain = None
            args.q_critic_pretrain = None
            args.save_q_critic = False

        elif marl_alg == "mappo":
            args.q_critic_pretrain = None
            args.save_q_critic = False

        elif marl_alg == "c3":
            cv = str(getattr(args, "c3_credit_variant", "reward_only") or "reward_only").lower().strip()
            if cv == "reward_only":
                args.q_critic_pretrain = None
                args.save_q_critic = False

    if str(getattr(args, "advantage_estimator", "gae") or "gae").lower().strip() != "gae":
        args.critic_pretrain = None

    if bool(getattr(args, "use_env_reward", False)) and marl_alg == "magrpo":
        if getattr(args, "advantage_estimator", None) == "gae":
            print("[Info] marl_algorithm=magrpo: switching --advantage_estimator gae -> reinforce.")
            args.advantage_estimator = "reinforce"

    if bool(getattr(args, "use_env_reward", False)) and marl_alg == "mappo":
        if getattr(args, "advantage_estimator", None) != "gae":
            print("[Info] marl_algorithm=mappo requires critic: forcing --advantage_estimator=gae.")
            args.advantage_estimator = "gae"

    if args.advantage_estimator == "gae" and args.critic_pretrain is None:
        if bool(getattr(args, "use_env_reward", False)):
            args.critic_pretrain = args.pretrain
        elif not args.remote_rm_url:
            if not args.reward_pretrain:
                raise SystemExit(
                    "ERROR: --reward_pretrain must be set when --remote_rm_url is not provided "
                    "(needed to infer --critic_pretrain)"
                )
            args.critic_pretrain = str(args.reward_pretrain).split(",")[0]
        else:
            args.critic_pretrain = args.pretrain

    if args.remote_rm_url:
        args.remote_rm_url = str(args.remote_rm_url).split(",")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print("[Warning] input_template contains literal \\n. In Bash use $'\\n' or in PowerShell use \"`n\".")

    if args.ring_attn_size > 1 and not args.packing_samples:
        print("[Warning] --ring_attn_size > 1 requires --packing_samples.")
        args.packing_samples = True

    if args.use_dynamic_batch:
        if not args.packing_samples:
            print("[Warning] Please enable --packing_samples when --use_dynamic_batch is enabled.")
            args.packing_samples = True
        if args.rollout_max_tokens_per_gpu is None:
            print("[Warning] Set --rollout_max_tokens_per_gpu to --train_max_tokens_per_gpu.")
            args.rollout_max_tokens_per_gpu = args.train_max_tokens_per_gpu

    if args.packing_samples and "flash_attention" not in args.attn_implementation:
        print("[Warning] Use flash_attention for best performance with --packing_samples.")
        args.attn_implementation = "flash_attention_2"

    if args.vllm_enable_sleep and not args.colocate_all_models:
        print("Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled.")
        args.vllm_enable_sleep = False

    if args.colocate_all_models and args.async_train:
        print("[Warning] --colocate_all_models in async RLHF only colocates DeepSpeed models.")

    if not args.vllm_generate_batch_size:
        args.vllm_generate_batch_size = args.rollout_batch_size


def _normalize_eval_only(args) -> None:
    """Eval-only disables training-only components."""
    if not bool(getattr(args, "eval_only", False)):
        return

    if _safe_float(getattr(args, "init_kl_coef", 0.0), 0.0) > 0.0:
        print("[Info] eval_only: auto-setting --init_kl_coef=0 (disable ref model).")
    args.init_kl_coef = 0

    if getattr(args, "critic_pretrain", None):
        print("[Info] eval_only: disabling critic model (--critic_pretrain=None).")
    args.critic_pretrain = None

    if getattr(args, "remote_rm_url", None):
        print("[Info] eval_only: disabling remote_rm_url.")
    args.remote_rm_url = None

    # EMA doubles the policy-side memory footprint and is not needed in eval_only.
    # Keep it only if the user explicitly asks for it.
    if bool(getattr(args, "enable_ema", False)) and str(os.environ.get("OPENRLHF_EVAL_KEEP_EMA", "0")).lower() not in {"1", "true", "yes"}:
        args.enable_ema = False
        print("[Info] eval_only: disabling EMA (--enable_ema=False).")

    # vLLM: in eval_only we commonly colocate a DS/HF policy model and a vLLM engine on the
    # same visible GPU. The default 0.70 can be too aggressive and make vLLM fail to start
    # once the policy model has allocated some memory. Keep any *smaller* user-specified
    # value; otherwise clamp to a safer default (overridable via env).
    if str(os.environ.get("OPENRLHF_EVAL_KEEP_VLLM_MEM_UTIL", "0")).lower() not in {"1", "true", "yes"}:
        try:
            util = float(getattr(args, "vllm_gpu_memory_utilization", 0.70) or 0.70)
        except Exception:
            util = 0.70
        target = float(os.environ.get("OPENRLHF_EVAL_VLLM_MEM_UTIL", "0.60"))
        if util > target:
            args.vllm_gpu_memory_utilization = target
            print(f"[Info] eval_only: auto-setting --vllm_gpu_memory_utilization={target:.2f}.")
            
    if getattr(args, "vllm_generate_batch_size", None) and args.vllm_generate_batch_size > 32:
        args.vllm_generate_batch_size = 32
        print("[Info] eval_only: capping --vllm_generate_batch_size=32.")


def normalize_args(args) -> Dict[str, object]:
    """
    Normalize/derive args. Does not init Ray and does not create remote actors.
    Returns a small context dict (currently: {"task_spec": ...} when c3_task is provided).
    """
    ctx: Dict[str, object] = {}

    task_spec = None
    if getattr(args, "c3_task", None):
        task_spec = load_task(args.c3_task)
        ctx["task_spec"] = task_spec
        _merge_task_defaults_into_args(args, task_spec)

    # Step6: accept legacy --c3_fanout_list scripts without changing training logic.
    _apply_c3_fanout_list_alias(args)

    _normalize_train_epochs(args)
    _normalize_dumps(args)
    _normalize_analysis_out_dir(args)
    _normalize_vllm_max_len(args)
    _normalize_vllm_num_engines_auto(args, ctx)
    _normalize_eval_only(args)

    if task_spec is not None:
        _detect_env_reward(args, task_spec)

        if not getattr(args, "rollout_generator_cls", None):
            args.rollout_generator_cls = _MAS_ROLLOUT_GENERATOR_CLS
            print(f"[Info] --c3_task detected; auto-setting --rollout_generator_cls={_MAS_ROLLOUT_GENERATOR_CLS}")

        if bool(getattr(args, "use_env_reward", False)) and str(getattr(args, "c3_env_name", "")) in ("MathEnv",):
            if getattr(args, "label_key", None) is None:
                args.label_key = "answer"
                print("[Info] MathEnv env-reward requires labels; auto-setting --label_key=answer")
        elif bool(getattr(args, "use_env_reward", False)) and str(getattr(args, "c3_env_name", "")) in ("CodeEnv",):
            if getattr(args, "label_key", None) is not None:
                print("[Info] CodeEnv env-reward: labels are optional; keeping --label_key as-is.")
            else:
                print("[Info] CodeEnv env-reward: labels are optional; leaving --label_key=None.")

    _normalize_marl(args)

    if task_spec is not None:
        _normalize_c3_task_meta(args, task_spec)

    _normalize_c3_algorithm_args(args)
    _normalize_policy_loss_and_adv(args)
    _normalize_misc_compat(args)
    _normalize_critic_schedule(args)

    return ctx


# ==============================================================================
# Parser building
# ==============================================================================


def _add_ray_resource_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ref_num_nodes", type=int, default=1)
    p.add_argument("--ref_num_gpus_per_node", type=int, default=2)
    p.add_argument("--reward_num_nodes", type=int, default=1)
    p.add_argument("--reward_num_gpus_per_node", type=int, default=2)
    p.add_argument("--actor_num_nodes", type=int, default=1)
    p.add_argument("--actor_num_gpus_per_node", type=int, default=2)
    p.add_argument("--critic_num_nodes", type=int, default=1)
    p.add_argument("--critic_num_gpus_per_node", type=int, default=2)

    p.add_argument("--colocate_actor_ref", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--colocate_critic_reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--colocate_all_models", action=argparse.BooleanOptionalAction, default=True)


def _add_vllm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--vllm_num_engines",
        type=int,
        default=_VLLM_AUTO_ENGINES_SENTINEL,
        help="Number of vLLM engines. Use -1 for auto (default).",
    )
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--vllm_sync_backend", type=str, default="nccl")
    p.add_argument("--vllm_sync_with_ray", action="store_true", default=False)
    p.add_argument("--enable_prefix_caching", action="store_true", default=False)
    p.add_argument("--enforce_eager", action="store_true", default=False)
    p.add_argument("--vllm_enable_sleep", action="store_true", default=False)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.70)
    p.add_argument("--enable_vllm_is_correction", action="store_true", default=False)
    p.add_argument("--vllm_is_truncated_threshold", type=float, nargs=2, default=[0.5, 5.0])
    p.add_argument("--use_icepop", action="store_true", default=False)
    p.add_argument("--vllm_generate_batch_size", type=int, default=32)


def _add_rollout_ext_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--async_train", action="store_true", default=False)
    p.add_argument("--rollout_generator_cls", type=str, default=None)
    p.add_argument("--c3_task", type=str, default=None)


def _add_policy_sharing_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--policy_sharing_mode",
        "--marl_policy_mode",
        dest="policy_sharing_mode",
        type=str,
        default="shared",
        choices=["shared", "per_role"],
    )
    p.add_argument(
        "--per_role_resource_mode",
        type=str,
        default="auto",
        choices=["auto", "compact", "balanced", "expanded"],
        help="per_role resource profile; auto adapts to available GPUs.",
    )
    p.add_argument(
        "--per_role_pg_bundles",
        type=int,
        default=None,
        help="Override per-role PG bundles when policy_sharing_mode=per_role and colocate_all_models=True.",
    )
    p.add_argument(
        "--shared_gpu_fraction_models",
        type=float,
        default=None,
        help="Override GPU fraction for colocated Ray actors (actor/ref/critic/reward) in hybrid mode.",
    )
    p.add_argument(
        "--vllm_force_separate_pg",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force vLLM to create its own PG even when colocate_all_models=True (auto in per_role).",
    )


def _add_marl_args(p: argparse.ArgumentParser) -> None:
    marl_group = p.add_argument_group("marl")
    marl_group.add_argument(
        "--marl_algorithm",
        type=str,
        default="auto",
        choices=["auto", "none", "magrpo", "mappo", "c3", "grpo"],
    )
    marl_group.add_argument("--magrpo_baseline", type=str, default=None, choices=["group_mean", "rloo"])
    marl_group.add_argument(
        "--magrpo_adv_unit",
        type=str,
        default=None,
        choices=["joint_action", "per_role"],
        help="MAGRPO advantage unit: joint_action (paper-aligned) or per_role (legacy).",
    )
    marl_group.add_argument(
        "--magrpo_token_normalize",
        action="store_true",
        default=False,
        help="Normalize MAGRPO advantages per token (optional ablation).",
    )
    marl_group.add_argument(
        "--mappo_normalize_scope",
        type=str,
        default=None,
        choices=["global", "group", "group_role", "episode", "episode_role"],
    )

    mappo_group = p.add_argument_group("MAPPO: centralized critic + step-GAE (role steps)")
    mappo_group.add_argument(
        "--mappo_state_max_len",
        type=int,
        default=None,
        help="Tokenizer max_length for MAPPO state-prompt critic input (default=2560; min=256).",
    )

    mappo_group.add_argument(
        "--mappo_token_normalize",
        dest="mappo_token_normalize",
        action="store_true",
        default=True,
        help=(
            "Normalize MAPPO step-advantages by the number of action tokens per sample before applying PPO loss. "
            "Helps avoid long-response overweighting under token-level loss reduction (default: enabled)."
        ),
    )
    mappo_group.add_argument(
        "--mappo_no_token_normalize",
        dest="mappo_token_normalize",
        action="store_false",
        help="Disable MAPPO token normalization (ablation/back-compat).",
    )


def _add_dump_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dump_first_c3_batch", type=str, default=None)
    p.add_argument("--dump_first_c3_batch_fields", type=str, default="prompts,rewards,info")
    p.add_argument("--dump_first_c3_batch_any", action="store_true", default=False)


def _add_c3_args(p: argparse.ArgumentParser) -> None:
    c3_group = p.add_argument_group("C3")

    c3_group.add_argument(
        "--c3_fanout",
        type=str,
        default=None,
        help=(
            "Comma-separated nested fanout per topo role. "
            "Enforced only for C3 training when marl_algorithm=c3 (not eval_only). "
            "Length must equal topo roles; product must equal --n_samples_per_prompt. "
            f"Example (2 roles, K=8): --c3_fanout {_C3_FANOUT_EXAMPLE_NESTED}"
        ),
    )

    # Step6: legacy alias for old scripts (space-separated).
    # Example: --c3_fanout_list 2 4  (equivalent to --c3_fanout 2,4)
    c3_group.add_argument(
        "--c3_fanout_list",
        dest="c3_fanout_list_arg",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Alias of --c3_fanout using space-separated ints. "
            f"Example (2A, K=8): --c3_fanout_list 2 4  (nested) or 8 1  (flat)."
        ),
    )

    # Step2: ablation knob (No LOO).
    c3_group.add_argument(
        "--c3_baseline_mode",
        type=str,
        default=None,
        choices=list(_C3_BASELINE_MODES),
        help="C3 baseline mode: loo (default) or full_mean (No LOO ablation).",
    )

    # Step2: ablation knob (No replay).
    c3_group.add_argument(
        "--c3_no_replay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "No replay ablation. Prefer flat fanout K,1,1,... (e.g. 2-role K=8 -> --c3_fanout 8,1). "
            "If enabled and --c3_fanout is unset, auto-fill a flat fanout."
        ),
    )

    c3_group.add_argument(
        "--c3_credit_variant",
        type=str,
        default="value_assisted",
        choices=["reward_only", "value_assisted", "value_only"],
        help="C3 credit variant (default: value_assisted).",
    )
    c3_group.add_argument(
        "--c3_va_alpha",
        type=float,
        default=None,
        help="Value-assisted baseline mixing coefficient alpha in [0,1] (default=1.0).",
    )

    c3_group.add_argument("--critic_ctx_limit", type=int, default=2560)
    c3_group.add_argument("--critic_forward_bs", type=int, default=4096)
    c3_group.add_argument("--critic_preamble_path", type=str, default=None)

    c3_group.add_argument("--q_critic_pretrain", type=str, default=None)
    c3_group.add_argument("--q_critic_learning_rate", type=float, default=5e-5)
    c3_group.add_argument("--q_critic_train_batch_size", type=int, default=512)
    c3_group.add_argument(
        "--q_critic_token_cache_size",
        type=int,
        default=0,
        help="LRU cache size for Q-critic tokenizer results (0 disables).",
    )
    c3_group.add_argument(
        "--q_critic_use_multi_steps",
        type=int,
        default=1,
        help=(
            "Use single-RPC multi-step Q-critic training via train_q_critic_multi_steps (default=1). "
            "Set to 0 to fall back to repeated single-step RPCs (debug/compat)."
        ),
    )

    c3_group.add_argument("--c3_critic_loss_type", type=str, choices=["bce", "mse"], default="bce")

    # Step2: dump args must align with rollout_generator._dump_enabled() semantics.
    c3_group.add_argument("--dump_rollouts_jsonl_path", type=str, default=None)
    c3_group.add_argument(
        "--dump_rollouts_every",
        type=int,
        default=None,
        help="Dump rollouts JSONL every N iterations (auto=1 when path is set).",
    )
    c3_group.add_argument(
        "--dump_rollouts_compact",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compact rollouts JSONL dump (analysis-friendly, smaller IO).",
    )
    c3_group.add_argument(
        "--dump_rollouts_max_experiences",
        type=int,
        default=1,
        help="Max experiences to include per rollouts dump record (default: 1).",
    )

    c3_group.add_argument("--dump_c3_batch_data_path", type=str, default=None)
    c3_group.add_argument(
        "--dump_c3_batch_data_max_rows",
        type=int,
        default=8,
        help="Max rows to include per c3_batch_data dump record (default: 8).",
    )
    c3_group.add_argument(
        "--keep_rollout_texts",
        action="store_true",
        default=None,
        help=(
            "Keep large rollout text/debug fields in Experience.info (prompt/state/output/traj_role_*). "
            "Default is auto: enabled when dumping rollouts/batch-data, otherwise disabled to prevent Ray node OOM."
        ),
    )

    c3_group.add_argument("--save_q_critic", action="store_true", default=False)


def _add_eval_and_ckpt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eval_steps", type=int, default=-1)
    p.add_argument(
        "--eval_at_start",
        action="store_true",
        default=True,
        help="Optionally run one evaluation at the very beginning of training (before any updates).",
    )
    p.add_argument(
        "--eval_steps_offset",
        type=int,
        default=0,
        help="Eval trigger offset. Eval happens when (global_step-offset) % eval_steps == 0.",
    )
    p.add_argument("--eval_every_ratio", type=float, default=0.0, help="Evaluate every ratio of total progress (0-1).")
    p.add_argument(
        "--eval_every_percent",
        type=float,
        default=0.0,
        help="Evaluate every percent of total progress (0-100). Overrides --eval_every_ratio if >0.",
    )
    p.add_argument("--eval_only", action="store_true", default=False)
    p.add_argument("--eval_global_step", type=int, default=0)
    p.add_argument("--eval_dump_path", type=str, default=None)
    p.add_argument("--eval_dump_mode", type=str, default="append", choices=["append", "overwrite"])

    p.add_argument("--save_steps", type=int, default=-1)
    p.add_argument(
        "--save_on_eval",
        action="store_true",
        default=False,
        help="Save a checkpoint immediately after each evaluation step (independent of --save_steps).",
    )
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--log_time_breakdown", type=int, default=0, help="Log per-iteration time breakdown (0 disables).")
    p.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    p.add_argument("--save_hf_ckpt", action="store_true", default=False)
    p.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    p.add_argument("--max_ckpt_num", type=int, default=3)
    p.add_argument("--max_ckpt_mem", type=int, default=1e8)
    p.add_argument("--load_checkpoint", action="store_true", default=False)
    p.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)


def _add_deepspeed_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--zero_stage", type=int, default=2)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--deepcompile", action="store_true", default=False)
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--enable_ema", action="store_true")
    p.add_argument("--ema_beta", type=float, default=0.992)
    p.add_argument("--zpg", type=int, default=1)
    p.add_argument("--adam_offload", action="store_true", default=False)
    p.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    p.add_argument("--use_liger_kernel", action="store_true", default=False)
    p.add_argument("--grad_accum_dtype", type=str, default=None)
    p.add_argument("--overlap_comm", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    p.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    p.add_argument("--deepspeed_enable_sleep", action="store_true", default=False)
    p.add_argument("--ds_tensor_parallel_size", type=int, default=1)

    p.add_argument("--packing_samples", action="store_true", default=False)
    p.add_argument("--use_dynamic_batch", action="store_true", default=False)
    p.add_argument("--rollout_max_tokens_per_gpu", type=int, default=None)
    p.add_argument("--train_max_tokens_per_gpu", type=int, default=16192)

    p.add_argument("--load_in_4bit", action="store_true", default=False)


def _add_ppo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--save_path", type=str, default="./ckpt")
    p.add_argument("--train_epochs", type=int, default=None)

    p.add_argument("--critic_target", type=str, default=None, choices=["auto", "v", "q", "all"])
    p.add_argument("--critic_warmup_steps", type=int, default=None)
    p.add_argument("--critic_warmup_prefetch_depth", type=int, default=0, help="Warmup-only rollout prefetch depth.")
    p.add_argument(
        "--critic_warmup_prefetch_timeout",
        type=float,
        default=60.0,
        help="Warmup-only prefetch consumer timeout in seconds.",
    )

    p.add_argument(
        "--critic_warmup_rollout_cache_dir",
        type=str,
        default="",
        help="Warmup rollout cache dir (empty disables).",
    )
    p.add_argument(
        "--critic_warmup_rollout_cache_mode",
        type=str,
        default="auto",
        choices=["auto", "read", "write", "refresh"],
        help="Warmup rollout cache mode.",
    )
    p.add_argument(
        "--critic_warmup_rollout_cache_slim",
        type=int,
        default=-1,
        help="Warmup cache slim mode: 1=slim, 0=full, -1=auto(default).",
    )
    p.add_argument("--critic_warmup_rollout_schedule", type=int, default=1, help="Warmup prompt schedule (1 on, 0 off).")

    p.add_argument(
        "--q_critic_async_overlap",
        type=int,
        default=0,
        help=(
            "Async overlap of Q-critic training (C3 only). 0 disables (default). "
            "Effective only when not colocate_all_models and not deepspeed_enable_sleep; "
            "auto-disabled during critic warmup."
        ),
    )
    p.add_argument("--critic_train_steps_per_iter", type=int, default=8)

    p.add_argument("--rollout_batch_size", type=int, default=512)
    p.add_argument("--micro_rollout_batch_size", type=int, default=8)

    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--prompt_max_len", type=int, default=1024)
    p.add_argument("--generate_max_len", type=int, default=1024)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=1e8)

    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--ptx_coef", type=float, default=0.05)
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--eps_clip_low_high", type=float, nargs=2, default=None)
    p.add_argument("--dual_clip", type=float, default=None)
    p.add_argument("--value_clip", type=float, default=0.5)
    p.add_argument("--lambd", type=float, default=1)
    p.add_argument("--gamma", type=float, default=1)

    p.add_argument("--micro_train_batch_size", type=int, default=4)
    p.add_argument("--train_batch_size", type=int, default=128)
    p.add_argument("--normalize_reward", action="store_true", default=False)

    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full_determinism", action="store_true", default=False)

    p.add_argument("--n_samples_per_prompt", type=int, default=8)
    p.add_argument("--save_value_network", action="store_true", default=False)
    p.add_argument("--actor_learning_rate", type=float, default=1e-6)
    p.add_argument("--critic_learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")

    p.add_argument("--kl_target", type=float, default=None)
    p.add_argument("--kl_horizon", type=float, default=10000)
    p.add_argument("--init_kl_coef", type=float, default=0.01)
    p.add_argument("--policy_loss_type", type=str, default="ppo", choices=["ppo", "gspo", "reinforce"])
    p.add_argument("--kl_estimator", type=str, default="k1", choices=["k1", "k2", "k3"])

    p.add_argument("--aux_loss_coef", type=float, default=0)
    p.add_argument("--entropy_loss_coef", type=float, default=None)
    p.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10))

    p.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"],
        default="gae",
    )
    p.add_argument("--use_kl_loss", action="store_true", default=False)
    p.add_argument("--no_advantage_std_norm", action="store_true", default=False)
    p.add_argument("--overlong_buffer_len", type=float, default=None)
    p.add_argument("--overlong_penalty_factor", type=float, default=1)

    p.add_argument("--ring_attn_size", type=int, default=1)
    p.add_argument("--ring_head_stride", type=int, default=1)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pretrain", type=str, default=None)
    p.add_argument("--pretrain_by_role_json", type=str, default=None)
    p.add_argument("--pretrain_by_role_pattern", type=str, default=None)

    p.add_argument("--reward_pretrain", type=str, default=None)
    p.add_argument("--remote_rm_url", type=str, default=None)
    p.add_argument("--critic_pretrain", type=str, default=None)

    p.add_argument("--value_head_prefix", type=str, default="score")
    p.add_argument("--ref_reward_offload", action="store_true", default=False)
    p.add_argument("--agent_func_path", type=str, default=None)


def _add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prompt_data", type=str, default=None)
    p.add_argument("--prompt_data_probs", type=str, default=None)
    p.add_argument("--prompt_split", type=str, default="train")

    p.add_argument("--eval_dataset", type=str, default=None)
    p.add_argument("--eval_split", type=str, default="train")
    p.add_argument("--eval_temperature", type=float, default=0.7)
    p.add_argument("--eval_n_samples_per_prompt", type=int, default=1)

    p.add_argument("--input_key", type=str, default="input")
    p.add_argument("--label_key", type=str, default=None)
    p.add_argument("--input_template", type=str, default=None)
    p.add_argument("--apply_chat_template", action="store_true", default=False)


def _add_logging_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--use_wandb", type=str, default=None)
    p.add_argument("--wandb_org", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    p.add_argument("--wandb_run_name", type=str, default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"))

    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None)

    p.add_argument("--log_dir", type=str, default=None, help="Local log directory. Default: <run_dir>/logs")
    p.add_argument("--ray_tmpdir", type=str, default=None, help="Ray temp/log directory. Default: <run_dir>/ray_tmp")
    p.add_argument("--log_console", action=argparse.BooleanOptionalAction, default=True, help="Also print logs to console.")
    p.add_argument(
        "--redirect_std_to_log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Redirect stdout/stderr to files under log_dir.",
    )

    p.add_argument(
        "--train_metrics_jsonl_path",
        type=str,
        default=None,
        help="Write training metrics as JSONL (rank0). Default: <run_dir>/train_metrics.jsonl. Empty string disables.",
    )
    p.add_argument("--train_metrics_every", type=int, default=1, help="Write JSONL every N optimization steps (rank0).")

    # Step2: stable analysis output root.
    p.add_argument(
        "--analysis_out_dir",
        type=str,
        default=None,
        help="Analysis output root directory (default: <run_dir>/analysis, else <ckpt_path>/analysis).",
    )

    p.add_argument("--use_tensorboard", type=str, default=None)
    p.add_argument("--perf", action="store_true", default=False)
    p.add_argument("--use_ms", action="store_true", default=False)


def _add_misc_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--reward_provider_cls", type=str, default="auto")
    p.add_argument("--strict_weights_version_check", action="store_true", default=False)
    p.add_argument("--dynamic_filtering", action="store_true", default=False)
    p.add_argument("--dynamic_filtering_reward_range", nargs=2, default=(0, 1), type=float)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    _add_ray_resource_args(p)
    _add_vllm_args(p)
    _add_rollout_ext_args(p)
    _add_policy_sharing_args(p)
    _add_marl_args(p)
    _add_dump_args(p)
    _add_c3_args(p)
    _add_eval_and_ckpt_args(p)
    _add_deepspeed_args(p)
    _add_ppo_args(p)
    _add_model_args(p)
    _add_dataset_args(p)
    _add_logging_args(p)
    _add_misc_args(p)

    return p