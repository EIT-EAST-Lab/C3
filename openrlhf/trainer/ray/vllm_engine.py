# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

import os
import queue
from typing import Any, List

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.utils.logging_utils import init_logger

from .utils import get_bundle_indices, ray_noset_visible_devices

logger = init_logger(__name__)


def _maybe_clamp_vllm_gpu_memory_utilization(kwargs: dict) -> None:
    # Clamp vLLM gpu_memory_utilization to what is feasible on startup.
    # This prevents hard failures when vLLM is colocated with other GPU consumers
    # (e.g., a DS/HF policy model) and the default utilization is too aggressive.
    if str(os.environ.get("OPENRLHF_VLLM_CLAMP_MEM_UTIL", "1")).lower() in {"0", "false", "no"}:
        return
    if "gpu_memory_utilization" not in kwargs:
        return

    try:
        req = float(kwargs.get("gpu_memory_utilization"))
    except Exception:
        return
    if not (0.0 < req < 1.0):
        return
    if not torch.cuda.is_available():
        return

    # torch.cuda.mem_get_info returns (free, total) for the current device.
    # Use a small headroom so vLLM's own allocator + CUDA context creation doesn't race.
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        free_ratio = float(free_b) / float(total_b)
    except Exception:
        return

    try:
        headroom_ratio = float(os.environ.get("OPENRLHF_VLLM_MEM_HEADROOM_RATIO", "0.02"))
    except Exception:
        headroom_ratio = 0.02

    max_util = max(0.05, free_ratio - headroom_ratio)
    if req > max_util:
        new_util = float(f"{max_util:.3f}")
        logger.warning(
            "vLLM gpu_memory_utilization=%.3f is too high for current free memory (free_ratio=%.3f). "
            "Clamping to %.3f. You can disable this via OPENRLHF_VLLM_CLAMP_MEM_UTIL=0.",
            req,
            free_ratio,
            new_util,
        )
        kwargs["gpu_memory_utilization"] = new_util


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


class BaseLLMRayActor:
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        kwargs.pop("agent_func_path", None)
        noset_visible_devices = ray_noset_visible_devices()
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        # Number of actors that will send prompt to this engine
        self.requests = {}
        self.response_queues = queue.Queue()

        full_determinism = kwargs.pop("full_determinism", False)
        if full_determinism:
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        self.kwargs = kwargs
        _maybe_clamp_vllm_gpu_memory_utilization(self.kwargs)

        import vllm
        from packaging import version

        if version.parse(vllm.__version__) >= version.parse("0.9.0"):
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


@ray.remote
class LLMRayActor(BaseLLMRayActor):
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        import vllm

        self.llm = vllm.LLM(*args, **self.kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weights(self, names, dtypes, shapes, empty_cache_last: bool = True):
        """Update many weights in a single Ray task.

        This avoids launching thousands of small Ray tasks (one per parameter),
        which can accumulate scheduler/metadata memory and eventually trigger
        Ray's node OOM killer on long runs.

        Notes:
            - Each element of (names, dtypes, shapes) corresponds to one parameter.
            - The actual tensor data is still sent via the collective broadcast
            inside the vLLM worker (see vllm_worker_wrap.update_weight).
        """
        if not names:
            return True
        assert len(names) == len(dtypes) == len(shapes), "update_weights: meta length mismatch"
        for i, (n, dt, sh) in enumerate(zip(names, dtypes, shapes)):
            empty_cache = bool(empty_cache_last) and (i == len(names) - 1)
            self.llm.collective_rpc("update_weight", args=(n, dt, sh, empty_cache))
        return True

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def update_weights_cuda_ipc(self, names, dtypes, shapes, ipc_handles_list, empty_cache_last: bool = True):
        """Update many weights via CUDA-IPC in a single Ray task."""
        if not names:
            return True
        assert len(names) == len(dtypes) == len(shapes) == len(ipc_handles_list), (
            "update_weights_cuda_ipc: meta/handles length mismatch"
        )
        for i, (n, dt, sh, ipc) in enumerate(zip(names, dtypes, shapes, ipc_handles_list)):
            empty_cache = bool(empty_cache_last) and (i == len(names) - 1)
            self.llm.collective_rpc("update_weight_cuda_ipc", args=(n, dt, sh, ipc, empty_cache))
        return True

    def set_weights_version(self, version: int):
        """Set weights version on all vLLM workers."""
        return self.llm.collective_rpc("set_weights_version", args=(int(version),))

    def get_weights_version(self) -> int:
        """Get weights version and assert all vLLM workers agree."""
        versions = self.llm.collective_rpc("get_weights_version", args=())
        if isinstance(versions, list):
            if not versions:
                return -1
            first = versions[0]
            if any(v != first for v in versions):
                raise RuntimeError(f"vLLM workers disagree on weights_version: {versions}")
            return int(first)
        return int(versions)

    def reset_prefix_cache(self):
        """Best-effort reset for vLLM prefix cache.

        Some vLLM versions may not expose reset_prefix_cache; keep it robust.
        """
        try:
            engine = getattr(self.llm, "llm_engine", None)
            if engine is None:
                return False
            fn = getattr(engine, "reset_prefix_cache", None)
            if callable(fn):
                fn()
                return True
        except Exception:
            return False
        return False

    def sleep(self, level=1):
        # 允许用环境变量全局控制（避免你每次开新终端忘记 export）
        level = int(os.environ.get("VLLM_SLEEP_LEVEL", str(level)))

        # (C3) In long training runs with mostly-unique prompts, vLLM prefix caching can
        # accumulate and eventually trigger severe slowdown (swap / scheduler degradation).
        # Since training commonly uses sleep/wake (colocate), we proactively reset prefix cache
        # on sleep to keep memory bounded.
        if os.environ.get("VLLM_RESET_PREFIX_CACHE_ON_SLEEP", "1").lower() not in ("0", "false", "no"):
            try:
                self.reset_prefix_cache()
            except Exception:
                pass

        self.llm.sleep(level=level)

        # 强制把缓存块尽量还给 CUDA driver，让 colocate 的训练进程能拿到显存
        if os.environ.get("VLLM_SLEEP_EMPTY_CACHE", "1").lower() not in ("0", "false", "no"):
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass

        # 返回一点诊断信息（可选）
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            return {"sleep_level": level, "cuda_free_gb": free_b / 1024**3, "cuda_total_gb": total_b / 1024**3}
        except Exception:
            return {"sleep_level": level}

    def wake_up(self):
        self.llm.wake_up()

    def add_requests(self, sampling_params, prompt_token_ids=None, prompts=None):
        """
        Process requests from rank0 and generate responses.
        Since only rank0 will send requests, we don't need to track actor ranks.
        """
        # Prefer text prompts to avoid HF<->vLLM tokenizer mismatch.
        if prompts is not None:
            # prompts: List[str]
            if len(prompts) == 0:
                # 加固：避免调用方误传空 prompts 导致 get_responses 永久阻塞
                self.response_queues.put([])
                return
            responses = self.llm.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            self.response_queues.put(responses)
            return

        if prompt_token_ids is not None and len(prompt_token_ids) == 0:
            self.response_queues.put([])
            return

        # Backward-compatible: token-id prompts.
        if prompt_token_ids is None:
            raise ValueError("add_requests requires either prompts (List[str]) or prompt_token_ids (List[List[int]]).")

        from vllm.inputs import TokensPrompt

        requests = [TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids]
        responses = self.llm.generate(prompts=requests, sampling_params=sampling_params)
        self.response_queues.put(responses)

    def get_responses(self):
        """
        Return the responses for the actor with the given rank
        """
        return self.response_queues.get()


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    llm_actor_cls=LLMRayActor,
    logprobs_mode=None,
    agent_func_path=None,
):
    import vllm
    from packaging import version

    assert version.parse(vllm.__version__) > version.parse("0.8.2"), "OpenRLHF only supports vllm > 0.8.2"

    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.2

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
        )

        additional_kwargs = {}
        if logprobs_mode:
            additional_kwargs["logprobs_mode"] = logprobs_mode
            additional_kwargs["max_logprobs"] = 1
            assert version.parse(vllm.__version__) > version.parse(
                "0.10.0"
            ), "vLLM > 0.10.0 is required for logprobs_mode"

        vllm_engines.append(
            llm_actor_cls.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_extension_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype="bfloat16",
                trust_remote_code=True,
                full_determinism=full_determinism,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=vllm_enable_sleep,
                agent_func_path=agent_func_path,
                **additional_kwargs,
            )
        )

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep")

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)
