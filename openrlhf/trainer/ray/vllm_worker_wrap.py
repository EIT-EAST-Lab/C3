# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

import torch
import inspect
import os

from openrlhf.trainer.ray.utils import get_physical_gpu_id


class WorkerWrap:
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=False
    ):
        """Init torch process group for model weights update"""
        import torch
        from openrlhf.utils.distributed_util import stateless_init_process_group

        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_with_ray = use_ray
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                self.device,
            )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import torch

        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        ipc_handle = ipc_handles[get_physical_gpu_id()]
        func, args = ipc_handle

        # Always use an int device id that is valid in *this* process.
        device_id = torch.cuda.current_device()

        # Robustly locate the "storage_device" arg across torch versions.
        try:
            params = list(inspect.signature(func).parameters)
        except Exception:
            params = []
        idx = None
        for key in ("storage_device", "device", "cuda_device", "device_id"):
            if key in params:
                idx = params.index(key)
                break
        # torch<=2.5 typical: storage_device is at index 6
        if idx is None and len(args) >= 7:
            idx = 6

        new_args = list(args)
        if idx is not None:
            new_args[idx] = device_id

        if os.environ.get("OPENRLHF_DEBUG_CUDA_IPC", "0") == "1":
            print(
                f"[CUDA_IPC] name={name} uuid={get_physical_gpu_id()} "
                f"func={getattr(func,'__name__',type(func))} idx={idx} device_id={device_id}"
            )

        weight = func(*new_args)

        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight
        torch.cuda.synchronize()

    # ---- Strict rollout sync helpers ----
    def set_weights_version(self, version: int):
        """Record a monotonically increasing weights version.

        After the actor broadcasts new weights to vLLM, we also set a version
        number so the trainer can verify rollout uses the latest weights.
        """
        self._weights_version = int(version)
        return self._weights_version

    def get_weights_version(self) -> int:
        return int(getattr(self, "_weights_version", -1))
