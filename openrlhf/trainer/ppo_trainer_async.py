"""
Async PPO trainer (Ray): split rollout generation and training into two actors.

Key memory-safety rule:
- DO NOT send prompts_dataloader.state_dict() per-rollout through Ray queues.
  It can be huge and will explode Ray host memory (object store + heap).
  Use OPENRLHF_ASYNC_DATALOADER_STATE_EVERY=N (>0) to send it sparsely.
"""

# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional, Tuple

import ray
import torch
from tqdm import tqdm

from openrlhf.trainer.ppo_trainer import BasePPOTrainer, _c3_dump_cfg
from openrlhf.trainer.ppo_trainer_plugins import _append_jsonl as _append_jsonl_buffered
from openrlhf.trainer.ppo_trainer_plugins import _is_rank0, _unpack_prompt_batch
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.dynamic_filtering import dyn_filter_update
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

_ASYNC_QUEUE_SIZE_ENV = "OPENRLHF_ASYNC_QUEUE_SIZE"
_ASYNC_DL_STATE_EVERY_ENV = "OPENRLHF_ASYNC_DATALOADER_STATE_EVERY"

_HEAVY_INFO_KEYS = (
    "prompt_text",
    "state_text",
    "output_text",
    "question",
    "traj_role_outputs",
    "traj_role_prompts",
    "c3_path",
    "messages",
)


# ==============================================================================
# Helpers
# ==============================================================================


def _get_train_epochs(args) -> int:
    try:
        return max(1, int(getattr(args, "train_epochs", 1) or 1))
    except Exception:
        return 1


def _get_env_int(name: str, default: int, *, min_v: int = 0) -> int:
    v = os.environ.get(name, None)
    if v is None:
        return max(min_v, int(default))
    try:
        return max(min_v, int(v))
    except Exception:
        return max(min_v, int(default))


def _decode_first_experience(tokenizer, experiences) -> str:
    if not experiences:
        return ""
    exp0 = experiences[0]
    seq = getattr(exp0, "sequences", None)
    if not isinstance(seq, torch.Tensor) or seq.numel() == 0:
        return ""
    try:
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        return tokenizer.batch_decode(seq[:1], skip_special_tokens=True)[0]
    except Exception:
        return ""


# ==============================================================================
# Sync gate between generation and vLLM weight broadcasts
# ==============================================================================


@ray.remote(num_cpus=0)
class SignalActor:
    """Gate generation vs vLLM weight updates (async friendly)."""

    def __init__(self) -> None:
        self.generating_event = asyncio.Event()
        self.update_weights_event = asyncio.Event()
        self.generating_event.set()
        self.update_weights_event.set()

    async def wait_generating(self):
        return await self.generating_event.wait()

    async def wait_update_weights(self):
        return await self.update_weights_event.wait()

    def set_generating(self, allow: bool) -> None:
        if allow:
            self.generating_event.set()
        else:
            self.generating_event.clear()

    def set_update_weights(self, allow: bool) -> None:
        if allow:
            self.update_weights_event.set()
        else:
            self.update_weights_event.clear()


# ==============================================================================
# Generation actor (async rollout sampling)
# ==============================================================================


@ray.remote
class GenerateSamplesActor(BasePPOTrainer):
    """Generate rollouts and push them to a Ray queue."""

    def __init__(self, *args, **kwargs):
        self.signal_actor = kwargs.pop("signal_actor")
        super().__init__(*args, **kwargs)

        self.samples_generator = self.generator_cls(
            self.vllm_engines,
            self.strategy,
            self.tokenizer,
            self.prompt_max_len,
        )
        self.prepare_datasets()

    def generate_samples(self, prompts, labels, **generate_kwargs):
        return self.samples_generator.generate_samples(prompts, labels, **generate_kwargs)

    @staticmethod
    def _wait_for_queue_space(q) -> None:
        ctr = 0
        while q.full():
            if (ctr % 10) == 0:
                logger.info("[async] queue full; waiting for trainer to consume ...")
            ctr += 1
            time.sleep(1)

    @staticmethod
    def _maybe_load_dataloader_state(dl, state: dict) -> None:
        if not state or dl is None or not hasattr(dl, "load_state_dict"):
            return
        try:
            dl.load_state_dict(state)
        except Exception as e:
            logger.warning(f"[WARN] async gen: failed to load prompts_dataloader state_dict, ignoring: {e}")

    def _maybe_capture_dataloader_state(self, rollout_idx_1based: int) -> Optional[dict]:
        every = _get_env_int(_ASYNC_DL_STATE_EVERY_ENV, 0, min_v=0)
        if every <= 0:
            return None
        if (int(rollout_idx_1based) % int(every)) != 0:
            return None
        dl = getattr(self, "prompts_dataloader", None)
        if dl is None or not hasattr(dl, "state_dict"):
            return None
        try:
            return dl.state_dict()
        except Exception:
            return None

    def fit(self, start_episode, queue, data_loader_state_dict, remote_reward_model=None):
        self._maybe_load_dataloader_state(getattr(self, "prompts_dataloader", None), data_loader_state_dict or {})

        train_epochs = _get_train_epochs(self.args)

        for epoch_idx in range(int(start_episode or 0), int(train_epochs)):
            total = None
            try:
                total = int(self.prompts_dataloader.__len__())
            except Exception:
                total = None

            pbar = tqdm(
                total=total,
                desc=f"Generate Epoch [{epoch_idx + 1}/{train_epochs}]",
                disable=False,
            )

            dyn_state: Tuple[list, int] = ([], 0)
            rollout_enqueued = 0

            for batch in self.prompts_dataloader:
                self._wait_for_queue_space(queue)

                # Allow generation; block weight broadcasts while generating.
                ray.get(self.signal_actor.wait_generating.remote())
                ray.get(self.signal_actor.set_update_weights.remote(False))

                try:
                    _, rand_prompts, labels, meta_jsons = _unpack_prompt_batch(batch, where="async generate")
                    gen_kwargs = dict(self.generate_kwargs)
                    if meta_jsons is not None:
                        gen_kwargs["all_metas"] = list(meta_jsons)

                    rollout_samples = self.generate_samples(
                        rand_prompts,
                        labels,
                        remote_reward_model=remote_reward_model,
                        **gen_kwargs,
                    )
                finally:
                    ray.get(self.signal_actor.set_update_weights.remote(True))

                pbar.update(1)

                pass_rate = None
                if self.args.dynamic_filtering:
                    selected, pass_rate, dyn_state = dyn_filter_update(
                        rollout_samples,
                        k=int(self.args.n_samples_per_prompt),
                        rollout_batch_size=int(self.args.rollout_batch_size),
                        reward_range=tuple(self.args.dynamic_filtering_reward_range),
                        state=dyn_state,
                    )
                    if selected is None:
                        buf_groups, _n_total = dyn_state
                        logger.info(
                            "[async gen] filtered_prompts %d < rollout_batch_size %s; continue sampling",
                            len(buf_groups),
                            self.args.rollout_batch_size,
                        )
                        continue

                    rollout_samples = selected
                    logger.info("[async gen] dynamic filtering pass rate: %.2f%%", float(pass_rate))

                rollout_enqueued += 1
                dl_state = self._maybe_capture_dataloader_state(rollout_enqueued)
                queue.put((rollout_samples, int(epoch_idx), dl_state, pass_rate))

            try:
                pbar.close()
            except Exception:
                pass

        queue.put("done")


# ==============================================================================
# Training actor (consume rollouts + PPO updates)
# ==============================================================================


@ray.remote
class TrainingActor(BasePPOTrainer):
    """Consume rollouts, build experiences, append buffers, run PPO updates."""

    def __init__(self, *args, **kwargs):
        self.signal_actor = kwargs.pop("signal_actor")
        self.remote_reward_model = kwargs.pop("remote_reward_model")
        self.q_critic_model_group = kwargs.pop("q_critic_model_group", None)

        super().__init__(*args, **kwargs)

        self.kl_ctl = (
            AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
            if self.kl_target
            else FixedKLController(self.init_kl_coef)
        )

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.tokenizer,
            remote_reward_model=self.remote_reward_model,
            q_critic_model_group=self.q_critic_model_group,
        )

        self._dump_path = getattr(self.strategy.args, "dump_rollouts_jsonl_path", None)
        self._dump_every = int(getattr(self.strategy.args, "dump_rollouts_every", 0) or 0)
        self._rollout_iter = 0

        self._init_wandb()
        self.eval_dataloader = None

    def _broadcast_to_vllm(self) -> None:
        if self.vllm_engines is None:
            return

        ray.get(self.signal_actor.set_generating.remote(False))
        ray.get(self.signal_actor.wait_update_weights.remote())

        self.policy_version += 1
        ray.get(
            self.actor_model_group.async_run_method(
                method_name="broadcast_to_vllm",
                weights_version=self.policy_version,
            )
        )

        ray.get(self.signal_actor.set_generating.remote(True))

    @staticmethod
    def _prune_info_inplace(exp_list, *, keep_texts: bool) -> None:
        if keep_texts or not isinstance(exp_list, list):
            return
        for e in exp_list:
            try:
                e.prompts = []
                e.labels = []
            except Exception:
                pass
            info = getattr(e, "info", None)
            if isinstance(info, dict):
                for k in _HEAVY_INFO_KEYS:
                    info.pop(k, None)

    def _maybe_dump_rollouts_jsonl(self, steps: int, experiences_all) -> None:
        if not (self._dump_path and self._dump_every > 0 and _is_rank0(self.strategy)):
            return
        if (int(self._rollout_iter) % int(self._dump_every)) != 0:
            return

        try:
            payload = {
                "ts": float(time.time()),
                "rollout_iter": int(self._rollout_iter),
                "steps": int(steps),
                "marl_algorithm": str(getattr(self.strategy.args, "marl_algorithm", "")),
                "num_experiences": int(len(experiences_all)) if isinstance(experiences_all, list) else 0,
                "experiences": [self._summarize_experience(e) for e in (experiences_all or [])],
            }
            payload.update(_c3_dump_cfg(self.strategy.args))
            _append_jsonl_buffered(str(self._dump_path), payload)
        except Exception as e:
            try:
                self.strategy.print(f"[WARN] async dump_rollouts_jsonl failed: {e}")
            except Exception:
                pass

    def fit(self, queue, steps: int) -> None:
        args = self.args
        steps = int(steps or 0)

        last_dl_state = None

        while True:
            item = queue.get()
            if item == "done":
                break

            rollout_samples, episode, dl_state, pass_rate = item
            if dl_state is not None:
                last_dl_state = dl_state

            experiences = self.experience_maker.make_experience_batch(rollout_samples)
            experiences_all = experiences

            if args.use_dynamic_batch:
                experiences_all = balance_experiences(experiences_all, args)

            from openrlhf.trainer.ppo_utils.experience_maker import Experience

            keep_texts = bool(getattr(args, "keep_rollout_texts", False))

            actor_payload = Experience.select(
                experiences_all,
                [
                    "index",
                    "sequences",
                    "attention_mask",
                    "action_mask",
                    "action_log_probs",
                    "base_action_log_probs",
                    "rollout_log_probs",
                    "advantages",
                    "returns",
                    "info",
                ],
            )
            self._prune_info_inplace(actor_payload, keep_texts=keep_texts)

            refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=actor_payload)

            if self.critic_model_group is not None:
                critic_payload = Experience.select(
                    experiences_all,
                    [
                        "index",
                        "sequences",
                        "attention_mask",
                        "action_mask",
                        "values",
                        "returns",
                        "critic_input_ids",
                        "critic_attention_mask",
                        "critic_action_mask",
                        "critic_values",
                        "critic_returns",
                        "info",
                    ],
                )
                self._prune_info_inplace(critic_payload, keep_texts=keep_texts)
                refs.extend(self.critic_model_group.async_run_method_batch(method_name="append", experience=critic_payload))

            ray.get(refs)

            status = self.ppo_train(steps)

            if "kl" in status:
                self.kl_ctl.update(status["kl"], int(args.rollout_batch_size) * int(args.n_samples_per_prompt))

            if self.args.dynamic_filtering and pass_rate is not None:
                status["dynamic_filtering_pass_rate"] = pass_rate

            # generated_samples for wandb/tb hooks
            decoded0 = _decode_first_experience(self.tokenizer, experiences_all)
            try:
                status["generated_samples"] = self._make_generated_sample_meta(experiences_all[0], decoded0)
            except Exception:
                status["generated_samples"] = {"text": decoded0}

            self._maybe_dump_rollouts_jsonl(steps, experiences_all)

            logger.info("✨ Global step %s: %s", int(steps), status)

            client_states = {
                "global_step": int(steps),
                "episode": int(episode),
                "data_loader_state_dict": last_dl_state,
            }
            self.save_logs_and_checkpoints(args, int(steps), None, status, client_states)

            self._rollout_iter += 1
            steps += 1

        if getattr(self, "_wandb", None) is not None:
            self._wandb.finish()
        if getattr(self, "_tensorboard", None) is not None:
            self._tensorboard.close()

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict=None, client_states=None):
        """Async path: logging only (checkpointing is handled elsewhere)."""

        if int(global_step) % int(getattr(args, "logging_steps", 1) or 1) != 0:
            return

        if getattr(self, "_wandb", None) is not None:
            ld = dict(logs_dict or {})
            if "generated_samples" in ld:
                try:
                    self._wandb_log_generated_sample(int(global_step), ld.pop("generated_samples"))
                except Exception:
                    pass
            logs = {f"train/{k}": v for k, v in {**ld, "global_step": int(global_step)}.items()}
            self._wandb.log(logs)
            return

        if getattr(self, "_tensorboard", None) is not None:
            for k, v in (logs_dict or {}).items():
                if k == "generated_samples":
                    continue
                try:
                    self._tensorboard.add_scalar(f"train/{k}", v, int(global_step))
                except Exception:
                    pass


# ==============================================================================
# Orchestrator
# ==============================================================================


@ray.remote
class PPOTrainerAsync:
    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__()

        # per_role policy is not supported in async trainer (yet)
        if isinstance(actor_model_group, dict) or isinstance(reference_model_group, dict) or isinstance(vllm_engines, dict):
            raise ValueError(
                "PPOTrainerAsync does NOT support per_role policy (dict actor/ref/vllm). "
                "Disable --async_train or use policy_sharing_mode=shared."
            )

        self.args = strategy.args
        self.strategy = strategy

        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group

        self.vllm_engines = vllm_engines
        self.prompt_max_len = prompt_max_len

        self.signal_actor = SignalActor.remote()

        if self.args.remote_rm_url and not self.args.remote_rm_url[0] == "agent":
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_reward_model = RemoteRewardModel.remote(self.args, self.args.remote_rm_url)
        else:
            self.remote_reward_model = None

        self.q_critic_model_group = generate_kwargs.pop("q_critic_model_group", None)

        self.generate_actor = GenerateSamplesActor.remote(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            signal_actor=self.signal_actor,
            **generate_kwargs,
        )

        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")

        self.trainer_actor = TrainingActor.remote(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            signal_actor=self.signal_actor,
            remote_reward_model=self.remote_reward_model,
            q_critic_model_group=self.q_critic_model_group,
            **generate_kwargs,
        )

        from ray.util.queue import Queue

        qsize = _get_env_int(_ASYNC_QUEUE_SIZE_ENV, 1, min_v=1)
        self.queue = Queue(maxsize=qsize)

    def fit(self) -> None:
        args = self.args

        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[0]
            logger.info("checkpoint_states: %s", checkpoint_states)
            ray.get(self.trainer_actor._broadcast_to_vllm.remote())
        else:
            checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

        steps = int(checkpoint_states.get("global_step", 0)) + 1
        episode = int(checkpoint_states.get("episode", 0) or 0)
        data_loader_state_dict = checkpoint_states.get("data_loader_state_dict", {}) or {}

        remote_reward_model = self.remote_reward_model if self.args.dynamic_filtering else None
        gen_ref = self.generate_actor.fit.remote(episode, self.queue, data_loader_state_dict, remote_reward_model)
        train_ref = self.trainer_actor.fit.remote(self.queue, steps)
        ray.get([gen_ref, train_ref])

    def get_max_steps(self):
        return ray.get(self.generate_actor.get_max_steps.remote())
