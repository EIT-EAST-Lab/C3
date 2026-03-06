# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

import math
import os
from abc import ABC
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional

import ray
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.models import ValueLoss, get_llm_for_sequence_regression
from openrlhf.models.utils import masked_mean
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states

from ..ppo_utils import NaiveReplayBuffer
from .launcher import BaseModelActor

# =============================================================================
# Q-critic view policy (Round B: hard-coded)
# =============================================================================
_Q_INCLUDE_FULL = False
_Q_EXPAND_PREFIX = "all_roles"
_Q_MAX_TEXTS_PER_SAMPLE = 0
_Q_PREFIX_SCOPE = "topo_prefix"


# =============================================================================
# Small helpers
# =============================================================================
def _canonical_alg(args) -> str:
    return str(getattr(args, "marl_algorithm", "") or "").strip().lower()


def _shape(x) -> Optional[tuple]:
    if x is None or not isinstance(x, torch.Tensor):
        return None
    return tuple(x.shape)


def _is_empty_mask(mask: Optional[torch.Tensor]) -> bool:
    return (mask is None) or (not isinstance(mask, torch.Tensor)) or (mask.numel() == 0) or (mask.dim() < 2) or (
        int(mask.shape[1]) == 0
    )


# =============================================================================
# Tokenization cache (Q-critic)
# =============================================================================
class TextEncodeCache:
    """Tiny LRU cache for tokenizer outputs (un-padded), padded on demand."""

    def __init__(self, tokenizer, cache_size: int):
        self.tokenizer = tokenizer
        self.cache_size = max(0, int(cache_size or 0))
        # key -> {"input_ids": list[int], "attention_mask": list[int]}
        self._cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()

    @staticmethod
    def _key(text: str, max_length: int) -> tuple:
        return (str(text), int(max_length))

    def encode(self, texts: List[str], *, max_length: int):
        """Return padded BatchEncoding (pt)."""
        if self.cache_size <= 0:
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_length),
            )

        max_length = int(max_length)
        enc_list: List[Optional[Dict[str, Any]]] = [None] * len(texts)

        missing_texts: List[str] = []
        missing_pos: List[int] = []

        # cache hits
        for i, t in enumerate(texts):
            k = self._key(t, max_length)
            v = self._cache.get(k, None)
            if v is not None:
                self._cache.move_to_end(k, last=True)
                enc_list[i] = v
            else:
                missing_texts.append(str(t))
                missing_pos.append(i)

        # tokenize misses (padding=False) then cache
        if missing_texts:
            new_enc = self.tokenizer(
                missing_texts,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            new_ids = new_enc["input_ids"]
            new_msk = new_enc.get("attention_mask", None)

            for j, t in enumerate(missing_texts):
                ids_j = new_ids[j]
                msk_j = ([1] * len(ids_j)) if new_msk is None else new_msk[j]

                v = {"input_ids": list(ids_j), "attention_mask": list(msk_j)}
                pos = missing_pos[j]
                enc_list[pos] = v

                k = self._key(t, max_length)
                self._cache[k] = v
                self._cache.move_to_end(k, last=True)

                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)

        # pad to tensors
        return self.tokenizer.pad(enc_list, padding=True, return_tensors="pt")  # type: ignore[arg-type]


# =============================================================================
# PPO V-critic trainer (replay-buffer driven)
# =============================================================================
class CriticPPOTrainer(ABC):
    def __init__(
        self,
        strategy,
        critic: torch.nn.Module,
        critic_optim: Optimizer,
        critic_scheduler,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        value_clip: float = 0.2,
        dataloader_pin_memory: bool = True,
        **kwargs,
    ):
        self.strategy = strategy
        self.args = strategy.args

        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_scheduler = critic_scheduler

        self.micro_train_batch_size = int(micro_train_batch_size)
        self.buffer_limit = int(buffer_limit)
        self.buffer_cpu_offload = bool(buffer_cpu_offload)
        self.value_clip = float(value_clip)
        self.dataloader_pin_memory = bool(dataloader_pin_memory)

        self.max_epochs = int(self.args.max_epochs)
        self.replay_buffer = NaiveReplayBuffer(
            self.micro_train_batch_size,
            self.buffer_limit,
            self.buffer_cpu_offload,
            getattr(self.args, "packing_samples", False),
            self.args.use_dynamic_batch,
        )

        self.critic_loss_fn = ValueLoss(self.value_clip)

        # Mixtral aux loss
        self.aux_loss = float(getattr(self.args, "aux_loss_coef", 0.0) or 0.0) > 1e-8

    def ppo_train(self, max_epochs_override: Optional[int] = None):
        # Dynamic batching rebuilds dataset each time
        if self.args.use_dynamic_batch:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        not_shuffle = (
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

        max_epochs = int(max_epochs_override) if max_epochs_override is not None else int(self.max_epochs)
        max_epochs = max(1, max_epochs)

        status_list: List[Dict[str, float]] = []
        for epoch in range(max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for step, experience in enumerate(pbar):
                experience.to_device(device)
                status = self.training_step(experience, step)
                status = self.strategy.all_reduce(status)  # DP
                status_list.append(status)
                pbar.set_postfix(status)

        if not status_list:
            return {}

        # mean aggregate
        status_mean = dict(status_list[0])
        for m in status_list[1:]:
            for k, v in m.items():
                status_mean[k] = status_mean.get(k, 0.0) + float(v)
        for k in list(status_mean.keys()):
            status_mean[k] /= float(len(status_list))
        return status_mean

    @staticmethod
    def _has_all_fields(experience: Experience, keys: List[str]) -> bool:
        for k in keys:
            if getattr(experience, k, None) is None:
                return False
        return True

    @staticmethod
    def _missing_fields(experience: Experience, keys: List[str]) -> List[str]:
        miss: List[str] = []
        for k in keys:
            if getattr(experience, k, None) is None:
                miss.append(k)
        return miss

    def _select_value_batch(self, experience: Experience):
        """Pick centralized macro-step critic fields if present; otherwise fallback to token fields."""
        central_keys = [
            "critic_input_ids",
            "critic_attention_mask",
            "critic_action_mask",
            "critic_values",
            "critic_returns",
        ]
        use_central = self._has_all_fields(experience, central_keys)

        marl_alg = _canonical_alg(self.args)

        # MAPPO: V-critic must never fall back to token-level masks/targets.
        if marl_alg == "mappo" and not use_central:
            missing = self._missing_fields(experience, central_keys)
            raise RuntimeError(
                "[MAPPO][FAIL-FAST] V-Critic must use centralized macro-step critic fields, "
                f"but got use_central=False. Missing={missing}. "
                "This usually means ReplayBuffer dropped critic_* fields or upstream did not build them."
            )

        if use_central:
            sequences = experience.critic_input_ids
            attention_mask = experience.critic_attention_mask
            action_mask = experience.critic_action_mask
            old_values = experience.critic_values
            returns = experience.critic_returns

            # Strong contract: macro-step critic mask must be [B,1].
            if marl_alg == "mappo":
                if not isinstance(action_mask, torch.Tensor) or action_mask.dim() != 2 or int(action_mask.shape[1]) != 1:
                    raise RuntimeError(
                        "[MAPPO][FAIL-FAST] critic_action_mask must be [B,1] for macro-step critic. "
                        f"got shape={_shape(action_mask)}"
                    )

            return sequences, attention_mask, action_mask, old_values, returns

        # token-level critic path
        sequences = experience.sequences
        attention_mask = experience.attention_mask
        action_mask = experience.action_mask
        old_values = experience.values
        returns = experience.returns

        # A clearer error than encoder-only IndexError later.
        if _is_empty_mask(action_mask):
            raise RuntimeError(
                "[V-Critic][FAIL-FAST] token-level action_mask is empty/invalid. "
                f"shape={_shape(action_mask)}. This indicates a bad rollout mask contract upstream."
            )

        return sequences, attention_mask, action_mask, old_values, returns

    def training_step(self, experience: Experience, step: int) -> Dict[str, float]:
        self.critic.train()

        sequences, attention_mask, action_mask, old_values, returns = self._select_value_batch(experience)

        packed_seq_lens = None

        # forward
        values, output = self.critic(
            sequences,
            action_mask=action_mask,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            values_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )

        critic_loss = self.critic_loss_fn(values, old_values, returns, action_mask=action_mask)

        aux_loss = output.aux_loss if self.aux_loss else 0
        loss = critic_loss + aux_loss * float(getattr(self.args, "aux_loss_coef", 0.0) or 0.0)
        if self.args.use_dynamic_batch:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]

        self.strategy.backward(loss, self.critic, self.critic_optim)

        if self.args.use_dynamic_batch:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")
        else:
            self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "values": float(masked_mean(values, action_mask).detach().item()),
            "critic_lr": float(self.critic_scheduler.get_last_lr()[0]),
        }


# =============================================================================
# PPO V-critic actor
# =============================================================================
@ray.remote(num_gpus=1)
class CriticModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps):
        args = strategy.args

        self._setup_distributed(strategy)
        critic = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=strategy.args.normalize_reward,
            attn_implementation=strategy.args.attn_implementation,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=strategy.args.value_head_prefix,
            init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(critic)
        strategy.print("reward normalization status: {}".format(strategy.args.normalize_reward))
        strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

        # tokenizer (only needed if saving value network)
        if strategy.args.save_value_network:
            encoder_only = (not getattr(critic.config, "is_decoder", False)) and (
                not getattr(critic.config, "is_encoder_decoder", False)
            )
            pad_side = "right" if encoder_only else "left"
            self.tokenizer = get_tokenizer(
                pretrain,
                critic,
                pad_side,
                strategy,
                use_fast=not strategy.args.disable_fast_tokenizer,
            )

        critic_optim = strategy.create_optimizer(
            critic,
            lr=args.critic_learning_rate,
            betas=args.adam_betas,
            weight_decay=args.l2,
        )

        critic_scheduler = get_scheduler(
            args.lr_scheduler,
            critic_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        self.critic, self.critic_optim, self.critic_scheduler = strategy.prepare(
            (critic, critic_optim, critic_scheduler),
            is_rlhf=True,
        )

        if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
            ckpt_path = os.path.join(args.ckpt_path, "_critic")
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            strategy.load_ckpt(self.critic, ckpt_path)

        if strategy.args.deepspeed_enable_sleep:
            self.offload_states()

        self.trainer = CriticPPOTrainer(
            strategy,
            critic=self.critic,
            critic_optim=self.critic_optim,
            critic_scheduler=self.critic_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            value_clip=args.value_clip,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
        pad_sequence: bool = False,  # compat: experience_maker may pass this
        **kwargs,  # swallow future kwargs
    ) -> torch.Tensor:
        """Return critic values (CPU float tensor)."""
        device = torch.cuda.current_device()
        self.critic.eval()
        with torch.no_grad():
            if attention_mask is None:
                attention_mask = (sequences != 0).long()
            if action_mask is None:
                action_mask = torch.ones_like(attention_mask, dtype=torch.float32)

            value = self.critic(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                values_allgather=True,
                packed_seq_lens=packed_seq_lens,
            )
        self.critic.train()
        return value.to("cpu")

    def append(self, experience):
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

    def fit(self, num_steps: int = 1):
        """Train critic; reuse buffer for multiple epoch-multipliers then clear once."""
        self.critic.train()

        steps = int(num_steps) if num_steps is not None else 1
        steps = max(1, steps)

        total_epochs = int(getattr(self.trainer, "max_epochs", 1)) * steps
        total_epochs = max(1, total_epochs)

        status = self.trainer.ppo_train(max_epochs_override=total_epochs)
        self.trainer.replay_buffer.clear()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args
        self.strategy.save_model(self.critic, self.tokenizer, args.save_path + "_critic")

    def save_checkpoint(self, tag):
        args = self.strategy.args
        self.strategy.save_ckpt(
            self.critic,
            os.path.join(args.ckpt_path, "_critic"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
        )

    def reload_states(self):
        reload_deepspeed_states(self.critic)

    def offload_states(self):
        offload_deepspeed_states(self.critic)


# Mark append as batched-capable for BaseModelActor.execute_batch
try:
    CriticModelActor.append.__openrlhf_batched__ = True
except Exception:
    pass


# =============================================================================
# Q-critic actor (C3/MAGRPO credit scoring)
# =============================================================================
@ray.remote(num_gpus=1)
class QCriticModelActor(BaseModelActor):
    """Dedicated Q-critic (not PPO value critic)."""

    @staticmethod
    def _extract_logits(model_out: Any) -> torch.Tensor:
        """Best-effort tensor extraction from HF/DS wrappers."""
        if isinstance(model_out, torch.Tensor):
            return model_out
        if hasattr(model_out, "logits") and isinstance(getattr(model_out, "logits"), torch.Tensor):
            return model_out.logits
        if isinstance(model_out, (tuple, list)) and len(model_out) > 0 and isinstance(model_out[0], torch.Tensor):
            return model_out[0]
        if isinstance(model_out, dict):
            for k in ("logits", "scores", "values", "value"):
                v = model_out.get(k, None)
                if isinstance(v, torch.Tensor):
                    return v
        raise TypeError(f"QCritic: unexpected model output type: {type(model_out)}")

    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps):
        args = strategy.args
        self._setup_distributed(strategy)

        # reward-style sequence regression head (EOS scalar)
        init_value_head = False
        try:
            base_pretrain = str(getattr(args, "pretrain", "") or "")
            if base_pretrain and str(pretrain) == base_pretrain:
                init_value_head = True
        except Exception:
            init_value_head = False

        model = get_llm_for_sequence_regression(
            pretrain,
            "reward",
            normalize_reward=False,
            attn_implementation=args.attn_implementation,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=args.value_head_prefix,
            init_value_head=init_value_head,
            packing_samples=args.packing_samples,
        )
        strategy.print(model)

        encoder_only = (not getattr(model.config, "is_decoder", False)) and (
            not getattr(model.config, "is_encoder_decoder", False)
        )
        pad_side = "right" if encoder_only else "left"
        self.tokenizer = get_tokenizer(
            pretrain,
            model,
            pad_side,
            strategy,
            use_fast=not getattr(args, "disable_fast_tokenizer", False),
        )

        cache_sz = int(getattr(args, "q_critic_token_cache_size", 0) or 0)
        self._q_tok_cache = TextEncodeCache(self.tokenizer, cache_sz) if cache_sz > 0 else None
        if cache_sz > 0:
            strategy.print(f"[QCritic] TextEncodeCache enabled: size={cache_sz}")

        q_lr = float(getattr(args, "q_critic_learning_rate", None) or args.critic_learning_rate)
        self.q_critic_optim = strategy.create_optimizer(model, lr=q_lr, betas=args.adam_betas, weight_decay=args.l2)
        self.q_critic_scheduler = get_scheduler(
            args.lr_scheduler,
            self.q_critic_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": q_lr * 0.1},
        )

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        self.q_critic, self.q_critic_optim, self.q_critic_scheduler = strategy.prepare(
            (model, self.q_critic_optim, self.q_critic_scheduler),
            is_rlhf=True,
        )

        if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_q_critic")):
            ckpt_path = os.path.join(args.ckpt_path, "_q_critic")
            strategy.print(f"[QCritic] Loading the checkpoint: {ckpt_path}")
            strategy.load_ckpt(self.q_critic, ckpt_path)

        if args.deepspeed_enable_sleep:
            self.offload_states()

    def _encode_texts(self, texts: List[str], *, max_length: int):
        """Tokenize with optional cache (pt)."""
        cache = getattr(self, "_q_tok_cache", None)
        if cache is not None:
            return cache.encode(texts, max_length=int(max_length))
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length),
        )

    @torch.no_grad()
    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        apply_sigmoid: bool = False,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        """Return logits by default, or probabilities if apply_sigmoid=True (CPU float32)."""
        device = torch.cuda.current_device()
        was_training = bool(getattr(self.q_critic, "training", False))
        self.q_critic.eval()

        if attention_mask is None:
            attention_mask = (sequences != 0).long()

        model_out = self.q_critic(
            sequences.to(device),
            attention_mask=attention_mask.to(device),
            ring_attn_group=self.strategy.ring_attn_group,
            pad_sequence=True,
            packed_seq_lens=packed_seq_lens,
        )
        logits = self._extract_logits(model_out).view(-1)
        out = torch.sigmoid(logits) if apply_sigmoid else logits

        if was_training:
            self.q_critic.train()
        return out.detach().to(dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def score_texts(self, *, texts: List[str], max_len: int = 2048, forward_bs: int = 64):
        """RPC-friendly scoring API: probs in [0,1] (CPU float32)."""
        if not texts:
            return torch.empty(0, dtype=torch.float32)

        bs = max(1, int(forward_bs))
        device = torch.cuda.current_device()
        nb = True

        was_training = bool(getattr(self.q_critic, "training", False))
        self.q_critic.eval()

        outs: List[torch.Tensor] = []
        for s in range(0, len(texts), bs):
            chunk = texts[s : s + bs]
            toks = self._encode_texts(chunk, max_length=int(max_len))

            ids = toks.input_ids.to(device, non_blocking=nb)
            msk = toks.attention_mask.to(device, non_blocking=nb)

            model_out = self.q_critic(
                ids,
                attention_mask=msk,
                ring_attn_group=self.strategy.ring_attn_group,
                pad_sequence=True,
            )
            logits = self._extract_logits(model_out).view(-1)
            probs = torch.sigmoid(logits)
            outs.append(probs.detach().to(dtype=torch.float32, device="cpu"))

            del toks, ids, msk, model_out, logits, probs

        if was_training:
            self.q_critic.train()

        return torch.cat(outs, dim=0)

    # -------------------- Q-critic training --------------------

    @staticmethod
    def _clip01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    @staticmethod
    def _pearsonr_safe(x: torch.Tensor, y: torch.Tensor) -> float:
        """Pearson correlation (robust to tiny variance)."""
        try:
            if x.numel() < 2 or y.numel() < 2:
                return 0.0
            x = x.float()
            y = y.float()
            vx = x - x.mean()
            vy = y - y.mean()
            denom = vx.std(unbiased=False) * vy.std(unbiased=False)
            if float(denom.item()) < 1e-12:
                return 0.0
            corr = (vx * vy).mean() / denom
            return float(corr.item())
        except Exception:
            return 0.0

    @staticmethod
    @contextmanager
    def _temp_disable_ckpt(ds_engine):
        """Temporarily disable HF checkpointing if present (plain or DS engine)."""
        m = getattr(ds_engine, "module", ds_engine)
        bb = getattr(m, "backbone", None)

        if bb is None and hasattr(m, "base_model_prefix"):
            try:
                bb = getattr(m, str(getattr(m, "base_model_prefix")), None)
            except Exception:
                bb = None

        if bb is None:
            yield
            return

        had = False
        try:
            if hasattr(bb, "gradient_checkpointing_disable"):
                bb.gradient_checkpointing_disable()
                had = True
            yield
        finally:
            if had and hasattr(bb, "gradient_checkpointing_enable"):
                bb.gradient_checkpointing_enable()

    def _load_preamble(self, critic_preamble_path: str) -> str:
        """Cache load of critic preamble JSON (C3 semantics)."""
        path = str(critic_preamble_path or "").strip()
        if not path:
            return ""
        prev = getattr(self, "_q_preamble_path", None)
        if prev == path and hasattr(self, "_q_preamble"):
            return str(getattr(self, "_q_preamble", "") or "")
        from c3.credit.c3.provider import load_critic_preamble_from_json

        pre = load_critic_preamble_from_json(path)
        self._q_preamble_path = path
        self._q_preamble = pre
        return pre

    def _build_views_and_targets(self, batch_data: List[dict], roles: List[str], layers, cfg: Dict[str, Any]):
        """Build (texts, targets) for Round B: prefix-only views (one per role)."""
        from c3.credit.c3.baselines import extract_question, format_for_q
        from c3.credit.c3.provider import prepend_preamble

        prefix_scope = _Q_PREFIX_SCOPE
        parents = cfg.get("parents", None)

        pre_path = str(
            cfg.get("critic_preamble_path", "") or getattr(self.strategy.args, "critic_preamble_path", "") or ""
        )
        preamble = self._load_preamble(pre_path) if pre_path else ""

        texts: List[str] = []
        tgts: List[float] = []

        for sample in batch_data:
            obs = sample.get("observation", "")
            q = extract_question(obs)

            joint_actions_by_k = sample.get("joint_actions_by_k", None)
            cands = sample.get("candidates", {}) or {}
            if not isinstance(cands, dict):
                cands = {}

            group_rewards = sample.get("group_rewards", None)
            if not (isinstance(group_rewards, list) and len(group_rewards) > 0):
                raise ValueError(
                    "train_q_critic requires sample['group_rewards'] as a non-empty List[float] (C3 all-k contract)."
                )
            K = int(len(group_rewards))
            tgts_k = [self._clip01(float(x)) for x in group_rewards]

            # Validate action sources
            if joint_actions_by_k is None:
                for r in roles:
                    xs = cands.get(r, None)
                    if not isinstance(xs, list) or len(xs) < K:
                        raise ValueError(
                            f"train_q_critic expects sample.candidates[{r!r}] list len>=K "
                            f"(got {type(xs)} len={len(xs) if isinstance(xs, list) else 'NA'}), K={K}"
                        )
            else:
                if not isinstance(joint_actions_by_k, list) or len(joint_actions_by_k) < K:
                    raise ValueError(
                        "train_q_critic expects sample.joint_actions_by_k List[dict] len>=K, "
                        f"got {type(joint_actions_by_k)} len={len(joint_actions_by_k) if isinstance(joint_actions_by_k, list) else 'NA'}, K={K}"
                    )

            for base_k in range(K):
                # Base joint action dict for this k
                if joint_actions_by_k is not None:
                    ja = joint_actions_by_k[base_k]
                    if not isinstance(ja, dict):
                        raise ValueError(f"joint_actions_by_k[{base_k}] must be a dict, got {type(ja)}")
                    base_actions = {r: str(ja.get(r, "")) for r in roles}
                else:
                    base_actions = {r: str(cands.get(r, [""])[base_k]) for r in roles}

                # Round B: one prefix view per role
                per_views: List[str] = []
                for r in roles:
                    per_views.append(
                        format_for_q(
                            q,
                            base_actions,
                            mode="prefix",
                            up_to_role=r,
                            layers=layers,
                            parents=parents,
                            prefix_scope=prefix_scope,
                            strict=True,
                        )
                    )

                reward_k = float(tgts_k[base_k])
                for t in per_views:
                    texts.append(prepend_preamble(preamble, t))
                    tgts.append(reward_k)

        return texts, tgts

    def train_q_critic(
        self, *, batch_data: List[dict], roles: List[str], layers, cfg: Dict[str, Any]
    ) -> Dict[str, float]:
        """Train Q-critic on Round B prefix-only views."""
        args = self.strategy.args

        texts, tgts = self._build_views_and_targets(batch_data, roles, layers, cfg)
        if not texts:
            return {"q/loss": 0.0, "q/num_texts": 0.0, "q/texts_per_sample": 0.0}

        # best-effort infer K for diagnostics
        k_rollouts = 0
        try:
            ks = []
            for smp in (batch_data or []):
                gr = smp.get("group_rewards", None)
                if isinstance(gr, list):
                    ks.append(len(gr))
            if ks:
                k0 = int(ks[0])
                k_rollouts = k0 if all(int(k) == k0 for k in ks) else int(min(int(k) for k in ks))
        except Exception:
            k_rollouts = 0

        if bool(getattr(args, "deepspeed_enable_sleep", False)):
            try:
                self.reload_states()
            except Exception:
                pass

        device = torch.cuda.current_device()
        N = int(len(texts))

        loss_type = str(cfg.get("loss_type", "mse")).lower()
        bs = max(1, int(cfg.get("train_batch_size", cfg.get("train_bs", 64)) or 64))
        ctx_limit = int(cfg.get("ctx_limit", 2048) or 2048)

        disable_ckpt_during_train = bool(cfg.get("disable_ckpt_during_train", False))
        max_grad_norm = float(cfg.get("max_grad_norm", getattr(args, "max_grad_norm", 0.5) or 0.5))

        amp_enabled = bool(getattr(args, "bf16", False) or getattr(args, "fp16", False))
        amp_dtype = torch.bfloat16 if bool(getattr(args, "bf16", False)) else torch.float16

        tgt_all = torch.tensor(tgts, dtype=torch.float32, device=device)

        # BCE bias (Laplace smoothing)
        logit_bias = 0.0
        pos_prior = float(tgt_all.mean().item()) if N > 0 else 0.0
        if loss_type == "bce" and N > 0:
            pos_sum = float(tgt_all.sum().item())
            pi = (pos_sum + 0.5) / (N + 1.0)
            pi = float(min(max(pi, 1e-6), 1.0 - 1e-6))
            logit_bias = float(math.log(pi / (1.0 - pi)))
            pos_prior = pi

        self.q_critic.train()
        self.q_critic_optim.zero_grad(set_to_none=True)

        cm = self._temp_disable_ckpt(self.q_critic) if disable_ckpt_during_train else nullcontext()

        loss_sum = 0.0
        mse_sum = 0.0
        bce_sum = 0.0
        pred_mean_sum = 0.0
        pred_std_sum = 0.0

        logit_adj_sum = 0.0
        logit_adj_sumsq = 0.0
        logit_adj_count = 0

        tgt_mean = float(tgt_all.mean().item()) if N > 0 else 0.0
        tgt_std = float(tgt_all.std().item()) if N > 1 else 0.0
        if tgt_std != tgt_std:
            tgt_std = 0.0

        pearson_pairs: List[tuple[torch.Tensor, torch.Tensor]] = []

        with cm:
            for s in range(0, N, bs):
                e = min(N, s + bs)
                w = float(e - s) / float(N)

                chunk = texts[s:e]
                toks = self._encode_texts(chunk, max_length=ctx_limit)

                ids = toks.input_ids.to(device, non_blocking=True)
                msk = toks.attention_mask.to(device, non_blocking=True)
                tgt = tgt_all[s:e]

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    model_out = self.q_critic(
                        ids,
                        attention_mask=msk,
                        ring_attn_group=self.strategy.ring_attn_group,
                        pad_sequence=True,
                    )
                    logits = self._extract_logits(model_out).view(-1).float()
                    if loss_type == "bce":
                        logits_adj = logits + float(logit_bias)
                        loss = F.binary_cross_entropy_with_logits(logits_adj, tgt, reduction="mean")
                    else:
                        loss = (logits - tgt).pow(2).mean()

                self.strategy.backward(loss * w, self.q_critic, self.q_critic_optim)

                with torch.no_grad():
                    if loss_type == "bce":
                        logits_adj = logits + float(logit_bias)

                        _la = logits_adj.detach().to(torch.float32)
                        logit_adj_sum += float(_la.sum().item())
                        logit_adj_sumsq += float((_la * _la).sum().item())
                        logit_adj_count += int(_la.numel())

                        q_prob = torch.sigmoid(logits_adj.detach())
                        pearson_pairs.append(
                            (q_prob.to(dtype=torch.float32, device="cpu"), tgt.detach().to(dtype=torch.float32, device="cpu"))
                        )

                        pred_mean_sum += float(q_prob.mean().item()) * w
                        _std = float(q_prob.std().item()) if q_prob.numel() > 1 else 0.0
                        if _std != _std:
                            _std = 0.0
                        pred_std_sum += _std * w

                        bce_sum += float(loss.detach().item()) * w
                        loss_sum += float(loss.detach().item()) * w
                    else:
                        pred = logits.detach()
                        pearson_pairs.append(
                            (pred.to(dtype=torch.float32, device="cpu"), tgt.detach().to(dtype=torch.float32, device="cpu"))
                        )

                        pred_mean_sum += float(pred.mean().item()) * w
                        _std = float(pred.std().item()) if pred.numel() > 1 else 0.0
                        if _std != _std:
                            _std = 0.0
                        pred_std_sum += _std * w

                        mse_sum += float(loss.detach().item()) * w
                        loss_sum += float(loss.detach().item()) * w

                del toks, ids, msk, tgt, model_out, logits, loss

        from openrlhf.utils.utils import safe_get_global_grad_norm

        is_ds = hasattr(self.q_critic, "step") and callable(getattr(self.q_critic, "step"))
        if (not is_ds) and max_grad_norm is not None and float(max_grad_norm) > 0:
            try:
                _ = torch.nn.utils.clip_grad_norm_(self.q_critic.parameters(), float(max_grad_norm))
            except Exception:
                pass

        gnorm_f = 0.0
        if not is_ds:
            try:
                gnorm_f = float(safe_get_global_grad_norm(self.q_critic))
            except Exception:
                gnorm_f = 0.0

        self.strategy.optimizer_step(self.q_critic_optim, self.q_critic, self.q_critic_scheduler, name="q_critic")

        if is_ds:
            try:
                gnorm_f = float(safe_get_global_grad_norm(self.q_critic))
            except Exception:
                gnorm_f = 0.0

        gnorm_avail = 1.0 if (gnorm_f == gnorm_f and gnorm_f > 0.0) else 0.0

        pred_cpu: Optional[torch.Tensor] = None
        tgt_cpu: Optional[torch.Tensor] = None
        q_pearson = 0.0
        with torch.no_grad():
            if pearson_pairs:
                pred_cpu = torch.cat([p.detach().reshape(-1) for (p, _) in pearson_pairs], dim=0)
                tgt_cpu = torch.cat([t.detach().reshape(-1) for (_, t) in pearson_pairs], dim=0)
                q_pearson = self._pearsonr_safe(pred_cpu, tgt_cpu)

        # Per-view: Round B is prefix-only => one view per role
        q_pearson_by_view: Dict[str, float] = {}
        q_pred_std_by_view: Dict[str, float] = {}
        try:
            n_texts = int(pred_cpu.numel()) if pred_cpu is not None else 0
            n_samples = int(len(batch_data))
            if (
                pred_cpu is not None
                and tgt_cpu is not None
                and n_samples > 0
                and n_texts % n_samples == 0
                and int(k_rollouts) > 0
            ):
                texts_per_sample_i = n_texts // n_samples
                if texts_per_sample_i % int(k_rollouts) == 0:
                    views_per_k = texts_per_sample_i // int(k_rollouts)

                    view_names = [f"prefix/{r}" for r in roles]
                    if len(view_names) != views_per_k:
                        view_names = [f"view{v}" for v in range(views_per_k)]

                    idx = torch.arange(n_texts)
                    view_slot = (idx % texts_per_sample_i) % views_per_k
                    for v in range(views_per_k):
                        m = view_slot == v
                        if int(m.sum().item()) < 2:
                            continue
                        p_v = pred_cpu[m]
                        t_v = tgt_cpu[m]
                        name = str(view_names[v]).replace("/", "_")
                        q_pearson_by_view[name] = self._pearsonr_safe(p_v, t_v)
                        q_pred_std_by_view[name] = float(p_v.std().item())
        except Exception:
            q_pearson_by_view = {}
            q_pred_std_by_view = {}

        pred_mean = float(pred_mean_sum)
        pred_std = float(pred_std_sum)

        lr = 0.0
        try:
            lr = float(self.q_critic_scheduler.get_last_lr()[0])
        except Exception:
            lr = 0.0

        out: Dict[str, float] = {
            "q/loss": float(loss_sum),
            "q/pred_mean": float(pred_mean),
            "q/pred_std": float(pred_std),
            "q/target_mean": float(tgt_mean),
            "q/target_std": float(tgt_std),
            "q/pearson": float(q_pearson),
            "q/num_texts": float(N),
            "q/texts_per_sample": float(N / max(1, len(batch_data))),
            "q/lr": float(lr),
            "q/grad_norm": float(gnorm_f),
            "q/grad_norm_available": float(gnorm_avail),
            "q/grad_norm_is_ds": float(1.0 if is_ds else 0.0),
        }

        if logit_adj_count > 0:
            _m = logit_adj_sum / logit_adj_count
            _v = (logit_adj_sumsq / logit_adj_count) - (_m * _m)
            out["q/logit_adj_mean"] = float(_m)
            out["q/logit_adj_std"] = float((_v if _v > 0 else 0.0) ** 0.5)
        else:
            out["q/logit_adj_mean"] = 0.0
            out["q/logit_adj_std"] = 0.0

        for _k, _v in q_pearson_by_view.items():
            out[f"q/pearson_{_k}"] = float(_v)
        for _k, _v in q_pred_std_by_view.items():
            out[f"q/pred_std_{_k}"] = float(_v)

        if loss_type == "bce":
            out["q/bce"] = float(bce_sum)
            out["q/pos_prior"] = float(pos_prior)
            out["q/logit_bias"] = float(logit_bias)
        else:
            out["q/mse"] = float(mse_sum)

        if bool(getattr(args, "deepspeed_enable_sleep", False)):
            try:
                self.offload_states()
            except Exception:
                pass

        return out

    def train_q_critic_multi_steps(self, batch_data, roles, layers, cfg: dict, num_steps: int = 1):
        """Train Q-critic multiple steps; aggregate metrics by mean."""
        n = int(num_steps or 0)
        if n <= 1:
            return self.train_q_critic(batch_data=batch_data, roles=roles, layers=layers, cfg=cfg)

        stats_list: List[Dict[str, float]] = []
        for _ in range(n):
            out = self.train_q_critic(batch_data=batch_data, roles=roles, layers=layers, cfg=cfg)
            if isinstance(out, dict):
                stats_list.append({k: float(v) for k, v in out.items() if isinstance(v, (int, float))})
            elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                stats_list.append({k: float(v) for k, v in out[0].items() if isinstance(v, (int, float))})
            else:
                raise TypeError(f"train_q_critic_multi_steps: unexpected train_q_critic return type: {type(out)}")

        agg: Dict[str, float] = {}
        if stats_list:
            keys = set()
            for d in stats_list:
                keys.update(d.keys())
            for k in keys:
                vv: List[float] = []
                for d in stats_list:
                    v = d.get(k, None)
                    if isinstance(v, (int, float)) and v == v:
                        vv.append(float(v))
                if vv:
                    agg[k] = float(sum(vv) / float(len(vv)))

        agg["q/multi_steps"] = float(int(n))
        return agg

    def save_model(self):
        args = self.strategy.args
        self.strategy.save_model(self.q_critic, self.tokenizer, args.save_path + "_q_critic")

    def save_checkpoint(self, tag):
        args = self.strategy.args
        self.strategy.save_ckpt(
            self.q_critic,
            os.path.join(args.ckpt_path, "_q_critic"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
        )

    def reload_states(self):
        reload_deepspeed_states(self.q_critic)

    def offload_states(self):
        offload_deepspeed_states(self.q_critic)
