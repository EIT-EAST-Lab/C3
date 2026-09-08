# openrlhf/models/model.py
# -*- coding: utf-8 -*-

# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

from typing import Optional

import deepspeed
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from openrlhf.utils.logging_utils import init_logger

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor

logger = init_logger(__name__)


def _pool_encoder_hidden(
    last_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    mode: str = "first_nonpad",
) -> torch.Tensor:
    """
    Robust pooling for encoder-only models.

    Why:
      - Under LEFT padding, hidden[:, 0, :] can be PAD (bad if you assume CLS at position 0).
      - Under RIGHT padding, hidden[:, -1, :] can be PAD if you use max_length padding (very common).

    mode:
      - "first_nonpad": pick the first position where attention_mask==1 (CLS/BOS under left-padding)
      - "last_nonpad":  pick the last position where attention_mask==1 (often EOS under tokenizers that add EOS)
    """
    if attention_mask is None:
        # fallback: assume token 0 is a good pooled token (CLS/BOS)
        return last_hidden_states[:, 0, :]

    am = attention_mask.to(dtype=torch.long)

    if mode == "last_nonpad":
        # last_nonpad = sum(mask)-1
        idx = am.sum(dim=1) - 1
        idx = torch.clamp(idx, min=0)
    else:
        # first_nonpad = argmax(mask) (works for both left/right padding)
        idx = am.argmax(dim=1)

    bidx = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
    return last_hidden_states[bidx, idx, :]


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    normalize_reward=False,
    attn_implementation="flash_attention_2",
    ds_config: dict = None,
    init_value_head=False,
    value_head_prefix="score",
    device_map=None,
    packing_samples=False,
    **kwargs,
) -> nn.Module:
    """Retrieve a transformer model with a sequence regression head on top.

    This function loads a pretrained transformer model and attaches a linear layer for sequence regression.

    Args:
        model_name_or_path (str): Path to the pretrained model.
        model_type (str): Type of the model, either "reward" or "critic".
        bf16 (bool, optional): Enable bfloat16 precision. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        normalize_reward (bool, optional): Normalize reward values. Defaults to False.
        attn_implementation (str, optional): Attention implementation. Defaults to "flash_attention_2".
        ds_config (dict, optional): Deepspeed configuration for model partitioning across GPUs when ZeRO-3 is enabled.
        init_value_head (bool, optional): Force-initialize the value head (even if present in checkpoint). Defaults to False.
        value_head_prefix (str, optional): Name/prefix for the value head. Defaults to "score".
        device_map (dict, optional): Map of devices for model loading. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.

    Returns:
        nn.Module: A pretrained transformer model with a sequence regression head.
    """
    assert model_type in ("critic", "reward"), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = attn_implementation

    # Prioritize using the value_head_prefix in the model configuration.
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    else:
        cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    # Ask HF to return loading info so we can detect (and safely initialize)
    # newly-added heads under ZeRO-3. Without this, value heads that are not
    # present in the checkpoint can remain effectively uninitialized when the
    # model is constructed inside deepspeed.zero.Init(), leading to NaNs or
    # extreme logits at the first training step.
    model, loading_info = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        output_loading_info=True,
        **kwargs,
    )

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

        # set_z3_leaf_modules is required for MoE models
        for m in model.modules():
            # https://github.com/microsoft/DeepSpeed/pull/4966
            if "SparseMoeBlock" in m.__class__.__name__:
                deepspeed.utils.set_z3_leaf_modules(model, [m.__class__])
                print(f"Setting zero3 leaf for model on class with name: {m.__class__.__name__}")
                break

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    missing_keys = set((loading_info or {}).get("missing_keys", []) or [])
    need_init_value_head = bool(init_value_head) or any(str(k).startswith(value_head_prefix) for k in missing_keys)

    if need_init_value_head:
        value_head = getattr(model, value_head_prefix)
        params = [value_head.weight]
        if getattr(value_head, "bias", None) is not None:
            params.append(value_head.bias)

        # Use model's initializer_range if available (BERT-style defaults to 0.02).
        init_std = float(getattr(config, "initializer_range", 0.02))

        if dschf is not None:
            logger.info(
                f"initialize {value_head_prefix} for ZeRO-3 sequence-regression model "
                f"(init_value_head={init_value_head}, missing={any(str(k).startswith(value_head_prefix) for k in missing_keys)}), "
                f"std={init_std}"
            )
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=init_std)
                    if getattr(value_head, "bias", None) is not None:
                        value_head.bias.data.zero_()
        else:
            value_head.weight.data.normal_(mean=0.0, std=init_std)
            if getattr(value_head, "bias", None) is not None:
                value_head.bias.data.zero_()

    return model


def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples
            # Encoder-only models (e.g., BERT/mmBERT) typically use CLS/BOS pooling for sequence regression.
            # Under left-padding, token 0 can be PAD; so we pool at first non-pad (CLS/BOS).
            # If you *really* want EOS pooling, set config.encoder_pooling = "last_nonpad".
            self._encoder_only = (not getattr(config, "is_decoder", False)) and (
                not getattr(config, "is_encoder_decoder", False)
            )

            # NOTE: packing_samples is designed for decoder-style token pooling (e.g., EOS of each packed sample).
            # Encoder-only sequence regression typically uses CLS/BOS pooling, which does not have a well-defined
            # per-sample pooling position under packed layouts.
            #
            # We therefore *gracefully* fall back to non-packed mode for encoder-only reward / Q-critic models,
            # while still allowing packing for decoder-only actor/ref training.
            if self._encoder_only and self.packing_samples:
                logger.warning(
                    "packing_samples=True is not supported for encoder-only reward/Q-critic (CLS pooling); "
                    "auto-disabling packing for this model (encoder-only)."
                )
                self.packing_samples = False

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            pad_sequence=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            batch, seqlen = input_ids.size()

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            forward_attention_mask = attention_mask

            if self.packing_samples:
                input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group
                )
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=forward_attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]

            if self._encoder_only:
                pooling_mode = getattr(self.config, "encoder_pooling", "first_nonpad")
                pooled = _pool_encoder_hidden(last_hidden_states, attention_mask, mode=pooling_mode)
                reward = getattr(self, self.value_head_prefix)(pooled).squeeze(-1)

                if (not self.training) and self.normalize_reward:
                    reward = (reward - self.mean) / self.std

                return (reward, outputs) if return_output else reward

            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if (not self.training) and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel


def _get_critic_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples
            # Encoder-only models use pooled scalar value for the whole sequence.
            self._encoder_only = (not getattr(config, "is_decoder", False)) and (
                not getattr(config, "is_encoder_decoder", False)
            )
            if self._encoder_only and self.packing_samples:
                # Critic/V-critic for encoder-only (e.g., BERT-style) uses a single pooled scalar.
                # Packing concatenates multiple samples into one sequence and would require segment-aware pooling.
                # We therefore disable packing for this model only and keep packing enabled for decoder-only models.
                logger.warning(
                    "packing_samples=True was requested, but encoder-only critic/V-critic (CLS pooling) does not "
                    "support sample packing. Disabling packing for this critic model instance."
                )
                self.packing_samples = False

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            values_allgather=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            batch, seqlen = input_ids.size()

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

            forward_attention_mask = attention_mask

            if self.packing_samples:
                input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group
                )
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=forward_attention_mask, position_ids=position_ids
            )

            if action_mask is None:
                assert return_output
                return outputs

            last_hidden_states = outputs["last_hidden_state"]

            if self._encoder_only:
                pooling_mode = getattr(self.config, "encoder_pooling", "first_nonpad")
                pooled = _pool_encoder_hidden(last_hidden_states, attention_mask, mode=pooling_mode)

                # pooled scalar value
                v = getattr(self, self.value_head_prefix)(pooled).squeeze(-1)

                # normalize reward
                if self.normalize_reward:
                    v = (v - self.mean) / self.std

                # Align to action_mask width; encoder-only critic outputs a single scalar.
                out = torch.zeros((batch, action_mask.shape[1]), device=v.device, dtype=v.dtype)
                out[:, 0] = v
                action_values = out * action_mask.float()

                if return_output:
                    return (action_values, outputs)
                else:
                    return action_values

            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)  # (B, S)

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

            values = values[:, :-1]

            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            action_values = values[:, -action_mask.shape[1] :] * action_mask.float()

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel
