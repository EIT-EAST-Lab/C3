# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

from typing import Optional

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import compute_entropy, log_probs_from_logits


class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        attn_implementation (str, optional): Attention mechanism implementation to use. Defaults to "flash_attention_2".
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        ds_config (dict, optional): Configuration for DeepSpeed. Defaults to None.
        device_map (dict, optional): Device mapping. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        attn_implementation="flash_attention_2",
        bf16=True,
        load_in_4bit=False,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        if isinstance(pretrain_or_model, str):
            attn_impl = attn_implementation

            # HuggingFace DeepSpeed integration (avoid global effects)
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841
            else:
                dschf = None  # noqa: F841

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

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

                # set_z3_leaf_modules is required for MoE models
                for m in self.model.modules():
                    if "SparseMoeBlock" in m.__class__.__name__:
                        deepspeed.utils.set_z3_leaf_modules(self.model, [m.__class__])
                        print(f"Setting zero3 leaf for model on class with name: {m.__class__.__name__}")
                        break

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model
            self.packing_samples = getattr(pretrain_or_model, "packing_samples", False)

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
        allgather_logits: bool = False,
        return_logprobs: bool = False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        return_entropy: bool = False,
    ) -> torch.Tensor:
        """Returns action log probs (masked) or full logprobs depending on flags."""
        batch, seqlen = sequences.size()
        forward_attention_mask = attention_mask

        if self.packing_samples:
            # unpad_and_slice_tensor returns rolled_sequences too
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            forward_attention_mask = None
        else:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=forward_attention_mask, position_ids=position_ids)

        # ❗️关键修复：
        # 不要把 logits 整体 .to(float32) —— 这会复制一份 (B,T,V) 的超大张量，直接 OOM。
        # 需要更高精度时，log_probs_from_logits 内部会选择合适的 CE 实现；entropy 也用 chunked。
        logits = output["logits"]

        if return_entropy:
            # compute_entropy is chunked to avoid allocating full softmax(logits)
            assert return_output
            entropy = compute_entropy(logits)
            if self.packing_samples:
                entropy = gather_and_pad_tensor(entropy, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            setattr(output, "entropy", entropy[:, :-1])

        return_action_log_probs = action_mask is not None
        if not return_action_log_probs and not return_logprobs:
            # Only return model output
            assert return_output
            if allgather_logits and self.packing_samples:
                output["logits"] = gather_and_pad_tensor(
                    output["logits"], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                )
            return output

        # Compute per-position log p(label) using CE (memory-friendly; no full log_softmax materialization)
        log_probs = log_probs_from_logits(
            logits,
            rolled_sequences,
            temperature=self.temperature,
            inplace_backward=not bool(return_entropy),
        )

        if self.packing_samples:
            log_probs = gather_and_pad_tensor(log_probs, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

        # Drop last position (since labels are rolled by 1)
        log_probs = log_probs[:, :-1]

        if not return_action_log_probs and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        # ---- Alignment safety checks (prevents silent training bugs) ----
        if action_mask.dim() != 2:
            raise ValueError(f"action_mask must be 2D [B, A], got {tuple(action_mask.shape)}")
        if action_mask.size(0) != log_probs.size(0):
            raise ValueError(
                f"batch mismatch: action_mask.B={action_mask.size(0)} vs log_probs.B={log_probs.size(0)}"
            )
        if action_mask.size(1) > log_probs.size(1):
            raise ValueError(
                f"action_mask length {action_mask.size(1)} > available log_probs length {log_probs.size(1)}; "
                f"this indicates a shift/slice alignment bug."
            )

        # Use only the tail corresponding to action tokens
        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()

        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
