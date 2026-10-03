"""
openrlhf/trainer/ray/logprob_chunking.py

Row chunking for the log-probability forwards of the experience phase.

Why this exists
---------------
On the MAPPO arm one question group reaches the policy and the reference model as a
single batch of 16 sequences, because `split_rollout_samples` keeps a group atomic
whenever the group is at least as large as `micro_rollout_batch_size`
(`openrlhf/trainer/ppo_utils/experience_maker.py`). The logits of that one forward are
`[16, T, 151936]`, about 19 GiB in bf16 at T of roughly 4300, and the cross-entropy
fallback that runs when flash-attention is unavailable materialises a second tensor of
the same size. Neither existing micro batch can split it: `micro_rollout_batch_size`
only merges small groups on that code path, and `micro_train_batch_size` governs the
training loop rather than this forward.

Why splitting it changes nothing
--------------------------------
A row's log probabilities depend on that row alone: attention is causal and the
attention mask is per row, so row i of the output is a function of row i of the input.
The group as a whole is needed later, by the advantage computation, and that reads the
finished per-row numbers. Splitting the forward at the RPC boundary and joining the
results along dim 0 therefore returns the same tensor. The one reservation is that
bf16 matrix multiplication reduces in an order that depends on the batch dimension, so
equality holds in the same sense as it does when a micro batch changes: the arithmetic
is the same, the last bits need not be.

The split belongs here and not in `Actor.forward` (`openrlhf/models/actor.py`), which
the training loop also calls with gradients on; slicing there would cut into the graph
that gradient accumulation builds. It also lives in a module of its own rather than in
`ppo_actor.py` so that a CPU-only machine can import and test it: `ppo_actor.py` imports
deepspeed and transformers at module scope, and this file imports nothing but torch.
"""

from typing import Any, Callable

import torch


def _slice_rows(value: Any, start: int, stop: int) -> Any:
    """Row slice of a tensor. Anything else, None or an action count, passes through."""
    if isinstance(value, torch.Tensor):
        return value[start:stop]
    return value


def chunked_forward(
    fn: Callable[[Any, Any, Any], torch.Tensor],
    sequences: torch.Tensor,
    action_mask: Any = None,
    attention_mask: Any = None,
    chunk: int = 0,
) -> torch.Tensor:
    """Call `fn` over `chunk` rows of the batch at a time and join the results on the CPU.

    `fn(sequences, action_mask, attention_mask)` is the caller's own forward, its device
    move included, so this function never names a device itself. Each partial result is
    moved to the CPU before the next chunk runs, which is what keeps the peak down, and
    the return value is those parts concatenated along dim 0.

    `chunk` of zero or less, a `sequences` that is not a two-dimensional batch, or a batch
    no larger than one chunk all take the single call the caller made before this switch
    existed, so the default is the old behaviour exactly.
    """
    if chunk <= 0 or sequences.dim() != 2 or int(sequences.shape[0]) <= chunk:
        return fn(sequences, action_mask, attention_mask).to("cpu")

    rows = int(sequences.shape[0])
    parts = []
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        parts.append(
            fn(
                sequences[start:stop],
                _slice_rows(action_mask, start, stop),
                _slice_rows(attention_mask, start, stop),
            ).to("cpu")
        )
    return torch.cat(parts, dim=0)
