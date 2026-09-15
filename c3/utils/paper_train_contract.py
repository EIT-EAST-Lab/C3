"""The paper training contract: this repository's copy of the paper's Table 3.

Table 3 of the paper (the shared hyperparameter table) is the authority for every
value in this module. `scripts/40_train/paper_train.sh` renders the recipe below
into the flags of `openrlhf.cli.train_ppo_ray`, so a reproduction run is
configured by the table rather than by whatever the trainer's argparse defaults
happen to be in a given revision.

Values that already equal a trainer default are spelled out anyway. A default is
a property of one revision of `openrlhf/cli/train_ppo_ray_tooling.py`, not a
promise to the reader, and it can move without anybody noticing here.

Changing a value in `PAPER_TRAIN_RECIPE` or in `PAPER_TRAIN_STORE_TRUE_FLAGS`
changes what this repository claims the paper did. Either change the paper too,
or move the flag into `PAPER_TRAIN_DEPARTURES`, where a departure has to name the
value the paper reports and the document that argues for the difference.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import NamedTuple

PAPER_TRAIN_BUDGET_B = 8

PAPER_TRAIN_METHOD_N_SAMPLES = MappingProxyType(
    {
        "MAPPO": PAPER_TRAIN_BUDGET_B,
        "MAGRPO": PAPER_TRAIN_BUDGET_B,
        "C3": PAPER_TRAIN_BUDGET_B,
    }
)

# Table 3, one entry per row that the trainer exposes as a flag. Keys are the
# flag names of openrlhf/cli/train_ppo_ray_tooling.py; values are the strings
# that follow them on the command line. The order here is the order the rendered
# argument string uses.
#
# Three rows of Table 3 are not in this mapping. The optimizer (AdamW) has no
# flag: it is what the trainer builds. The evaluation budget B lives in
# PAPER_TRAIN_METHOD_N_SAMPLES above, because the flag it feeds depends on the
# method. The seed and the base model are arguments of the run, not of the
# recipe, and the training script supplies them per run.
PAPER_TRAIN_RECIPE = MappingProxyType(
    {
        # Learning rates (the critic rate applies to MAPPO and C3; MAGRPO has no
        # critic to spend it on) and the schedule: cosine with a 3 percent warmup.
        "--actor_learning_rate": "1e-6",
        "--critic_learning_rate": "5e-5",
        "--lr_scheduler": "cosine_with_min_lr",
        "--lr_warmup_ratio": "0.03",
        # Batch size in instances, rollout batch size, and the two micro batches.
        "--train_batch_size": "256",
        "--rollout_batch_size": "128",
        "--micro_train_batch_size": "64",
        "--micro_rollout_batch_size": "32",
        # PPO epochs per batch, and the clip ratio.
        "--max_epochs": "5",
        "--eps_clip": "0.2",
        # The KL penalty: initial coefficient, adaptive target, estimator.
        "--init_kl_coef": "0.01",
        "--kl_target": "0.1",
        "--kl_estimator": "k3",
        # Gradient clipping norm, and the weight decay the trainer calls --l2.
        "--max_norm": "1.0",
        "--l2": "0.0",
        # The prompt half of the effective sequence length. The generation half
        # is a declared departure, see PAPER_TRAIN_DEPARTURES.
        "--prompt_max_len": "2560",
    }
)

# Precision. This one is store_true in the trainer's parser, so it takes no value.
PAPER_TRAIN_STORE_TRUE_FLAGS = ("--bf16",)


class PaperTrainDeparture(NamedTuple):
    """One flag the reproduction runs deliberately set away from Table 3."""

    value: str
    paper_value: str
    reason: str


# Deliberate departures from Table 3. `scripts/40_train/paper_train.sh` passes
# these itself, next to the rendered recipe, so that a reader who diffs the
# script against the table finds the difference explained rather than hidden.
PAPER_TRAIN_DEPARTURES = MappingProxyType(
    {
        "--generate_max_len": PaperTrainDeparture(
            value="2048",
            paper_value="512",
            reason=(
                "docs/32_evaluation_protocol.md, section 2 (Decoding): at a 512 token cap three of "
                "the candidate suites measure truncation rather than ability, and not one AIME "
                "problem reaches an answer inside the cap. Training and evaluation share the 2048 cap."
            ),
        ),
        "--eval_every_ratio": PaperTrainDeparture(
            value="0.10",
            paper_value="0.05",
            reason=(
                "docs/32_evaluation_protocol.md, the table of the two evaluations: the in-training "
                "evaluation runs every ten percent of the run. It is the monitoring curve only, so "
                "its interval does not enter any reported number; the main table comes from "
                "scripts/70_rebuild/final_eval.py on the finished model."
            ),
        ),
    }
)


def get_paper_train_n_samples(method: str) -> int:
    key = str(method or "").strip().upper()
    try:
        return int(PAPER_TRAIN_METHOD_N_SAMPLES[key])
    except KeyError as exc:
        supported = ", ".join(PAPER_TRAIN_METHOD_N_SAMPLES.keys())
        raise KeyError(f"Unsupported paper training method {method!r}. Expected one of: {supported}") from exc


def render_paper_train_args() -> str:
    """Render the recipe as one command line argument string.

    The order is stable: the flags of PAPER_TRAIN_RECIPE in declaration order,
    each followed by its value, then PAPER_TRAIN_STORE_TRUE_FLAGS. Every token is
    free of whitespace, so the caller may split the string on whitespace. The
    departures are not rendered here: the training script passes them itself.
    """
    parts = []
    for flag, value in PAPER_TRAIN_RECIPE.items():
        parts.append(flag)
        parts.append(value)
    parts.extend(PAPER_TRAIN_STORE_TRUE_FLAGS)
    return " ".join(parts)
