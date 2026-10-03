"""The paper training contract: the three training recipes of the paper, by name.

The paper is "The Trace Is the State: Exact Credit Assignment for LLM Agent
Teams" (arXiv:2603.06859v3). Its hyperparameter tables are the authority for
every value in this module: Table 5 (the settings all three methods share),
Table 6 (the settings specific to C3) and the baseline details of Appendix B.2.
A setting the paper does not state is taken from the configuration of the run
the paper reports, and its comment says so.

Three recipes, one per trained table of the paper:

    main   Table 2: the 5-seed run of the 2-agent workflow (Reasoner, Actor).
    long   Table 7: the long-generation run of the 2-agent workflow, trained
           once and evaluated five times.
    a3     Table 8: the 5-seed run of the 3-agent workflow (Reasoner, Actor,
           Verifier), under the settings of the 2-agent 5-seed run.

`scripts/40_train/paper_train.sh` renders a recipe into the flags of
`openrlhf.cli.train_ppo_ray`, so a reproduction run is configured by the tables
rather than by whatever the trainer's argparse defaults happen to be in a given
revision. Values that already equal a trainer default are spelled out anyway: a
default is a property of one revision of `openrlhf/cli/train_ppo_ray_tooling.py`,
not a promise to the reader, and it can move without anybody noticing here.

Changing a value here changes what this repository claims the paper did.

The command line of this module is what the training script calls:

    python -m c3.utils.paper_train_contract recipes
    python -m c3.utils.paper_train_contract task  --recipe main
    python -m c3.utils.paper_train_contract seeds --recipe main
    python -m c3.utils.paper_train_contract args  --recipe main --method C3
"""

from __future__ import annotations

import argparse
import sys
from types import MappingProxyType
from typing import Mapping, NamedTuple, Optional, Sequence, Tuple

#: Table 5, "Rollout budget B per instance": every method spends 8 terminal runs
#: per training instance. C3 spends them as its fanout tree (Table 6); the two
#: baselines draw 8 independent full-episode rollouts (Appendix B.2).
PAPER_TRAIN_BUDGET_B = 8

PAPER_TRAIN_METHODS = ("MAPPO", "MAGRPO", "C3")

PAPER_TRAIN_METHOD_N_SAMPLES = MappingProxyType(
    {
        "MAPPO": PAPER_TRAIN_BUDGET_B,
        "MAGRPO": PAPER_TRAIN_BUDGET_B,
        "C3": PAPER_TRAIN_BUDGET_B,
    }
)

#: The recipe `scripts/40_train/paper_train.sh` runs when RECIPE is not set.
PAPER_TRAIN_DEFAULT_RECIPE = "main"


class Setting(NamedTuple):
    """One flag of a recipe: the value that follows it, and where it comes from.

    `value` is None for a store_true flag of the trainer, which takes no value.
    """

    value: Optional[str]
    source: str


class PaperTrainRecipe(NamedTuple):
    """One named training recipe of the paper."""

    name: str
    table: str
    summary: str
    task_yaml: str
    train_seeds: Tuple[int, ...]
    #: Flags every method of the recipe passes, in rendering order.
    shared: Mapping[str, Setting]
    #: Flags one method passes on top of the shared ones, in rendering order.
    per_method: Mapping[str, Mapping[str, Setting]]


def _settings(*items: Tuple[str, Optional[str], str]) -> Mapping[str, Setting]:
    out = {}
    for flag, value, source in items:
        if flag in out:
            raise ValueError("flag listed twice in one block: %s" % flag)
        out[flag] = Setting(value, source)
    return MappingProxyType(out)


# -----------------------------------------------------------------------------
# The settings all three recipes share
# -----------------------------------------------------------------------------

#: Table 5 rows that hold for every run of the paper, in rendering order. The
#: optimizer (AdamW) has no flag: it is what the trainer builds. The base model
#: (Qwen3-4B-Instruct-2507) and the seed are arguments of the run, not of the
#: recipe, and the training script supplies them per run.
_SHARED = (
    # Appendix B.2: both roles share the parameters and one objective.
    ("--policy_sharing_mode", "shared", "Appendix B.2"),
    ("--actor_learning_rate", "1e-6", "Table 5, learning rate (actor)"),
    ("--lr_scheduler", "cosine_with_min_lr", "Table 5, learning rate schedule: cosine"),
    ("--lr_warmup_ratio", "0.03", "Table 5, 3% warmup ratio"),
    ("--train_batch_size", "256", "Table 5, batch size"),
    ("--rollout_batch_size", "128", "Table 5, rollout batch size"),
    ("--max_epochs", "1", "Table 5, PPO epochs per batch"),
    ("--eps_clip", "0.2", "Table 5, PPO clip ratio"),
    ("--init_kl_coef", "0.01", "Table 5, KL penalty coefficient (init)"),
    ("--kl_target", "0.1", "Table 5, KL target"),
    ("--kl_estimator", "k3", "Table 5, KL estimator"),
    ("--prompt_max_len", "2560", "Table 5, prompt limit"),
    ("--temperature", "0.7", "Table 5, sampling"),
    ("--top_p", "0.8", "Table 5, sampling"),
    ("--top_k", "20", "Table 5, sampling"),
    ("--l2", "0.0", "Table 5, weight decay"),
    ("--max_norm", "1.0", "Table 5, gradient clipping norm"),
    ("--n_samples_per_prompt", str(PAPER_TRAIN_BUDGET_B), "Table 5, rollout budget B per instance"),
    ("--bf16", None, "Table 5, precision"),
    # The in-training evaluation feeds the monitoring curve only; its decoding is
    # the configuration of the reported runs.
    ("--eval_temperature", "0.7", "run configuration"),
)

#: The 5-seed run (Tables 2 and 8), on top of _SHARED.
_FIVE_SEED = (
    ("--micro_train_batch_size", "64", "Table 5, micro-batch (5-seed run)"),
    ("--micro_rollout_batch_size", "32", "Table 5, micro-batch (5-seed run)"),
    ("--train_epochs", "5", "Table 5, passes over the training problems (5-seed run)"),
    ("--generate_max_len", "512", "Table 5, generation limit (5-seed run)"),
    ("--eval_every_ratio", "0.05", "Table 5, eval interval (5-seed run)"),
    ("--eval_n_samples_per_prompt", "1", "run configuration"),
    ("--use_kl_loss", None, "run configuration: the KL term of Table 5 enters the loss"),
    # Sequence packing with a per-GPU token budget: how the micro-batches of
    # Table 5 are laid out in memory.
    ("--packing_samples", None, "run configuration"),
    ("--use_dynamic_batch", None, "run configuration"),
    ("--train_max_tokens_per_gpu", "32768", "run configuration"),
    ("--rollout_max_tokens_per_gpu", "32768", "run configuration"),
)

#: The long-generation run (Table 7), on top of _SHARED.
_LONG = (
    ("--micro_train_batch_size", "4", "Table 5, micro-batch (long-generation run)"),
    ("--micro_rollout_batch_size", "8", "Table 5, micro-batch (long-generation run)"),
    ("--train_epochs", "2", "Table 5, passes over the training problems (long-generation run)"),
    ("--max_samples", "5000", "Table 5, 5,000 problems sampled per pass (long-generation run)"),
    ("--generate_max_len", "2048", "Table 5, generation limit (long-generation run)"),
    ("--eval_every_ratio", "0.25", "Table 5, eval interval (long-generation run)"),
    ("--eval_n_samples_per_prompt", "4", "run configuration"),
    # Table 7 reports MAGRPO's model at step 250 beside its final model, so the
    # step-250 checkpoint has to outlive the run. The trainer runs
    # max_steps = floor(5000 / 32) x 2 = 312 global steps here (32 prompts per
    # step, the trainer's --vllm_generate_batch_size default, with the last
    # partial batch dropped), and evaluates every round(0.25 x 312) = 78 steps,
    # each evaluation followed by a checkpoint. A save every 250 steps writes
    # global_step250 and nothing else (500 > 312); after it comes one more
    # checkpoint, at the evaluation of step 312, and none at the end of training
    # because 312 is a multiple of 78. The trainer deletes the oldest checkpoint
    # once max_ckpt_num of them are on disk, so 3 keeps global_step250 to the end
    # of the run with one checkpoint to spare.
    ("--save_steps", "250", "Table 7, the step-250 model"),
    ("--max_ckpt_num", "3", "Table 7, the step-250 model"),
)

# -----------------------------------------------------------------------------
# The settings of one method
# -----------------------------------------------------------------------------

#: Table 6, the C3 credit: the reward-only variant with the leave-one-out
#: baseline, no critic.
_C3 = (
    ("--c3_credit_variant", "reward_only", "Table 6, credit variant"),
    ("--c3_baseline_mode", "loo", "Table 6, baseline mode"),
)

#: Table 6: one Reasoner group of n = 2 alternatives, each continued by an Actor
#: group of 4, so 2 x 4 = 8 = B terminal runs per instance.
_C3_FANOUT_2A = (("--c3_fanout", "2,4", "Table 6, branching per role (2 x 4 = B)"),)

#: The 3-agent workflow adds the Verifier, and the budget stays B = 8; the
#: per-role split is the configuration of the reported run.
_C3_FANOUT_3A = (("--c3_fanout", "2,2,2", "run configuration (2 x 2 x 2 = B)"),)

#: Appendix B.2 and Table 5: the centralized critic of MAPPO.
_MAPPO = (
    ("--critic_pretrain", "jhu-clsp/mmBERT-base", "Appendix B.2, mmBERT-base, a separate model"),
    ("--critic_learning_rate", "5e-5", "Table 5, learning rate (critic, MAPPO only)"),
    ("--mappo_normalize_scope", "global", "Appendix B.2"),
    ("--mappo_state_max_len", "2560", "Appendix B.2"),
    ("--critic_warmup_steps", "64", "Appendix B.2, critic warmup of 64 steps"),
)

#: Appendix B.2: the group-relative advantage of MAGRPO.
_MAGRPO = (
    ("--magrpo_baseline", "group_mean", "Appendix B.2"),
    ("--magrpo_adv_unit", "joint_action", "Appendix B.2"),
)

#: Per-method settings of the 5-seed run that the paper does not state.
_MAPPO_FIVE_SEED = (("--critic_train_steps_per_iter", "1", "run configuration"),)
_MAGRPO_FIVE_SEED = (("--magrpo_token_normalize", None, "run configuration"),)
#: The C3 advantage is computed by the multi-agent path of the experience maker,
#: which does not read this flag; with reward_only no critic is built either way.
_C3_FIVE_SEED = (("--advantage_estimator", "reinforce", "run configuration"),)

#: Per-method settings of the long-generation run that the paper does not state.
#: The log-probability chunk is a memory setting: it splits one forward pass into
#: row chunks and returns the same numbers, which is what lets the 2048-token
#: groups of the two baselines fit on an 80 GB card.
_MAPPO_LONG = (
    ("--critic_train_steps_per_iter", "8", "run configuration"),
    ("--logprob_micro_batch_size", "4", "run configuration, memory only"),
)
_MAGRPO_LONG = (("--logprob_micro_batch_size", "4", "run configuration, memory only"),)


def _recipe(
    name: str,
    table: str,
    summary: str,
    task_yaml: str,
    train_seeds: Sequence[int],
    shared: Sequence[Tuple[str, Optional[str], str]],
    per_method: Mapping[str, Sequence[Tuple[str, Optional[str], str]]],
) -> PaperTrainRecipe:
    shared_map = _settings(*shared)
    methods = {}
    for method in PAPER_TRAIN_METHODS:
        block = _settings(*per_method.get(method, ()))
        clash = sorted(set(block) & set(shared_map))
        if clash:
            raise ValueError("recipe %s: %s sets shared flag(s) again: %s" % (name, method, clash))
        methods[method] = block
    return PaperTrainRecipe(
        name=name,
        table=table,
        summary=summary,
        task_yaml=task_yaml,
        train_seeds=tuple(int(s) for s in train_seeds),
        shared=shared_map,
        per_method=MappingProxyType(methods),
    )


PAPER_TRAIN_RECIPES: Mapping[str, PaperTrainRecipe] = MappingProxyType(
    {
        "main": _recipe(
            "main",
            "Table 2",
            "the 5-seed run of the 2-agent workflow, 512-token generation limit",
            "configs/tasks/math_5seed.yaml",
            (0, 1, 2, 3, 4),
            _SHARED + _FIVE_SEED,
            {
                "C3": _C3 + _C3_FANOUT_2A + _C3_FIVE_SEED,
                "MAPPO": _MAPPO + _MAPPO_FIVE_SEED,
                "MAGRPO": _MAGRPO + _MAGRPO_FIVE_SEED,
            },
        ),
        "long": _recipe(
            "long",
            "Table 7",
            "the long-generation run of the 2-agent workflow, 2048-token generation limit, "
            "one trained model per method evaluated five times",
            "configs/tasks/math.yaml",
            (0,),
            _SHARED + _LONG,
            {
                "C3": _C3 + _C3_FANOUT_2A,
                "MAPPO": _MAPPO + _MAPPO_LONG,
                "MAGRPO": _MAGRPO + _MAGRPO_LONG,
            },
        ),
        "a3": _recipe(
            "a3",
            "Table 8",
            "the 5-seed run of the 3-agent workflow, under the settings of the 2-agent 5-seed run",
            "configs/tasks/math_5seed_a3.yaml",
            (0, 1, 2, 3, 4),
            _SHARED + _FIVE_SEED,
            {
                "C3": _C3 + _C3_FANOUT_3A + _C3_FIVE_SEED,
                "MAPPO": _MAPPO + _MAPPO_FIVE_SEED,
                "MAGRPO": _MAGRPO + _MAGRPO_FIVE_SEED,
            },
        ),
    }
)


# -----------------------------------------------------------------------------
# Lookups
# -----------------------------------------------------------------------------


def get_paper_train_recipe(name: Optional[str] = None) -> PaperTrainRecipe:
    key = str(name or PAPER_TRAIN_DEFAULT_RECIPE).strip().lower()
    try:
        return PAPER_TRAIN_RECIPES[key]
    except KeyError as exc:
        supported = ", ".join(PAPER_TRAIN_RECIPES)
        raise KeyError(f"Unknown paper training recipe {name!r}. Expected one of: {supported}") from exc


def _method_key(method: str) -> str:
    key = str(method or "").strip().upper()
    if key not in PAPER_TRAIN_METHOD_N_SAMPLES:
        supported = ", ".join(PAPER_TRAIN_METHOD_N_SAMPLES.keys())
        raise KeyError(f"Unsupported paper training method {method!r}. Expected one of: {supported}")
    return key


def get_paper_train_n_samples(method: str) -> int:
    return int(PAPER_TRAIN_METHOD_N_SAMPLES[_method_key(method)])


def paper_train_settings(recipe: Optional[str], method: str) -> Mapping[str, Setting]:
    """Every flag one (recipe, method) run passes: the shared block, then the method's."""
    r = get_paper_train_recipe(recipe)
    key = _method_key(method)
    merged = dict(r.shared)
    merged.update(r.per_method[key])
    return MappingProxyType(merged)


def render_paper_train_args(recipe: Optional[str] = None, method: str = "C3") -> str:
    """Render one (recipe, method) run as one command line argument string.

    The order is stable: the shared flags in declaration order, then the
    method's. A store_true flag stands alone; every other flag is followed by its
    value. Every token is free of whitespace, so the caller may split the string
    on whitespace.
    """
    parts = []
    for flag, setting in paper_train_settings(recipe, method).items():
        parts.append(flag)
        if setting.value is not None:
            parts.append(setting.value)
    for token in parts:
        if not token or any(ch.isspace() for ch in token):
            raise ValueError(f"recipe token is empty or holds whitespace: {token!r}")
    return " ".join(parts)


# -----------------------------------------------------------------------------
# Command line, for scripts/40_train/paper_train.sh
# -----------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m c3.utils.paper_train_contract",
        description="Print one piece of a paper training recipe.",
    )
    ap.add_argument("what", choices=("recipes", "task", "seeds", "args"))
    ap.add_argument("--recipe", default=PAPER_TRAIN_DEFAULT_RECIPE)
    ap.add_argument("--method", default="")
    args = ap.parse_args(argv)

    try:
        if args.what == "recipes":
            print(" ".join(PAPER_TRAIN_RECIPES))
            return 0
        recipe = get_paper_train_recipe(args.recipe)
        if args.what == "task":
            print(recipe.task_yaml)
        elif args.what == "seeds":
            print(" ".join(str(s) for s in recipe.train_seeds))
        else:
            print(render_paper_train_args(recipe.name, args.method))
    except KeyError as exc:
        print(f"ERROR: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
