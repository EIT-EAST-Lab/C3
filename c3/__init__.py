"""C3: counterfactual credit assignment for multi-agent LLM training.

The package sits beside the vendored `openrlhf/` tree rather than inside it, so
the upstream core (cli, trainer, ray, utils) stays readable against its origin
and every C3 addition is one import away.

Subpackages:
  task       the loaders for the task and role files under `configs/`
  protocol   the role graph, the prompt renderer and the rollout generator
  credit     the counterfactual credit provider, its critic input and scoring
  baselines  the comparison advantage calculators: MAPPO, MAGRPO, group baseline
  envs       task environments and their reward functions
  rewards    the reward provider registry the rollout generator builds from
  analysis   offline replay, buckets, metrics and the rebuild study
  reporting  the CLIs that turn run artifacts into the tables and figures
  utils      shared helpers: budget ledger, collision guard, environment check
"""

from importlib import resources as _resources

__all__ = ["task", "protocol", "envs", "baselines", "credit"]


def package_root() -> str:
    """Return the filesystem path of this package root (best-effort)."""
    return str(_resources.files(__package__))
