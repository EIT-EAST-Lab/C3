# -*- coding: utf-8 -*-
"""
Credit assignment modules for MARL/C3.

This package hosts the code that converts team-level reward into per-role
advantages: C3's counterfactual credit assignment with a centralized Q-critic.
The pipeline in OpenRLHF will:
- Use a centralized critic (V or Q) to estimate counterfactual baselines
- Convert team reward into per-role scalar advantages
- Broadcast scalar advantages to token-level advantages for PPO updates

NOTE:
  Legacy diagonal counterfactual group construction (prepare_cf_groups / regenerate / variance gating)
  was removed in the Rule-B nested rollout refactor. Rule-B credit assignment uses sibling-group
  leave-one-out baselines (see provider.py + materialize.materialize_c3_tree_groups).
"""
