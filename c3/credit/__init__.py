# -*- coding: utf-8 -*-
"""
Credit assignment modules for MARL/C3.

This package hosts the code that converts team-level reward into per-role
advantages: C3's counterfactual credit assignment with a centralized Q-critic.
The pipeline in OpenRLHF will:
- Use a centralized critic (V or Q) to estimate counterfactual baselines
- Convert team reward into per-role scalar advantages
- Broadcast scalar advantages to token-level advantages for PPO updates
"""
