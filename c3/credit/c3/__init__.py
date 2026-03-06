# -*- coding: utf-8 -*-
"""
C3 credit assignment (counterfactual) utilities.

The C3 pipeline in OpenRLHF will:
- Use a centralized critic (V or Q) to estimate counterfactual baselines
- Convert team reward into per-role scalar advantages
- Broadcast scalar advantages to token-level advantages for PPO updates
"""
