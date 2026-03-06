from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RewardRequest:
    # For remote reward model:
    query_text: str
    prompt_text: str
    label_text: Optional[str]

    # For env reward / debugging:
    meta: Dict[str, Any]


@dataclass
class RewardResult:
    reward: float
    score: Optional[float] = None
    source: str = "unknown"          # e.g. "env" / "remote_rm"
    info: Optional[Dict[str, Any]] = None
