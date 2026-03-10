from __future__ import annotations

from types import MappingProxyType

PAPER_TRAIN_BUDGET_B = 8

PAPER_TRAIN_METHOD_N_SAMPLES = MappingProxyType(
    {
        "MAPPO": PAPER_TRAIN_BUDGET_B,
        "MAGRPO": PAPER_TRAIN_BUDGET_B,
        "C3": PAPER_TRAIN_BUDGET_B,
    }
)


def get_paper_train_n_samples(method: str) -> int:
    key = str(method or "").strip().upper()
    try:
        return int(PAPER_TRAIN_METHOD_N_SAMPLES[key])
    except KeyError as exc:
        supported = ", ".join(PAPER_TRAIN_METHOD_N_SAMPLES.keys())
        raise KeyError(f"Unsupported paper training method {method!r}. Expected one of: {supported}") from exc
