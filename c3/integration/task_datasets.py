# -*- coding: utf-8 -*-
"""
C3 task datasets loader.

Build HF `datasets.Dataset` objects from a TaskSpec:
  - train_datasets -> merged training dataset
  - eval_suites    -> dict[name, dataset]

Features:
  - local json/jsonl(.gz) + HF hub datasets
  - per-entry `limit` (train/eval)
  - global `max_train_samples` / `max_eval_samples` (applied AFTER mixing)
  - `datasource` column for logging
  - normalize prompt/answer columns into `input` / `answer`
  - sampling_mode: concat | interleave | merge_epoch
    - merge_epoch returns an epoch-aware builder for trainer-side epoch rebuild

Note:
  - merge_epoch requires finite-length (map-style) datasets (needs stable epoch size).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# PromptDataset expects `input` and `answer`.
_PROMPT_KEY_CANDIDATES: Tuple[str, ...] = ("input", "question", "problem", "prompt", "instruction", "query", "text")
_ANSWER_KEY_CANDIDATES: Tuple[str, ...] = ("answer", "golden", "label", "target")

_JSON_SUFFIXES: Tuple[str, ...] = (".json", ".jsonl", ".json.gz", ".jsonl.gz")


# -----------------------------------------------------------------------------
# Public container
# -----------------------------------------------------------------------------


@dataclass
class TaskDatasets:
    train: Any  # datasets.Dataset (or datasets.IterableDataset in non-merge modes)
    evals: Dict[str, Any]  # name -> datasets.Dataset / IterableDataset
    train_builder: Optional["EpochTrainDatasetBuilder"] = None  # only for sampling_mode=merge_epoch


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _hf_datasets():
    try:
        import datasets  # type: ignore
    except Exception as e:
        raise ImportError("Please install 'datasets' to use C3 task datasets loader.") from e
    return datasets


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _to_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        v = int(x)
    except Exception:
        return None
    return v if v > 0 else None


def _is_local_file(path: str) -> bool:
    if not isinstance(path, str):
        return False
    p = path.strip()
    if not p:
        return False
    if os.path.exists(p):
        return True
    pl = p.lower()
    return any(pl.endswith(suf) for suf in _JSON_SUFFIXES)


def _repo_root_from_task_spec(task_spec: Any) -> Optional[Path]:
    raw = getattr(task_spec, "repo_root", None)
    if isinstance(raw, str) and raw.strip():
        p = Path(raw).expanduser().resolve()
        if p.exists():
            return p

    task_path = getattr(task_spec, "task_path", None)
    if isinstance(task_path, str) and task_path.strip():
        tp = Path(task_path).expanduser().resolve()
        for p in [tp.parent] + list(tp.parents):
            if (p / "configs" / "tasks").is_dir() and (p / "c3").exists():
                return p
    return None


def _task_dir_from_task_spec(task_spec: Any) -> Optional[Path]:
    task_path = getattr(task_spec, "task_path", None)
    if isinstance(task_path, str) and task_path.strip():
        return Path(task_path).expanduser().resolve().parent
    return None


def _resolve_local_path(path: str, *, repo_root: Optional[Path], task_dir: Optional[Path]) -> str:
    p = str(path or "").strip()
    if not p:
        return p

    cand = Path(p).expanduser()
    if cand.exists():
        return str(cand.resolve())

    if cand.is_absolute():
        return str(cand)

    if repo_root is not None:
        repo_cand = (repo_root / cand).resolve()
        if repo_cand.exists():
            return str(repo_cand)

    if task_dir is not None:
        task_cand = (task_dir / cand).resolve()
        if task_cand.exists():
            return str(task_cand)

    if repo_root is not None:
        return str((repo_root / cand).resolve())
    if task_dir is not None:
        return str((task_dir / cand).resolve())
    return str(cand)


def _resolve_local_data_files(data_files: Any, *, repo_root: Optional[Path], task_dir: Optional[Path]) -> Any:
    if isinstance(data_files, str):
        return _resolve_local_path(data_files, repo_root=repo_root, task_dir=task_dir)
    if isinstance(data_files, (list, tuple)):
        return [_resolve_local_data_files(x, repo_root=repo_root, task_dir=task_dir) for x in data_files]
    if isinstance(data_files, dict):
        return {k: _resolve_local_data_files(v, repo_root=repo_root, task_dir=task_dir) for k, v in data_files.items()}
    return data_files


def _infer_json_builder(fmt: Optional[str], path_or_files: Any) -> Optional[str]:
    # HF "json" builder supports jsonl as well.
    if fmt:
        f = str(fmt).strip().lower()
        if f in {"json", "jsonl"}:
            return "json"

    def _looks_json(s: str) -> bool:
        sl = s.lower()
        return any(sl.endswith(suf) for suf in _JSON_SUFFIXES)

    if isinstance(path_or_files, str) and _looks_json(path_or_files):
        return "json"

    if isinstance(path_or_files, (list, tuple)) and path_or_files:
        if _looks_json(str(path_or_files[0])):
            return "json"

    if isinstance(path_or_files, dict) and path_or_files:
        any_path = next(iter(path_or_files.values()))
        if _looks_json(str(any_path)):
            return "json"

    return None


def _normalize_sampling_mode(mode: str) -> str:
    m = str(mode or "concat").strip().lower()
    if not m:
        return "concat"
    if m in {"concat", "concatenate"}:
        return "concat"
    if m in {"interleave", "weighted", "mix"}:
        return "interleave"
    if m in {"merge_epoch", "merge", "epoch_merge"}:
        return "merge_epoch"
    raise ValueError(f"Unsupported sampling_mode={m!r}. Expected: concat | interleave | merge_epoch")


def _normalize_ds_entry(entry: Union[str, Mapping[str, Any]], default_split: str) -> Dict[str, Any]:
    if isinstance(entry, str):
        return {
            "path": entry,
            "split": default_split,
            "name": os.path.basename(entry).replace(".", "_") or "dataset",
            "weight": 1.0,
            "streaming": False,
            "limit": None,
        }
    if not isinstance(entry, Mapping):
        raise TypeError(f"Dataset entry must be str or mapping, got {type(entry)}")

    d = dict(entry)

    # unify keys
    if "dataset" in d and "path" not in d:
        d["path"] = d.pop("dataset")
    if "repo_id" in d and "path" not in d:
        d["path"] = d.pop("repo_id")
    if "subset" in d and "config" not in d:
        d["config"] = d.pop("subset")

    d["split"] = d.get("split") or default_split

    if not d.get("name"):
        p = str(d.get("path") or d.get("data_files") or "dataset")
        d["name"] = os.path.basename(p).replace(".", "_") or "dataset"

    try:
        d["weight"] = float(d.get("weight", 1.0) or 1.0)
    except Exception:
        d["weight"] = 1.0

    d["streaming"] = bool(d.get("streaming", False))
    d["limit"] = _to_int_or_none(d.get("limit", None))

    return d


def _wrap_data_files_for_requested_split(data_files: Any, split: str) -> Any:
    """
    Local json/jsonl with HF json builder defaults to 'train'.
    If user requests split != 'train', map files to that split name.
    """
    if not split or split == "train" or data_files is None:
        return data_files

    if isinstance(data_files, str):
        return {split: data_files}

    if isinstance(data_files, (list, tuple)):
        return {split: list(data_files)}

    if isinstance(data_files, dict):
        if split in data_files:
            return data_files
        if len(data_files) == 1 and "train" in data_files:
            return {split: data_files["train"]}
        return data_files

    return data_files


# -----------------------------------------------------------------------------
# Dataset transforms (columns / caps)
# -----------------------------------------------------------------------------


def _add_datasource_column(ds: Any, datasource: str) -> Any:
    """
    Add/overwrite `datasource` column.
    Prefer batched map to avoid building huge python lists.
    """
    if hasattr(ds, "map"):

        def _fn(batch):
            n = len(next(iter(batch.values()))) if isinstance(batch, dict) and batch else 1
            return {"datasource": [datasource] * n}

        try:
            # HF Dataset / IterableDataset both support map; batched speeds up.
            return ds.map(_fn, batched=True)
        except Exception:
            pass

    # Fallback: Dataset.add_column (requires full list)
    if hasattr(ds, "add_column") and hasattr(ds, "__len__"):
        n = len(ds)
        if "datasource" in getattr(ds, "column_names", []):
            ds = ds.remove_columns(["datasource"])
        return ds.add_column("datasource", [datasource] * n)

    raise TypeError(f"Unsupported dataset type for adding datasource: {type(ds)}")


def _detect_first_existing_column(ds: Any, candidates: Tuple[str, ...]) -> Optional[str]:
    cols = list(getattr(ds, "column_names", []) or [])
    for k in candidates:
        if k in cols:
            return k
    return None


def _ensure_column(ds: Any, datasource: str, src_key: str, dst_key: str) -> Any:
    if src_key == dst_key:
        return ds

    cols = list(getattr(ds, "column_names", []) or [])

    if hasattr(ds, "rename_column") and (dst_key not in cols) and (src_key in cols):
        try:
            out = ds.rename_column(src_key, dst_key)
            logger.info("[C3] normalize column %s: %s -> %s", datasource, src_key, dst_key)
            return out
        except Exception:
            pass

    if hasattr(ds, "map"):

        def _fn(ex):
            ex = dict(ex)
            if dst_key not in ex and src_key in ex:
                ex[dst_key] = ex.get(src_key)
            return ex

        try:
            out = ds.map(_fn)
            logger.info("[C3] add column %s: %s <- %s", datasource, dst_key, src_key)
            return out
        except Exception:
            return ds

    return ds


def _ensure_input_column(ds: Any, datasource: str) -> Any:
    key = _detect_first_existing_column(ds, _PROMPT_KEY_CANDIDATES)
    return ds if key is None else _ensure_column(ds, datasource, key, "input")


def _ensure_answer_column(ds: Any, datasource: str) -> Any:
    key = _detect_first_existing_column(ds, _ANSWER_KEY_CANDIDATES)
    return ds if key is None else _ensure_column(ds, datasource, key, "answer")


def _apply_limit(ds: Any, limit: Optional[int], *, name: str) -> Any:
    """Per-entry cap. For iterable datasets we use `.take()` if available."""
    n = _to_int_or_none(limit)
    if n is None:
        return ds

    if hasattr(ds, "select") and hasattr(ds, "__len__"):
        try:
            L = len(ds)
            return ds.select(range(min(int(n), int(L))))
        except Exception:
            return ds

    if hasattr(ds, "take"):  # IterableDataset
        try:
            return ds.take(int(n))
        except Exception:
            logger.warning("[C3] limit ignored for %s (take failed).", name)
            return ds

    logger.warning("[C3] limit ignored for %s (no select/take).", name)
    return ds


def _apply_global_cap(ds: Any, cap: Optional[int]) -> Any:
    """Global cap AFTER mixing (map-style only)."""
    n = _to_int_or_none(cap)
    if n is None:
        return ds
    if hasattr(ds, "select") and hasattr(ds, "__len__"):
        try:
            L = len(ds)
            return ds.select(range(min(int(n), int(L))))
        except Exception:
            return ds
    return ds


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def _load_one_dataset(
    entry: Dict[str, Any],
    cache_dir: Optional[str],
    *,
    repo_root: Optional[Path] = None,
    task_dir: Optional[Path] = None,
) -> Any:
    datasets = _hf_datasets()

    path = entry.get("path", None)
    if isinstance(path, str):
        path = _resolve_local_path(path, repo_root=repo_root, task_dir=task_dir)
    split = str(entry.get("split", "train") or "train")
    data_files = _resolve_local_data_files(entry.get("data_files", None), repo_root=repo_root, task_dir=task_dir)
    data_files = _wrap_data_files_for_requested_split(data_files, split)

    config = entry.get("config", None)
    revision = entry.get("revision", None)
    streaming = bool(entry.get("streaming", False))
    fmt = entry.get("format", None)

    # Local file path => json builder
    if isinstance(path, str) and _is_local_file(path):
        builder = _infer_json_builder(fmt, path) or "json"
        local_files = _wrap_data_files_for_requested_split(path, split)
        return datasets.load_dataset(
            builder,
            data_files=local_files,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )

    # Local json/jsonl via data_files
    if data_files is not None:
        builder = _infer_json_builder(fmt, data_files)
        if builder == "json":
            return datasets.load_dataset(
                "json",
                data_files=data_files,
                split=split,
                cache_dir=cache_dir,
                streaming=streaming,
            )

    # HF hub dataset repo
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"Invalid dataset path in entry: {entry}")

    kwargs: Dict[str, Any] = dict(
        path=path,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    if config:
        kwargs["name"] = config
    if revision:
        kwargs["revision"] = revision
    if data_files is not None:
        kwargs["data_files"] = data_files

    return datasets.load_dataset(**kwargs)


# -----------------------------------------------------------------------------
# Mixing modes
# -----------------------------------------------------------------------------


def _interleave_train_datasets(
    datasets_list: Sequence[Any],
    weights: Sequence[float],
    seed: int,
    sampling_mode: str,
) -> Any:
    datasets = _hf_datasets()

    mode = _normalize_sampling_mode(sampling_mode)
    if len(datasets_list) == 1:
        return datasets_list[0]

    if mode == "concat":
        return datasets.concatenate_datasets(list(datasets_list))

    # interleave
    w = [float(x) for x in weights]
    s = float(sum(w))
    probs = [x / s for x in w] if s > 0 else None
    return datasets.interleave_datasets(
        list(datasets_list),
        probabilities=probs,
        seed=int(seed),
        stopping_strategy="all_exhausted",
    )


@dataclass
class EpochTrainDatasetBuilder:
    """
    Epoch-level mixed dataset builder for sampling_mode=merge_epoch.

    Intended semantics:
      - Each epoch rebuilds a *new order* of the merged dataset (deterministic by seed+epoch).
      - Epoch size should NOT be bottlenecked by the smallest dataset.
      - By default, consume all datasets once per epoch (like interleave(all_exhausted)),
        but with per-epoch randomness.

    Note:
      - Requires finite-length (map-style) datasets for stable epoch sizing.
    """

    datasets: Sequence[Any]
    weights: Sequence[float]
    base_seed: int = 42
    reshuffle_each_epoch: bool = True
    max_train_samples: Optional[int] = None

    def _epoch_seed(self, epoch: int) -> int:
        # If reshuffle_each_epoch is False, keep epoch seed stable.
        return int(self.base_seed) + (int(epoch) if self.reshuffle_each_epoch else 0)

    def build(self, epoch: int = 0) -> Any:
        datasets = _hf_datasets()
        if not self.datasets:
            raise ValueError("merge_epoch requires at least one train dataset.")

        ds_list = list(self.datasets)

        # Enforce finite-length datasets for stable epoch sizing.
        for i, ds in enumerate(ds_list):
            try:
                _ = len(ds)
            except Exception as e:
                raise TypeError(
                    "merge_epoch requires finite-length (map-style) datasets. "
                    f"dataset_idx={i}, type={type(ds)}"
                ) from e

        seed = self._epoch_seed(epoch)

        if len(ds_list) == 1:
            out = ds_list[0]
        else:
            ws = [float(w) for w in self.weights]
            if any((not (w > 0.0)) for w in ws):
                raise ValueError(f"merge_epoch requires all weights > 0, got: {ws}")
            s = float(sum(ws))
            probs = [w / s for w in ws] if s > 0 else None

            # Key: all_exhausted => consume each dataset once per epoch, total size ~ sum(len_i)
            out = datasets.interleave_datasets(
                ds_list,
                probabilities=probs,
                seed=int(seed),
                stopping_strategy="all_exhausted",
            )

        # Optional: global shuffle for extra mixing (still deterministic per epoch)
        if self.reshuffle_each_epoch and hasattr(out, "shuffle"):
            try:
                out = out.shuffle(seed=int(seed))
            except Exception:
                pass

        return _apply_global_cap(out, self.max_train_samples)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def load_task_datasets(
    task_spec: Any,
    *,
    cache_dir: Optional[str] = None,
    default_train_split: str = "train",
    default_eval_split: str = "test",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> TaskDatasets:
    env = getattr(task_spec, "environment", None)
    env_cfg = env if isinstance(env, dict) else {}
    repo_root = _repo_root_from_task_spec(task_spec)
    task_dir = _task_dir_from_task_spec(task_spec)

    seed = int(env_cfg.get("mix_seed", 42))
    sampling_mode = _normalize_sampling_mode(str(env_cfg.get("sampling_mode", "concat") or "concat"))
    reshuffle_each_epoch = bool(env_cfg.get("reshuffle_each_epoch", False))

    # ---- train ----
    train_entries_raw = getattr(task_spec, "train_datasets", None)
    if train_entries_raw is None and isinstance(env_cfg, dict):
        train_entries_raw = env_cfg.get("train_datasets", None)
    train_entries = [_normalize_ds_entry(x, default_train_split) for x in _as_list(train_entries_raw)]
    if not train_entries:
        raise ValueError("TaskSpec.train_datasets is empty; cannot build training dataset.")

    train_dsets: List[Any] = []
    train_weights: List[float] = []

    for ent in train_entries:
        name = str(ent.get("name", "train"))
        ds = _load_one_dataset(ent, cache_dir=cache_dir, repo_root=repo_root, task_dir=task_dir)

        ds = _apply_limit(ds, ent.get("limit", None), name=name)
        ds = _add_datasource_column(ds, name)
        ds = _ensure_input_column(ds, name)
        ds = _ensure_answer_column(ds, name)

        train_dsets.append(ds)
        train_weights.append(float(ent.get("weight", 1.0)))

    train_builder: Optional[EpochTrainDatasetBuilder] = None
    if sampling_mode == "merge_epoch":
        train_builder = EpochTrainDatasetBuilder(
            datasets=tuple(train_dsets),
            weights=tuple(train_weights),
            base_seed=int(seed),
            reshuffle_each_epoch=bool(reshuffle_each_epoch),
            max_train_samples=_to_int_or_none(max_train_samples),
        )
        train = train_builder.build(epoch=0)
    else:
        train = _interleave_train_datasets(train_dsets, train_weights, seed=seed, sampling_mode=sampling_mode)
        train = _apply_global_cap(train, max_train_samples)

    # ---- eval ----
    evals: Dict[str, Any] = {}
    eval_raw = getattr(task_spec, "eval_suites", None)
    if eval_raw is None and isinstance(env_cfg, dict):
        eval_raw = env_cfg.get("eval_suites", None)

    if isinstance(eval_raw, Mapping):
        suites = list(eval_raw.items())
    else:
        entries = _as_list(eval_raw)
        suites = []
        for i, e in enumerate(entries):
            if isinstance(e, Mapping) and e.get("name"):
                suites.append((str(e["name"]), e))
            else:
                suites.append((f"eval_{i}", e))

    for suite_name, suite_entry in suites:
        ent = _normalize_ds_entry(suite_entry, default_eval_split)
        name = str(suite_name)

        ds = _load_one_dataset(ent, cache_dir=cache_dir, repo_root=repo_root, task_dir=task_dir)

        ds = _apply_limit(ds, ent.get("limit", None), name=name)
        ds = _add_datasource_column(ds, name)
        ds = _ensure_input_column(ds, name)
        ds = _ensure_answer_column(ds, name)
        ds = _apply_global_cap(ds, max_eval_samples)

        evals[name] = ds

    return TaskDatasets(train=train, evals=evals, train_builder=train_builder)
