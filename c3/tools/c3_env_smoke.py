# c3/tools/c3_env_smoke.py
"""
C3 env smoke test.

Goal:
- Make sure task loading, 2-agent prompt rendering (Reasoner -> Actor), and evaluator wiring
  are all functioning end-to-end without crashing.

Usage:
  PYTHONPATH=. python -m c3.tools.c3_env_smoke --task path/to/task.yaml --limit 1
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import random
import sys
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


# ----------------------------- small utilities -----------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np  # optional

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # optional

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _call_compat(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call fn with only the kwargs it accepts (robust to signature drift)."""
    try:
        sig = inspect.signature(fn)
    except Exception:
        return fn(*args, **kwargs)

    accepted: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return fn(*args, **accepted)


def _import_first(candidates: Sequence[Tuple[str, Optional[str]]]) -> Any:
    """
    Import the first available candidate.

    candidates: [(module_path, attr_name_or_None), ...]
      - If attr is None: return imported module
      - Else: return getattr(module, attr)
    """
    last_err: Optional[BaseException] = None
    for mod_path, attr in candidates:
        try:
            mod = __import__(mod_path, fromlist=["*"])
            return mod if attr is None else getattr(mod, attr)
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        "Failed to import any candidate:\n"
        + "\n".join([f"- {m}{'' if a is None else ':' + a}" for m, a in candidates])
        + (f"\n\nLast error: {last_err!r}" if last_err else "")
    )


def _to_jsonable(x: Any) -> Any:
    """Best-effort conversion for printing evaluator results."""
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    # common numpy/torch scalars
    for attr in ("item",):
        try:
            if hasattr(x, attr) and callable(getattr(x, attr)):
                return _to_jsonable(x.item())
        except Exception:
            pass
    return repr(x)


# ----------------------------- task + data -----------------------------


def _load_task_yaml(task_path: str) -> Dict[str, Any]:
    with open(task_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Task YAML must be a mapping/dict, got: {type(data)}")
    return data


def _load_task_obj(task_path: str) -> Any:
    """
    Prefer project's canonical load_task(); fall back to raw YAML dict.

    NOTE:
      In this repo, canonical loaders include:
        - c3.integration.marl_specs: load_task(task_path: str)
    """
    task_spec = _load_task_yaml(task_path)

    load_task_candidates = [
        # ✅ repo-actual loaders (prefer these)
        ("c3.integration.marl_specs", "load_task"),
        # (optional) other forks/branches
        ("c3.task", "load_task"),
        ("c3.tasks", "load_task"),
        ("c3.envs.task", "load_task"),
        ("c3.envs.tasks", "load_task"),
        ("openrlhf.tasks", "load_task"),
        ("openrlhf.envs.tasks", "load_task"),
    ]

    try:
        load_task = _import_first(load_task_candidates)

        # Most robust: pass path positionally (handles load_task(task_path: str))
        try:
            return load_task(task_path)
        except TypeError:
            # Fallback: compatible keyword names if a branch uses different param name
            return _call_compat(load_task, task_path=task_path, path=task_path, task_yaml=task_path)

    except Exception:
        # IMPORTANT: don't silently swallow this; it hides real integration issues.
        print("[smoke] WARN: load_task() failed; falling back to raw YAML dict.", file=sys.stderr)
        traceback.print_exc()
        return task_spec


def _take_first_n_dicts(ds: Any, n: int) -> List[Dict[str, Any]]:
    """
    Best-effort: take first n examples from a HF Dataset / IterableDataset / iterable.
    """
    out: List[Dict[str, Any]] = []
    if n <= 0:
        return out

    # Map-style HF Dataset: supports __len__ and __getitem__
    try:
        L = len(ds)  # type: ignore[arg-type]
        for i in range(min(int(n), int(L))):
            ex = ds[i]  # type: ignore[index]
            if isinstance(ex, dict):
                out.append(ex)
        if out:
            return out
    except Exception:
        pass

    # IterableDataset or generic iterable
    try:
        for ex in ds:  # type: ignore[assignment]
            if isinstance(ex, dict):
                out.append(ex)
            if len(out) >= n:
                break
        return out
    except Exception:
        return out


def _iter_instances(task_obj: Any, *, task_yaml_path: str, limit: int, seed: int) -> List[Dict[str, Any]]:
    """
    Try common patterns to fetch examples.
    - Prefer C3 canonical dataset loader (load_task_datasets) when task_obj is a TaskSpec.
    - Otherwise fall back to legacy patterns.
    """
    # Pattern 0 (preferred in this repo): build datasets from TaskSpec via load_task_datasets()
    if not isinstance(task_obj, dict):
        try:
            load_task_datasets = _import_first([("c3.integration.task_datasets", "load_task_datasets")])
            task_spec_for_ds = task_obj

            # c3.integration.marl_specs.TaskSpec keeps dataset specs in
            # task_spec.environment.{train_datasets,eval_suites}, while
            # load_task_datasets expects top-level attributes. Bridge that shape.
            env = getattr(task_obj, "environment", None)
            if isinstance(env, dict) and (
                not hasattr(task_obj, "train_datasets") or not hasattr(task_obj, "eval_suites")
            ):
                class _TaskSpecCompat:
                    pass

                compat = _TaskSpecCompat()
                compat.environment = env
                compat.train_datasets = env.get("train_datasets", [])
                compat.eval_suites = env.get("eval_suites", [])
                task_spec_for_ds = compat

            td = _call_compat(load_task_datasets, task_spec=task_spec_for_ds)

            # td is TaskDatasets(train=..., evals={...})
            evals = getattr(td, "evals", None)
            train = getattr(td, "train", None)

            # Prefer eval suite if available; else use train.
            if isinstance(evals, dict) and evals:
                for key in ("test", "eval", "validation", "valid", "dev"):
                    if key in evals:
                        xs = _take_first_n_dicts(evals[key], limit)
                        if xs:
                            return xs
                first_suite = next(iter(evals.values()))
                xs = _take_first_n_dicts(first_suite, limit)
                if xs:
                    return xs

            if train is not None:
                xs = _take_first_n_dicts(train, limit)
                if xs:
                    return xs

        except Exception:
            pass

    # Pattern A: task_obj.get_instances(limit=..., seed=...)
    if hasattr(task_obj, "get_instances") and callable(getattr(task_obj, "get_instances")):
        xs = _call_compat(task_obj.get_instances, limit=limit, seed=seed)
        if isinstance(xs, list) and all(isinstance(i, dict) for i in xs):
            return xs

    # Pattern B: task_obj.dataset (list[dict]) or {"train": [...], "test": [...]}
    ds = getattr(task_obj, "dataset", None)
    if isinstance(ds, list) and all(isinstance(i, dict) for i in ds):
        return ds[:limit]
    if isinstance(ds, dict):
        for split in ("train", "validation", "valid", "dev", "test"):
            if split in ds and isinstance(ds[split], list) and ds[split]:
                xs = ds[split]
                if all(isinstance(i, dict) for i in xs):
                    return xs[:limit]

    # Pattern C: YAML spec has inline data
    if isinstance(task_obj, dict):
        for key in ("instances", "data", "examples", "dataset"):
            val = task_obj.get(key)
            if isinstance(val, list) and val and all(isinstance(i, dict) for i in val):
                return val[:limit]
        if isinstance(task_obj.get("splits"), dict):
            splits = task_obj["splits"]
            for split in ("train", "validation", "valid", "dev", "test"):
                val = splits.get(split)
                if isinstance(val, list) and val and all(isinstance(i, dict) for i in val):
                    return val[:limit]

    raise RuntimeError(
        "Could not obtain instances from task.\n"
        "Tried:\n"
        "  - C3 load_task_datasets(task_spec) -> TaskDatasets(evals/train)\n"
        "  - task.get_instances(limit=..., seed=...)\n"
        "  - task.dataset as list[dict]\n"
        "  - task.dataset as dict of splits\n"
        "  - task YAML includes inline 'instances'/'data'/'examples'\n"
        f"\nHint: check your task yaml ({task_yaml_path}) has train_datasets/eval_suites and that 'datasets' is installed."
    )


def _get_question_text(ex: Dict[str, Any]) -> str:
    for k in ("question", "prompt", "input", "problem", "query", "text"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return json.dumps(_to_jsonable(ex), ensure_ascii=False)


def _get_answer_text(ex: Dict[str, Any]) -> Optional[str]:
    for k in ("answer", "gold", "target", "label", "solution", "output"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v
    for k in ("answers", "solutions", "targets"):
        v = ex.get(k)
        if isinstance(v, list) and v and isinstance(v[0], str) and v[0].strip():
            return v[0]
    return None


def _get_task_hint(task_obj: Any, task_spec: Dict[str, Any]) -> str:
    if isinstance(task_spec, dict):
        hint = str(task_spec.get("name") or task_spec.get("task") or task_spec.get("id") or "")
        if hint.strip():
            return hint

    for attr in ("name", "task", "id"):
        try:
            v = getattr(task_obj, attr, None)
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass
    return ""


# ----------------------------- prompt rendering -----------------------------


def _resolve_prompt_render_module() -> Any:
    candidates = [
        ("c3.prompt_render", None),
        ("c3.prompts.prompt_render", None),
        ("c3.mas.prompt_render", None),
        ("c3.mas.prompt_renderer", None),
    ]
    return _import_first(candidates)


def _render_prompt_for_role(
    prompt_render_mod: Any,
    *,
    ex: Dict[str, Any],
    task_obj: Any,
    role: str,
    roles_topo: Sequence[str] = ("Reasoner", "Actor"),
    role_outputs: Optional[Mapping[str, str]] = None,
    topo_so_far: Optional[Sequence[str]] = None,
) -> str:
    """
    Render a single role prompt.

    Compatible with prompt_render modules that define:
      - build_render_context(..., question=..., role_outputs=..., topo_so_far=..., ...)
      - render_role_prompt(ctx, role=...)
    """
    build_ctx = getattr(prompt_render_mod, "build_render_context", None)
    render_role = getattr(prompt_render_mod, "render_role_prompt", None)

    q = _get_question_text(ex)
    ro = dict(role_outputs or {})
    topo = list(topo_so_far or [])

    if callable(build_ctx) and callable(render_role):
        try:
            ctx = _call_compat(
                build_ctx,
                # required by some variants (including yours)
                question=q,
                role_outputs=ro,
                topo_so_far=topo,
                # common optional knobs across variants
                role=role,
                roles=list(roles_topo),
                roles_topo=list(roles_topo),
                topo=list(roles_topo),
                example=ex,
                ex=ex,
                task=task_obj,
                task_obj=task_obj,
            )
            prompt = _call_compat(render_role, ctx, role=role, question=q, example=ex, ex=ex)
            return prompt if isinstance(prompt, str) else str(prompt)
        except Exception:
            # If prompt renderer itself threw (signature mismatch or internal error), fall back.
            pass

    # Fallback: minimal deterministic prompt.
    if role.lower() == "reasoner":
        return f"You are the Reasoner. Produce a short plan.\n\nQuestion:\n{q}\n"
    return f"You are the Actor. Solve the task.\n\nQuestion:\n{q}\n"


# ----------------------------- evaluator wiring -----------------------------


class _AttrDict(dict):
    """Allow dict keys to be accessed as attributes (best-effort)."""

    __getattr__ = dict.get  # type: ignore[assignment]


def _wrap_task_spec_for_replay(task_spec: Any) -> Any:
    """
    replay._OpenRLHFEvaluator expects getattr(task_spec, "environment", None),
    but YAML loader returns a dict. Wrap it so .environment works.
    """
    if isinstance(task_spec, dict):
        return _AttrDict(task_spec)
    return task_spec


def _resolve_reward_fn(task_obj: Any) -> Optional[Callable[..., Any]]:
    for name in ("reward_fn", "reward", "scorer", "score_fn", "metric_fn"):
        fn = getattr(task_obj, name, None)
        if callable(fn):
            return fn
    return None


def _fallback_reward_fn(*, prediction: str, label: Any, meta: Any = None, **_: Any) -> float:
    if label is None:
        return 0.0
    return 1.0 if str(prediction).strip() == str(label).strip() else 0.0


def _build_evaluator(task_obj: Any, task_spec_for_eval: Any) -> Any:
    """
    Try common evaluator creation patterns.

    This repo's C3 replay evaluator (_OpenRLHFEvaluator) uses:
      - __init__(task_spec, reward_fn, roles_topo)
      - evaluate(restart, role_outputs, meta) -> float
    """
    if hasattr(task_obj, "build_evaluator") and callable(getattr(task_obj, "build_evaluator")):
        return _call_compat(task_obj.build_evaluator, task=task_obj, task_spec=task_spec_for_eval, cfg=task_spec_for_eval)

    ev = getattr(task_obj, "evaluator", None)
    if ev is not None:
        return ev

    candidates = [
        ("c3.envs.evaluator", "build_evaluator"),
        ("c3.evaluator", "build_evaluator"),
        ("openrlhf.envs.evaluator", "build_evaluator"),
        ("openrlhf.evaluator", "build_evaluator"),
    ]
    try:
        build_evaluator = _import_first(candidates)
        return _call_compat(build_evaluator, task=task_obj, task_spec=task_spec_for_eval, cfg=task_spec_for_eval)
    except Exception:
        pass

    try:
        OpenEval = _import_first([("c3.analysis.replay", "_OpenRLHFEvaluator")])
        reward_fn = _resolve_reward_fn(task_obj) or _fallback_reward_fn
        roles_topo = ("Reasoner", "Actor")
        return OpenEval(task_spec=_wrap_task_spec_for_replay(task_spec_for_eval), reward_fn=reward_fn, roles_topo=roles_topo)
    except Exception as e:
        raise RuntimeError(f"Failed to build evaluator via replay._OpenRLHFEvaluator fallback: {e!r}") from e


def _detect_eval_interface(evaluator: Any) -> str:
    if hasattr(evaluator, "evaluate") and callable(getattr(evaluator, "evaluate")):
        try:
            sig = inspect.signature(evaluator.evaluate)
            params = sig.parameters
            if "restart" in params and "role_outputs" in params:
                return "replay"
            if "prediction" in params or "pred" in params or "example" in params or "ex" in params:
                return "prediction"
        except Exception:
            return "prediction"
        return "prediction"

    if callable(evaluator):
        return "callable"

    return "unknown"


class _RestartShim:
    """Minimal shim matching the replay RestartState interface used by _OpenRLHFEvaluator."""

    def __init__(self, meta: Mapping[str, Any]) -> None:
        self.meta = dict(meta)


def _eval_once(
    evaluator: Any,
    *,
    ex: Dict[str, Any],
    prediction: str,
    role_outputs: Optional[Mapping[str, str]] = None,
) -> Any:
    style = _detect_eval_interface(evaluator)

    if style == "prediction":
        return _call_compat(getattr(evaluator, "evaluate"), example=ex, ex=ex, prediction=prediction, pred=prediction)

    if style == "replay":
        label = _get_answer_text(ex)
        meta: Dict[str, Any] = {"label": label}
        restart = _RestartShim(meta)

        ro: Dict[str, str] = dict(role_outputs or {})
        answer_role = getattr(evaluator, "answer_role", None)
        if isinstance(answer_role, str) and answer_role:
            ro.setdefault(answer_role, prediction)
        ro.setdefault("Actor", prediction)

        return getattr(evaluator, "evaluate")(restart=restart, role_outputs=ro, meta=meta)

    if style == "callable":
        return _call_compat(evaluator, example=ex, ex=ex, prediction=prediction, pred=prediction)

    raise TypeError(f"Evaluator is neither callable nor has .evaluate(): {type(evaluator)}")


# ----------------------------- fake model -----------------------------


def _fake_reasoner_output(ex: Dict[str, Any]) -> str:
    return "Plan: restate the question, identify constraints, produce the final answer."


def _fake_actor_output(task_hint: str, ex: Dict[str, Any]) -> str:
    hint = (task_hint or "").lower()
    ans = _get_answer_text(ex)

    if any(k in hint for k in ("math", "gsm", "arithmetic", "algebra")) and ans is not None:
        return ans
    if any(k in hint for k in ("code", "mbpp", "humaneval", "evalplus")):
        return ""
    return ans or ""


# ----------------------------- main -----------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="c3_env_smoke", description="C3 env smoke test (task+prompts+evaluator).")
    p.add_argument("--task", type=str, required=True, help="Path to task YAML.")
    p.add_argument("--limit", type=int, default=2, help="Number of examples to test.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--print_example", type=int, default=1, help="Whether to print prompts/pred/result (0/1).")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    _seed_everything(int(args.seed))

    task_spec_yaml = _load_task_yaml(args.task)
    task_obj = _load_task_obj(args.task)

    try:
        prompt_render_mod = _resolve_prompt_render_module()
    except Exception as e:
        print("[smoke] ERROR: failed to import prompt_render module.", file=sys.stderr)
        print(repr(e), file=sys.stderr)
        return 2

    task_spec_for_eval = task_obj if not isinstance(task_obj, dict) else task_spec_yaml
    try:
        evaluator = _build_evaluator(task_obj, task_spec_for_eval)
    except Exception as e:
        print("[smoke] ERROR: failed to build evaluator.", file=sys.stderr)
        print(repr(e), file=sys.stderr)
        return 3

    try:
        instances = _iter_instances(task_obj, task_yaml_path=args.task, limit=int(args.limit), seed=int(args.seed))
    except Exception as e:
        print("[smoke] ERROR: failed to load instances from task.", file=sys.stderr)
        print(repr(e), file=sys.stderr)
        return 4

    task_hint = _get_task_hint(task_obj, task_spec_yaml)

    ok = 0
    for idx, ex in enumerate(instances):
        q = _get_question_text(ex)

        try:
            roles_topo = ("Reasoner", "Actor")

            reasoner_prompt = _render_prompt_for_role(
                prompt_render_mod,
                ex=ex,
                task_obj=task_obj,
                role="Reasoner",
                roles_topo=roles_topo,
                role_outputs={},
                topo_so_far=[],
            )
            reasoner_out = _fake_reasoner_output(ex)

            actor_prompt = _render_prompt_for_role(
                prompt_render_mod,
                ex=ex,
                task_obj=task_obj,
                role="Actor",
                roles_topo=roles_topo,
                role_outputs={"Reasoner": reasoner_out},
                topo_so_far=["Reasoner"],
            )
            actor_out = _fake_actor_output(task_hint, ex)

            result = _eval_once(
                evaluator,
                ex=ex,
                prediction=actor_out,
                role_outputs={"Reasoner": reasoner_out, "Actor": actor_out},
            )
            ok += 1

            if int(args.print_example):
                print("\n" + "=" * 88)
                print(f"[smoke] example {idx+1}/{len(instances)}")
                print("-" * 88)
                print("[question]")
                print(q)
                print("-" * 88)
                print("[prompt: Reasoner]")
                print(reasoner_prompt)
                print("[fake Reasoner output]")
                print(reasoner_out)
                print("-" * 88)
                print("[prompt: Actor]")
                print(actor_prompt)
                print("[fake Actor output]")
                print(actor_out)
                print("-" * 88)
                print("[evaluator result]")
                print(json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2))
                print("=" * 88)

        except Exception:
            print("\n" + "=" * 88, file=sys.stderr)
            print(f"[smoke] ERROR on example {idx+1}/{len(instances)}", file=sys.stderr)
            print(f"[question]\n{q}", file=sys.stderr)
            print("-" * 88, file=sys.stderr)
            traceback.print_exc()
            print("=" * 88, file=sys.stderr)

    if ok == 0:
        print("[smoke] No successful examples. See errors above.", file=sys.stderr)
        return 5

    print(f"[smoke] Done. success={ok}/{len(instances)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
