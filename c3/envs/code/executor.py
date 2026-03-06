# -*- coding: utf-8 -*-
"""
Sandbox-ish code execution utilities for CodeEnv (MBPP / MBPP+).

Core rules:
- Never execute untrusted code in the main process.
- Execute candidate code + tests in a subprocess with wall-clock timeout.
- Apply best-effort resource limits and restricted builtins/imports.
- Prevent "log bombs" via stdout/stderr redirection.

Env knobs:
- C3_CODE_MP_CTX: spawn | fork | forkserver
  default: Linux -> forkserver, others -> spawn
- C3_CODE_STARTUP_GRACE_S: extra seconds added to join()
  default: spawn/forkserver -> max(60, timeout*2), fork -> 3
- C3_CODE_MAX_MEM_MB: int, 0 disables RLIMIT_AS (default: 4096)
- C3_CODE_CPU_LIMIT_S: int, 0 => auto (default: max(timeout+1, 16))

Safety knobs:
- C3_CODE_MAX_CODE_CHARS: max extracted candidate code chars (default: 20000)
- C3_CODE_MAX_ASSERT_CHARS: max total assert/test chars (default: 4000)
- C3_CODE_STDIO_MODE: devnull | truncate (default: devnull)
- C3_CODE_STDIO_MAX_BYTES: truncate cap for stdout/stderr (default: 65536)

Import whitelist:
- Default is stdlib-only (compute-oriented). Extend via C3_CODE_EXTRA_IMPORTS="a,b,c".
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------
# Regex helpers (fenced code extraction / setup import splitting)
# ---------------------------------------------------------------------

_RE_FENCE_PY = re.compile(r"```python\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_RE_FENCE_ANY = re.compile(r"```\s*(.*?)```", flags=re.DOTALL)
_RE_IMPORT_LINE = re.compile(r"^\s*(import\s+\w|from\s+\w)", flags=0)

_VALID_MP_CTX = {"spawn", "fork", "forkserver"}


# ---------------------------------------------------------------------
# Env knobs
# ---------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        v = int(str(os.environ.get(name, "")).strip() or default)
        return v
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    s = os.environ.get(name, default)
    return default if s is None else str(s).strip()


_CODE_MAX_CHARS = _env_int("C3_CODE_MAX_CODE_CHARS", 20000)
_ASSERT_MAX_CHARS = _env_int("C3_CODE_MAX_ASSERT_CHARS", 4000)

_STDIO_MODE = _env_str("C3_CODE_STDIO_MODE", "devnull").lower() or "devnull"
_STDIO_MAX_BYTES = _env_int("C3_CODE_STDIO_MAX_BYTES", 65536)


# ---------------------------------------------------------------------
# mp context / timeouts
# ---------------------------------------------------------------------

def _get_mp_ctx():
    if not hasattr(mp, "get_context"):
        return mp

    name = (os.environ.get("C3_CODE_MP_CTX") or "").strip().lower()
    if not name:
        name = "forkserver" if sys.platform.startswith("linux") else "spawn"
    if name not in _VALID_MP_CTX:
        name = "spawn"

    try:
        return mp.get_context(name)
    except Exception:
        return mp.get_context("spawn")


def _ctx_name(ctx: Any) -> str:
    try:
        return str(ctx.get_start_method())  # type: ignore[attr-defined]
    except Exception:
        return "spawn"


def _startup_grace_s(ctx_name: str, timeout_s: int) -> int:
    v = _env_int("C3_CODE_STARTUP_GRACE_S", 0)
    if v > 0:
        return v
    if ctx_name in {"spawn", "forkserver"}:
        return max(60, int(timeout_s) * 2)
    return 3


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_mbpp_tests(
    candidate_code: str,
    ref_code: Optional[str],
    sample_meta: Dict[str, Any],
    timeout: int = 15,
    *,
    mem_mb: Optional[int] = None,
    cpu_s: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    MBPP/MBPP+ evaluator.

    Meta keys:
      - MBPP:  test_setup_code, test_list, challenge_test_list
      - MBPP+: test_imports,   test_list, test (script fallback)

    Args:
      mem_mb/cpu_s: optional per-call overrides for RLIMIT_AS/RLIMIT_CPU.
                    When None, read from C3_CODE_MAX_MEM_MB / C3_CODE_CPU_LIMIT_S.
    """
    setup_code, tests, challenge, test_script = _assemble_tests(sample_meta)
    _ = ref_code  # kept for compatibility; not executed by default.

    base_total = int(len(tests) + len(challenge) + (1 if (test_script and not tests) else 0))
    if base_total == 0:
        return 0.0, _missing_tests_info(sample_meta)

    timeout_s = max(1, int(timeout))
    ctx = _get_mp_ctx()
    ctx_name = _ctx_name(ctx)
    grace_s = _startup_grace_s(ctx_name, timeout_s)

    # Preflight: reject empty / absurdly long code without spawning.
    extracted = _extract_code(str(candidate_code or ""))
    if not extracted.strip():
        info = _empty_code_info(sample_meta, base_total)
        return 0.0, info
    if len(extracted) > max(1, int(_CODE_MAX_CHARS)):
        info = _too_long_code_info(sample_meta, base_total, len(extracted))
        return 0.0, info

    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_worker_run,
        args=(
            str(candidate_code or ""),
            str(setup_code or ""),
            list(tests),
            list(challenge),
            test_script,
            timeout_s,
            mem_mb,
            cpu_s,
            child_conn,
        ),
        daemon=True,
    )

    timed_out = False
    try:
        proc.start()
        try:
            child_conn.close()
        except Exception:
            pass

        proc.join(timeout=timeout_s + grace_s)
        if proc.is_alive():
            timed_out = True
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.join(timeout=0.2)
            except Exception:
                pass
            result = _timeout_result(base_total)
        else:
            result = _recv_or_default(parent_conn, base_total)
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        try:
            proc.close()
        except Exception:
            pass

    total = int(result.get("total", 0)) or 0
    passed = int(result.get("passed", 0)) or 0
    score = float(passed) / float(total) if total > 0 else 0.0

    info = {
        "env": "CodeEnv",
        "task_id": sample_meta.get("task_id"),
        "total": total,
        "passed": passed,
        "failed": max(0, total - passed),
        "fail_examples": result.get("fail_examples", []),
        "errors": result.get("errors", []),
        "traceback": result.get("traceback"),
        "timed_out": timed_out,
        "used_tests": len(tests),
        "used_challenge": len(challenge),
        "used_test_script": bool(test_script and not tests),
        "mp_ctx": ctx_name,
        "timeout_s": timeout_s,
        "startup_grace_s": grace_s,
        "mem_mb": mem_mb,
        "cpu_s": cpu_s,
        "stdio_mode": _STDIO_MODE,
    }
    return score, info


@dataclass
class CodeExecResult:
    passed: bool
    score: float
    details: Dict[str, Any]


def run_code_in_sandbox(*, code: str, tests: Optional[str] = None, timeout_s: float = 2.0) -> CodeExecResult:
    """Single-script sandbox runner."""
    meta: Dict[str, Any] = {}
    if tests:
        meta["test"] = tests

    score, info = run_mbpp_tests(
        candidate_code=code,
        ref_code=None,
        sample_meta=meta,
        timeout=int(timeout_s),
    )
    total = int(info.get("total", 0) or 0)
    passed_n = int(info.get("passed", 0) or 0)
    return CodeExecResult(passed=bool(total > 0 and passed_n == total), score=score, details=info)


# ---------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------

def _missing_tests_info(sample_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "env": "CodeEnv",
        "task_id": sample_meta.get("task_id"),
        "total": 0,
        "passed": 0,
        "failed": 0,
        "fail_examples": [],
        "errors": ["MissingTests"],
        "traceback": None,
        "timed_out": False,
        "used_tests": 0,
        "used_challenge": 0,
        "used_test_script": False,
    }


def _empty_code_info(sample_meta: Dict[str, Any], base_total: int) -> Dict[str, Any]:
    return {
        "env": "CodeEnv",
        "task_id": sample_meta.get("task_id"),
        "total": base_total,
        "passed": 0,
        "failed": base_total,
        "fail_examples": ["<empty candidate code>"],
        "errors": ["EmptyCode"],
        "traceback": None,
        "timed_out": False,
        "used_tests": int(len(sample_meta.get("test_list", []) or [])),
        "used_challenge": int(len(sample_meta.get("challenge_test_list", []) or [])),
        "used_test_script": bool(sample_meta.get("test", None)),
    }


def _too_long_code_info(sample_meta: Dict[str, Any], base_total: int, code_len: int) -> Dict[str, Any]:
    return {
        "env": "CodeEnv",
        "task_id": sample_meta.get("task_id"),
        "total": base_total,
        "passed": 0,
        "failed": base_total,
        "fail_examples": [f"CodeTooLong(len={code_len},max={_CODE_MAX_CHARS})"],
        "errors": ["CodeTooLong"],
        "traceback": None,
        "timed_out": False,
        "used_tests": int(len(sample_meta.get("test_list", []) or [])),
        "used_challenge": int(len(sample_meta.get("challenge_test_list", []) or [])),
        "used_test_script": bool(sample_meta.get("test", None)),
    }


def _timeout_result(base_total: int) -> Dict[str, Any]:
    return {
        "passed": 0,
        "total": base_total,
        "fail_examples": ["<timeout>"],
        "errors": ["TimeoutExceeded"],
        "traceback": None,
    }


def _recv_or_default(conn: Any, base_total: int) -> Dict[str, Any]:
    if conn.poll(0.2):
        try:
            out = conn.recv()
            if isinstance(out, dict):
                return out
            return {
                "passed": 0,
                "total": base_total,
                "fail_examples": ["<bad result type>"],
                "errors": ["BadResultType"],
                "traceback": None,
            }
        except Exception:
            return {
                "passed": 0,
                "total": base_total,
                "fail_examples": ["<recv failed>"],
                "errors": ["RecvFailed"],
                "traceback": None,
            }
    return {
        "passed": 0,
        "total": base_total,
        "fail_examples": ["<no result>"],
        "errors": ["NoResult"],
        "traceback": None,
    }


# ---------------------------------------------------------------------
# Worker + execution
# ---------------------------------------------------------------------

def _silence_torch_dynamo_atexit_best_effort() -> None:
    """Best-effort: reduce noisy atexit dumps if torch is present."""
    try:
        import atexit

        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        du = sys.modules.get("torch._dynamo.utils", None)
        if du is None:
            return
        fn = getattr(du, "dump_compile_times", None)
        if fn is None:
            return
        while True:
            try:
                atexit.unregister(fn)
            except Exception:
                break
        try:
            setattr(du, "dump_compile_times", lambda *a, **k: None)
        except Exception:
            pass
    except Exception:
        return


class _LimitedTextBuffer(io.TextIOBase):
    """Truncate writes by total UTF-8 bytes (best-effort)."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self._buf: List[str] = []
        self._n = 0

    def write(self, s: str) -> int:  # type: ignore[override]
        if not isinstance(s, str):
            s = str(s)
        if self.max_bytes <= 0 or self._n >= self.max_bytes:
            return len(s)

        b = s.encode("utf-8", "ignore")
        rem = self.max_bytes - self._n
        if len(b) <= rem:
            self._buf.append(s)
            self._n += len(b)
            return len(s)

        self._buf.append(b[:rem].decode("utf-8", "ignore"))
        self._n = self.max_bytes
        return len(s)

    def getvalue(self) -> str:
        return "".join(self._buf)


@contextlib.contextmanager
def _redirect_stdio():
    """Default devnull; optional truncation buffer."""
    mode = _STDIO_MODE
    if mode == "truncate":
        out = _LimitedTextBuffer(_STDIO_MAX_BYTES)
        err = _LimitedTextBuffer(_STDIO_MAX_BYTES)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            yield (out, err)
        return

    # devnull (default): avoid pipe blockage + log bombs
    with open(os.devnull, "w") as dn:
        with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            yield None


def _assemble_tests(meta: Dict[str, Any]) -> Tuple[str, List[str], List[str], Optional[str]]:
    setup_code = str(meta.get("test_setup_code", "") or "")
    tests = list(meta.get("test_list", []) or [])
    challenge = list(meta.get("challenge_test_list", []) or [])

    imports = meta.get("test_imports", [])
    if imports:
        setup_code = (setup_code + "\n" + "\n".join(str(x) for x in imports)).strip()

    test_script = meta.get("test", None)
    if isinstance(test_script, str):
        test_script = test_script.strip() or None
    else:
        test_script = None

    return setup_code, tests, challenge, test_script


def _worker_run(
    candidate_code: str,
    setup_code: str,
    tests: List[str],
    challenge: List[str],
    test_script: Optional[str],
    timeout_s: int,
    mem_mb: Optional[int],
    cpu_s: Optional[int],
    conn: Any,
) -> None:
    _silence_torch_dynamo_atexit_best_effort()

    base_total = int(len(tests) + len(challenge) + (1 if (test_script and not tests) else 0))
    try:
        res = _exec_all(
            candidate_code=candidate_code,
            setup_code=setup_code,
            tests=tests,
            challenge=challenge,
            test_script=test_script,
            timeout_s=timeout_s,
            mem_mb=mem_mb,
            cpu_s=cpu_s,
        )
    except Exception as e:
        res = {
            "passed": 0,
            "total": base_total,
            "fail_examples": [f"<exec error: {e.__class__.__name__}: {e}>"],
            "errors": [repr(e)],
            "traceback": traceback.format_exc(),
        }

    try:
        conn.send(res)
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _sum_len_bounded(parts: Iterable[Any], limit: int) -> int:
    """Fast-ish: stop once exceeding limit."""
    limit = max(0, int(limit))
    total = 0
    for x in parts:
        try:
            s = x if isinstance(x, str) else str(x)
        except Exception:
            s = ""
        total += len(s)
        if total > limit:
            return total
    return total


def _exec_all(
    *,
    candidate_code: str,
    setup_code: str,
    tests: List[str],
    challenge: List[str],
    test_script: Optional[str],
    timeout_s: int,
    mem_mb: Optional[int],
    cpu_s: Optional[int],
) -> Dict[str, Any]:
    env = _mk_safe_env()
    base_total = int(len(tests) + len(challenge) + (1 if (test_script and not tests) else 0))

    _apply_rlimits(timeout_s, mem_mb=mem_mb, cpu_s=cpu_s)

    # Hard caps to avoid pathological compile/exec storms.
    code = _extract_code(candidate_code)
    if not code.strip():
        return {
            "passed": 0,
            "total": base_total,
            "fail_examples": ["<empty candidate code>"],
            "errors": ["EmptyCode"],
            "traceback": None,
        }
    if len(code) > max(1, int(_CODE_MAX_CHARS)):
        return {
            "passed": 0,
            "total": base_total,
            "fail_examples": [f"CodeTooLong(len={len(code)},max={_CODE_MAX_CHARS})"],
            "errors": ["CodeTooLong"],
            "traceback": None,
        }

    assert_budget = max(1, int(_ASSERT_MAX_CHARS))
    if _sum_len_bounded(list(tests) + list(challenge), assert_budget) > assert_budget:
        return {
            "passed": 0,
            "total": base_total,
            "fail_examples": [f"AssertsTooLong(max={_ASSERT_MAX_CHARS})"],
            "errors": ["AssertsTooLong"],
            "traceback": None,
        }

    deferred_setup = ""
    with _redirect_stdio():
        if setup_code:
            imp, rest = _split_setup_imports(setup_code)
            if imp:
                exec(imp, env, env)
            if rest:
                try:
                    exec(rest, env, env)
                except NameError:
                    deferred_setup = rest  # likely references candidate symbols

        exec(code, env, env)

        if deferred_setup:
            exec(deferred_setup, env, env)

        passed = 0
        total = 0
        failures: List[str] = []

        for group_name, group in (("test_list", tests), ("challenge_test_list", challenge)):
            for i, assert_code in enumerate(group):
                total += 1
                try:
                    exec(str(assert_code), env, env)
                    passed += 1
                except Exception as e:
                    failures.append(f"{group_name}[{i}]: {e.__class__.__name__}: {e}")

        if (not tests) and test_script:
            total += 1
            try:
                exec(test_script, env, env)
                passed += 1
            except Exception as e:
                failures.append(f"test_script: {e.__class__.__name__}: {e}")

    return {
        "passed": passed,
        "total": total,
        "fail_examples": failures[:20],
        "errors": [],
        "traceback": None,
    }


def _apply_rlimits(timeout_s: int, *, mem_mb: Optional[int], cpu_s: Optional[int]) -> None:
    """Best-effort resource limits (Linux-only; silently no-op elsewhere)."""
    try:
        import resource  # type: ignore

        # CPU
        if cpu_s is None:
            cpu = _env_int("C3_CODE_CPU_LIMIT_S", 0)
            cpu = max(int(timeout_s) + 1, 16) if cpu <= 0 else cpu
        else:
            cpu = int(cpu_s)
        if cpu > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))

        # Memory (address space)
        if mem_mb is None:
            m = _env_int("C3_CODE_MAX_MEM_MB", 4096)
        else:
            m = int(mem_mb)
        if m > 0:
            lim = int(m) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except Exception:
        return


def _split_setup_imports(code: str) -> Tuple[str, str]:
    """Split setup_code into import-only prefix and the rest (order-preserving)."""
    import_lines: List[str] = []
    rest_lines: List[str] = []
    for ln in (code or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            rest_lines.append(ln)
            continue
        if _RE_IMPORT_LINE.match(s):
            import_lines.append(ln)
        else:
            rest_lines.append(ln)
    return "\n".join(import_lines).strip(), "\n".join(rest_lines).strip()


def _extract_code(s: str) -> str:
    """Extract python code from fenced formats; fallback to raw text."""
    s = (s or "").strip()
    if not s:
        return ""
    m = _RE_FENCE_PY.search(s)
    if m:
        return (m.group(1) or "").strip()
    m = _RE_FENCE_ANY.search(s)
    if m:
        return (m.group(1) or "").strip()
    return s


# ---------------------------------------------------------------------
# Sandbox environment
# ---------------------------------------------------------------------

def _mk_safe_env() -> Dict[str, Any]:
    import builtins as _builtins

    allowed_builtins = {
        # core
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "callable": callable,
        "complex": complex,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "print": print,
        # runtime essentials
        "isinstance": isinstance,
        "issubclass": issubclass,
        "getattr": getattr,
        "setattr": setattr,
        "hasattr": hasattr,
        "delattr": delattr,
        "iter": iter,
        "next": next,
        "repr": repr,
        "chr": chr,
        "ord": ord,
        "divmod": divmod,
        "frozenset": frozenset,
        "type": type,
        "object": object,
        "__build_class__": _builtins.__build_class__,
        # exceptions
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "RuntimeError": RuntimeError,
        "AssertionError": AssertionError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "StopIteration": StopIteration,
    }

    # Default: stdlib-only (keep cold-start light, reduce abuse surface).
    allowed_imports = {
        "math",
        "re",
        "string",
        "typing",
        "dataclasses",
        "collections",
        "itertools",
        "functools",
        "operator",
        "heapq",
        "bisect",
        "statistics",
        "fractions",
        "decimal",
        "random",
    }

    extra = _env_str("C3_CODE_EXTRA_IMPORTS", "")
    if extra:
        for name in extra.split(","):
            name = name.strip()
            if name:
                allowed_imports.add(name)

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        n = str(name or "")
        base = n.split(".", 1)[0] if n else ""
        if n not in allowed_imports and base not in allowed_imports:
            raise ImportError(f"import of '{n}' is not allowed")
        mod = _builtins.__import__(n, globals, locals, fromlist, level)
        return sys.modules.get(n, mod)

    safe_builtins = dict(allowed_builtins)
    safe_builtins["__import__"] = _safe_import

    return {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
    }
