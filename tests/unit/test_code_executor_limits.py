"""The sandbox resource limits land on the worker child and on nothing else.

`_apply_rlimits` sets a CPU budget and a 4 GB address space with `setrlimit`, which
applies to whatever process calls it. That is right for the child that runs one
untrusted sample and wrong for every other caller. On 2026-09-19 a test reached
`_exec_all` directly, the limits landed on the pytest process, and the Linux CI run
ended with an INTERNALERROR inside pathlib instead of with a failing test: on Windows
`import resource` fails and the same call is a silent no-op, so the same commit looked
clean there. These tests pin both halves of the guard that followed.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Tuple

import pytest

from c3.envs.code import executor as code_executor


def _real_limits() -> Dict[str, Any]:
    """RLIMIT_CPU and RLIMIT_AS of this process, or None where there is no `resource`."""
    try:
        import resource  # type: ignore
    except Exception:
        return {}
    return {
        "RLIMIT_CPU": resource.getrlimit(resource.RLIMIT_CPU),
        "RLIMIT_AS": resource.getrlimit(resource.RLIMIT_AS),
    }


def _fake_resource(recorder: List[Tuple[int, Tuple[int, int]]]) -> types.ModuleType:
    """A stand-in for `resource` that records instead of limiting anything.

    `_apply_rlimits` imports the module inside the function, so replacing it in
    sys.modules is enough to watch the calls without a Linux runner and without putting
    a real limit on the process running the tests.
    """
    module = types.ModuleType("resource")
    module.RLIMIT_CPU = 0
    module.RLIMIT_AS = 9
    module.setrlimit = lambda which, pair: recorder.append((which, pair))
    module.getrlimit = lambda which: (-1, -1)
    return module


def test_a_main_process_keeps_its_own_resource_limits() -> None:
    """The guard: a call from a process multiprocessing did not start changes nothing."""
    before = _real_limits()
    if not before:
        pytest.skip("this platform has no `resource` module, so there is nothing to limit")

    assert code_executor.mp.parent_process() is None, "pytest is expected to be a main process"
    code_executor._apply_rlimits(120, mem_mb=None, cpu_s=None)
    assert _real_limits() == before


def test_a_worker_child_still_gets_the_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half: the limits are not gone, they are the child's.

    The child is simulated rather than started: what is under test is the guard, and a
    real child would put a real 4 GB limit somewhere for no reason.
    """
    calls: List[Tuple[int, Tuple[int, int]]] = []
    monkeypatch.setitem(sys.modules, "resource", _fake_resource(calls))
    monkeypatch.setattr(code_executor.mp, "parent_process", lambda: object())

    code_executor._apply_rlimits(120, mem_mb=None, cpu_s=None)

    assert calls == [
        (0, (121, 121)),  # RLIMIT_CPU: the timeout plus one second
        (9, (4096 * 1024 * 1024, 4096 * 1024 * 1024)),  # RLIMIT_AS: C3_CODE_MAX_MEM_MB
    ]


def test_the_evaluation_path_still_asks_for_the_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_exec_all` must keep requesting them, so that the child keeps receiving them.

    `_apply_rlimits` is replaced by a recorder, so the one call in this path that can
    touch the process running the tests does not, and the sample executed is `pass`.
    """
    seen: List[Tuple[int, Any, Any]] = []
    monkeypatch.setattr(
        code_executor,
        "_apply_rlimits",
        lambda timeout_s, *, mem_mb, cpu_s: seen.append((timeout_s, mem_mb, cpu_s)),
    )

    result = code_executor._exec_all(
        candidate_code="pass",
        setup_code="",
        tests=[],
        challenge=[],
        test_script=None,
        timeout_s=120,
        mem_mb=None,
        cpu_s=None,
    )

    assert seen == [(120, None, None)]
    assert result["total"] == 0
