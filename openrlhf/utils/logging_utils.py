# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for OpenRLHF.

This module keeps backward compatibility with the original `init_logger(name)` API
while adding opt-in local file logging.

Env vars (all optional):
- OPENRLHF_LOG_DIR: if set, append logs to a per-process file under this directory.
- OPENRLHF_LOG_PREFIX: filename prefix (default: "openrlhf").
- OPENRLHF_LOG_CONSOLE: "0" disables most console logs (keeps ERROR+), otherwise INFO.
- OPENRLHF_LOG_LEVEL: root/openrlhf logger level (default: DEBUG).
- OPENRLHF_REDIRECT_STD: "1" redirects stdout/stderr to files in OPENRLHF_LOG_DIR.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        msg = super().format(record)
        if getattr(record, "message", "") != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


class _TeeStream:
    """A minimal tee stream compatible with libraries expecting a real file-like object.

    Ray (faulthandler) requires sys.stderr to have a working fileno().
    """

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, s):
        # Behave like a text stream: return number of chars written.
        n = 0
        for st in self._streams:
            try:
                st.write(s)
                n = len(s)
            except Exception:
                pass
        return n

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        # Report TTY status based on the first stream (usually the console one).
        try:
            return bool(getattr(self._streams[0], "isatty", lambda: False)())
        except Exception:
            return False

    def writable(self):
        return True

    def fileno(self):
        """Return an OS-level file descriptor.

        Prefer the last underlying stream's fileno (usually the redirected log file),
        so faulthandler writes go into the log file instead of the console.
        """
        for st in reversed(self._streams):
            try:
                fn = getattr(st, "fileno", None)
                if fn is not None:
                    return fn()
            except Exception:
                continue
        raise OSError("No underlying stream supports fileno()")

    def close(self):
        """Close only non-stdio streams (i.e., the files we opened)."""
        for st in self._streams:
            # Never close the original stdio objects.
            if st in (sys.__stdout__, sys.__stderr__, sys.__stdin__):
                continue
            try:
                close_fn = getattr(st, "close", None)
                if close_fn is not None:
                    close_fn()
            except Exception:
                pass

    def __getattr__(self, name):
        # Delegate attributes like encoding/errors to the first stream that has them.
        for st in self._streams:
            try:
                if hasattr(st, name):
                    return getattr(st, name)
            except Exception:
                continue
        raise AttributeError(name)


_root_logger = logging.getLogger("openrlhf")
_default_handler: Optional[logging.Handler] = None

_file_handler: Optional[logging.Handler] = None
_file_handler_path: Optional[str] = None
_std_redirected: bool = False


def _env_flag(key: str, default: bool = False) -> bool:
    v = os.getenv(key, "")
    if v == "":
        return bool(default)
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _console_level_from_env() -> int:
    v = os.getenv("OPENRLHF_LOG_CONSOLE", "1").strip().lower()
    if v in {"0", "false", "no", "n", "off"}:
        return logging.ERROR
    return logging.INFO


def _root_level_from_env() -> int:
    s = os.getenv("OPENRLHF_LOG_LEVEL", "DEBUG").strip().upper()
    return getattr(logging, s, logging.DEBUG)


def _file_path_from_env() -> Optional[str]:
    log_dir = os.getenv("OPENRLHF_LOG_DIR", "").strip()
    if not log_dir:
        return None
    prefix = os.getenv("OPENRLHF_LOG_PREFIX", "openrlhf").strip() or "openrlhf"
    # keep filenames filesystem-friendly
    prefix = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in prefix)
    pid = os.getpid()
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{prefix}.pid{pid}.log")


def _ensure_file_handler() -> Optional[logging.Handler]:
    """Create a per-process file handler if OPENRLHF_LOG_DIR is set."""
    global _file_handler, _file_handler_path

    path = _file_path_from_env()
    if path is None:
        return None

    if _file_handler is not None and _file_handler_path == path:
        return _file_handler

    # Close previous handler (if any) and re-create.
    if _file_handler is not None:
        try:
            _file_handler.close()
        except Exception:
            pass

    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT))

    _file_handler = fh
    _file_handler_path = path
    return fh


def _attach_handler_once(logger: logging.Logger, handler: logging.Handler) -> None:
    for h in logger.handlers:
        if h is handler:
            return
        # de-dupe by file path for FileHandler
        if isinstance(h, logging.FileHandler) and isinstance(handler, logging.FileHandler):
            try:
                if getattr(h, "baseFilename", None) == getattr(handler, "baseFilename", None):
                    return
            except Exception:
                pass
    logger.addHandler(handler)


def _setup_logger() -> None:
    """Initialize the module-level openrlhf root logger."""
    global _default_handler

    _root_logger.setLevel(_root_level_from_env())

    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _root_logger.addHandler(_default_handler)

    _default_handler.setLevel(_console_level_from_env())
    _default_handler.setFormatter(NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT))

    fh = _ensure_file_handler()
    if fh is not None:
        _attach_handler_once(_root_logger, fh)

    # Avoid propagating to parent root logger.
    _root_logger.propagate = False


def _redirect_std_streams_if_needed(log_dir: str, *, tee: bool = True) -> None:
    global _std_redirected
    if _std_redirected:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    prefix = os.getenv("OPENRLHF_LOG_PREFIX", "openrlhf").strip() or "openrlhf"
    prefix = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in prefix)

    out_path = Path(log_dir) / f"{prefix}.pid{pid}.stdout"
    err_path = Path(log_dir) / f"{prefix}.pid{pid}.stderr"

    f_out = open(out_path, "a", encoding="utf-8", buffering=1)
    f_err = open(err_path, "a", encoding="utf-8", buffering=1)

    quiet_console = os.getenv("OPENRLHF_LOG_CONSOLE", "1").strip().lower() in {"0", "false", "no", "off"}
    do_tee = bool(tee and (not quiet_console))

    sys.stdout = _TeeStream(sys.__stdout__, f_out) if do_tee else f_out  # type: ignore
    sys.stderr = _TeeStream(sys.__stderr__, f_err) if do_tee else f_err  # type: ignore

    _std_redirected = True


def setup_run_logging(
    run_dir: str,
    *,
    log_subdir: str = "logs",
    prefix: str = "openrlhf",
    console: Optional[bool] = None,
    redirect_std: Optional[bool] = None,
    tee_std: bool = True,
) -> str:
    """Enable local file logging for the current process (and any children inheriting env vars).

    Returns the resolved log directory.
    """
    run_dir = str(run_dir)
    log_dir = str(Path(run_dir) / log_subdir)

    os.environ["OPENRLHF_LOG_DIR"] = log_dir
    os.environ.setdefault("OPENRLHF_LOG_PREFIX", prefix)

    if console is not None:
        os.environ["OPENRLHF_LOG_CONSOLE"] = "1" if bool(console) else "0"

    if redirect_std is not None:
        os.environ["OPENRLHF_REDIRECT_STD"] = "1" if bool(redirect_std) else "0"

    # (Re)configure root handler/levels and attach file handler.
    _setup_logger()

    # Attach file handler to existing loggers that were created before setup_run_logging().
    fh = _ensure_file_handler()
    if fh is not None:
        for obj in list(logging.Logger.manager.loggerDict.values()):
            if not isinstance(obj, logging.Logger):
                continue
            # keep it conservative: only attach to loggers that already use our default handler
            # or are within the openrlhf namespace.
            if obj.name.startswith("openrlhf") or _default_handler in obj.handlers:
                _attach_handler_once(obj, fh)

    if _env_flag("OPENRLHF_REDIRECT_STD", default=False):
        _redirect_std_streams_if_needed(log_dir, tee=tee_std)

    return log_dir


# Initialize on import (thread-safe by import semantics).
_setup_logger()


def init_logger(name: str):
    """Create/return a logger with OpenRLHF formatting and (optional) file logging."""
    # Ensure root is set up (levels/handlers may change at runtime via env).
    _setup_logger()

    logger = logging.getLogger(name)
    logger.setLevel(_root_level_from_env())

    if _default_handler is not None:
        _attach_handler_once(logger, _default_handler)

    fh = _ensure_file_handler()
    if fh is not None:
        _attach_handler_once(logger, fh)

    logger.propagate = False
    return logger
