"""Athanor's own log — what happened, kept, so every failure can be learned from.

Bill, 2026-09-28: *"we would want a comprehensive logging setup to ensure we
learn the most possible."*

WHERE (all local; nothing is ever sent anywhere)

* ``<data_dir>/logs/athanor-YYYY-MM-DD.jsonl`` — one JSON object per line.
* ``<data_dir>/logs/crash-YYYY-MM-DD.log`` — if llama.cpp (or anything
  native) kills the process, Python's ``faulthandler`` writes every thread's
  stack here as it dies. A normal exception never gets here; it is in the
  JSON log with its traceback.

WHAT

* every run of the command line: arguments, versions, working folder, exit
  code, duration, and any exception with its full traceback;
* every vocabulary load: the file (size, modification time), how long it
  took, what llama.cpp made of it, and llama.cpp's COMPLETE load log at every
  level, debug included;
* anything llama.cpp says inside a captured block;
* a **breadcrumb**, flushed to disk before it continues, ahead of each
  native operation that could take the process down: the last breadcrumb
  before a crash says what Athanor was doing when it happened.

CONTROL: ``ATHANOR_LOG=0`` turns it off. A host can call ``configure``. The
format is ``docs/formats/log.md``. Logging never breaks the work: a log that
cannot be written is skipped, silently.
"""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1

_lock = threading.Lock()
_state: dict = {"enabled": None, "dir": None, "crash_file": None}


def configure(*, enabled: bool | None = None, directory: str | os.PathLike | None = None) -> None:
    """Turn the log on or off, or send it to another folder (for hosts)."""
    if enabled is not None:
        _state["enabled"] = bool(enabled)
    if directory is not None:
        _state["dir"] = Path(directory)


def reset() -> None:
    """Back to the defaults: on unless ATHANOR_LOG=0, in the host's data folder."""
    _state["enabled"] = None
    _state["dir"] = None


def enabled() -> bool:
    if _state["enabled"] is not None:
        return _state["enabled"]
    return os.environ.get("ATHANOR_LOG", "1").strip().lower() not in ("0", "off", "false", "no")


def log_dir() -> Path:
    if _state["dir"] is not None:
        return _state["dir"]
    from .host import get_host
    return get_host().data_dir() / "logs"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def log_path(day: datetime | None = None) -> Path:
    return log_dir() / f"athanor-{(day or _now()).strftime('%Y-%m-%d')}.jsonl"


def event(kind: str, /, *, sync: bool = False, **fields: Any) -> None:
    """One line in the log. ``sync`` forces it to disk before returning."""
    if not enabled():
        return
    try:
        from .util import to_jsonable
        now = _now()
        rec = {"time": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
               "format": FORMAT_VERSION, "pid": os.getpid(), "event": kind, **fields}
        line = json.dumps(to_jsonable(rec), ensure_ascii=False) + "\n"
        path = log_path(now)
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                if sync:
                    f.flush()
                    os.fsync(f.fileno())
    except Exception:
        pass  # the log must never be the thing that fails


def breadcrumb(what: str, /, **fields: Any) -> None:
    """Written to disk BEFORE a native call that could end the process."""
    event("breadcrumb", sync=True, what=what, **fields)


def exception(kind: str, exc: BaseException, /, **fields: Any) -> None:
    event(kind, error=type(exc).__name__, message=str(exc),
          traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
          **fields)


def enable_crash_log() -> Path | None:
    """Have a native crash write every thread's stack to the crash log.

    The command line turns this on. A host that already uses faulthandler
    should leave it alone (faulthandler holds one file per process).
    """
    if not enabled():
        return None
    try:
        path = log_dir() / f"crash-{_now().strftime('%Y-%m-%d')}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(path, "a", encoding="utf-8")
        # no arguments here: they can hold the user's text (the JSON log
        # keeps a redacted copy under the same pid)
        f.write(f"\n=== {_now().isoformat(timespec='seconds')} pid {os.getpid()}\n")
        f.flush()
        faulthandler.enable(file=f, all_threads=True)
        old = _state.get("crash_file")
        _state["crash_file"] = f
        if old is not None and old is not f:
            try:
                old.close()
            except Exception:
                pass
        return path
    except Exception:
        return None


def read_events(day: datetime | None = None) -> list:
    """The log of one day, parsed (damaged lines skipped)."""
    path = log_path(day)
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
