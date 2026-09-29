"""The host port — how an application lends Athanor what it has.

A host is any object with these five methods. Athanor never imports its
host; the host sets itself with ``set_host(...)``. ``NullHost`` is the
default and needs nothing, which is how the command line runs.

What a host lends:

* ``borrow_gpu(reason)`` — a context manager held around anything that
  loads weights onto the GPU. A host with its own model loaded unloads it on
  entry and restores it on exit (ATK does this with its AI-queue ticket).
* ``vram_plan()`` — what the host knows about free GPU memory, or None.
* ``model_dirs()`` — folders the host keeps models in (searched by the
  library views; never globbed blindly).
* ``data_dir()`` — where the notebook, recordings and results go.
* ``settings()`` — a dict of host settings Athanor may read (read-only).

Phase 1 (the file tabs) loads no weights, so ``borrow_gpu`` is not yet
called; the port is here now so hosts are written against the final shape.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import ContextManager, Protocol, runtime_checkable


@runtime_checkable
class Host(Protocol):
    def borrow_gpu(self, reason: str) -> ContextManager: ...
    def vram_plan(self) -> dict | None: ...
    def model_dirs(self) -> list: ...
    def data_dir(self) -> Path: ...
    def settings(self) -> dict: ...


def default_data_dir() -> Path:
    """``ATHANOR_DATA`` if set; else the per-user data folder."""
    env = os.environ.get("ATHANOR_DATA")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Athanor"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Athanor"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "athanor"


class NullHost:
    """The host for running on its own: no GPU hand-off, no model folders
    beyond ``ATHANOR_MODELS`` (os.pathsep-separated), the per-user data
    folder, no settings."""

    def borrow_gpu(self, reason: str) -> ContextManager:
        return contextlib.nullcontext()

    def vram_plan(self) -> dict | None:
        return None

    def model_dirs(self) -> list:
        env = os.environ.get("ATHANOR_MODELS", "")
        return [Path(p) for p in env.split(os.pathsep) if p]

    def data_dir(self) -> Path:
        return default_data_dir()

    def settings(self) -> dict:
        return {}


_host: Host = NullHost()


def set_host(host: Host) -> None:
    """Install the application's host. Checked against the protocol."""
    global _host
    if not isinstance(host, Host):
        missing = [m for m in ("borrow_gpu", "vram_plan", "model_dirs", "data_dir", "settings")
                   if not callable(getattr(host, m, None))]
        raise TypeError(f"not a Host — missing {', '.join(missing)}")
    _host = host


def get_host() -> Host:
    return _host


def reset_host() -> None:
    """Back to NullHost (tests, and hosts shutting down)."""
    global _host
    _host = NullHost()
