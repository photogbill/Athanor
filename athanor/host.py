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


def source_checkout() -> Path | None:
    """The folder Athanor runs from when it is a source checkout (or an
    editable install of one): ``pyproject.toml`` beside the ``athanor``
    package. None when it is installed into site-packages."""
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").is_file() and (root / "athanor" / "__init__.py").is_file():
        return root
    return None


def data_dir_choice() -> tuple[Path, str]:
    """(the data folder, how it was chosen) — the rule behind ``default_data_dir``:

    1. ``ATHANOR_DATA`` (or ``--data`` on the command line), when set;
    2. ``<checkout>/data`` when Athanor runs from a source checkout — its
       data stays beside its code, on the drive the code lives on, never
       in a Windows user profile it was not told about (Bill, 2026-10-02);
    3. otherwise the per-user data folder of the platform
       (``%LOCALAPPDATA%\\Athanor``, ``~/Library/Application Support/Athanor``,
       ``$XDG_DATA_HOME/athanor``).

    A host application replaces all of this with its own ``data_dir``."""
    env = os.environ.get("ATHANOR_DATA")
    if env:
        return Path(env), "ATHANOR_DATA"
    checkout = source_checkout()
    if checkout is not None:
        return checkout / "data", "checkout"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Athanor", "per-user"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Athanor", "per-user"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "athanor", "per-user"


def default_data_dir() -> Path:
    """``ATHANOR_DATA`` if set; else ``<checkout>/data`` when running from a
    source checkout; else the per-user data folder (``data_dir_choice``)."""
    return data_dir_choice()[0]


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
