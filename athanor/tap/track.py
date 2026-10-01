"""The Tap's side of a Waterfall recording: one fixed-size record per step.

A recording with a Tap has a third file, ``<stem>.athrec-tap``, beside the
meta and data files: for every step (every token the model wrote), the rows
the Tap copied from the forward pass that produced that step's logits —
the experts each layer chose, or each layer's output, or whatever was
asked for. Record layout, little-endian, packed::

    uint32      flags          1 = a stream was missing at this step,
                               2 = nothing was captured at this step,
                               4 = the Tap had stopped (limit or error)
    <stream>    one field per stream, in the order the meta file lists
                them (``tap.layout``): int32 or float32, a fixed count

The meta file's ``tap`` block describes it (``docs/formats/recording.md``);
numpy reads it in one line:
``np.fromfile(path, dtype=layout_dtype(meta['tap']['layout']))``
(``athanor.waterfall.fileformat.layout_dtype``).
"""

from __future__ import annotations

import time

import numpy as np

from ..waterfall.fileformat import layout_dtype
from .core import split_name

TAP_FLAG_PARTIAL = 1
TAP_FLAG_NONE = 2
TAP_FLAG_STOPPED = 4
FLAG_MEANINGS = {TAP_FLAG_PARTIAL: "a stream was missing at this step",
                 TAP_FLAG_NONE: "nothing was captured at this step",
                 TAP_FLAG_STOPPED: "the Tap had stopped (its limit, or an error)"}
CHUNK = 256
DEFAULT_LIMIT = 2 << 30          # 2 GiB of records held in memory


def _order(name: str):
    kind, layer = split_name(name)
    return (layer is None, layer if layer is not None else 0, kind)


class TapTrack:
    """Collects one record per recorded step. ``step(i, rows)`` with the
    step's index keeps it aligned with the Waterfall even if a step is
    missed; ``save`` pads to the recording's length."""

    def __init__(self, streams, *, limit_bytes: int = DEFAULT_LIMIT, facts: dict | None = None):
        self.streams = list(streams)
        self.limit_bytes = int(limit_bytes)
        self.facts = dict(facts or {})
        self.layout: list | None = None      # [{"name", "kind", "layer", "dtype", "count"}]
        self.dtype: np.dtype | None = None
        self._chunks: list = []
        self._buf = None
        self._n_buf = 0
        self.n_steps = 0
        self.error: str | None = None
        self.stopped_at: int | None = None
        self.unavailable: str | None = None  # the Tap never ran, and why
        self.notes: dict = {}
        self.seconds = 0.0
        self.sessions: list = []             # Tap.report() of each part

    # --------------------------------------------------------------- steps
    def step(self, i: int, rows: dict | None) -> None:
        """The rows for step ``i`` (name -> 1-D array), or None if the Tap
        gave nothing."""
        start = time.perf_counter()
        try:
            while self.n_steps < i:          # a step the Tap missed
                self._append(None, TAP_FLAG_NONE)
            if self.n_steps > i:
                return
            if self.stopped_at is not None:
                self._append(None, TAP_FLAG_STOPPED)
                return
            if rows and self.layout is None:
                self._fix_layout(rows)
            if self.layout is not None and self.limit_bytes and \
                    (self.n_steps + 1) * self.dtype.itemsize > self.limit_bytes:
                self.stopped_at = self.n_steps
                self.note(f"the Tap stopped at step {self.n_steps}: its records reached the "
                          f"limit of {self.limit_bytes:,} bytes")
                self._append(None, TAP_FLAG_STOPPED)
                return
            self._append(rows, 0)
        except Exception as exc:              # the Tap stops; the recording goes on
            self.fail(exc)                    # (the next step, or finish(), re-aligns)
        finally:
            self.seconds += time.perf_counter() - start

    def _fix_layout(self, rows: dict) -> None:
        layout = []
        for name in sorted(rows, key=_order):
            a = np.asarray(rows[name]).reshape(-1)
            kind, layer = split_name(name)
            dt = "<i4" if a.dtype.kind in "iu" else "<f4"
            layout.append({"name": name, "kind": kind, "layer": layer, "dtype": dt,
                           "count": int(a.shape[0])})
        self.layout = layout
        self.dtype = layout_dtype(layout)
        # steps before this one captured nothing; they become blank records
        blanks, self.n_steps = self.n_steps, 0
        for _ in range(blanks):
            self._append_blank(TAP_FLAG_NONE)

    def _next_row(self):
        if self._buf is None or self._n_buf == len(self._buf):
            if self._buf is not None:
                self._chunks.append(self._buf)
            self._buf = np.zeros(CHUNK, self.dtype)
            self._n_buf = 0
        row = self._buf[self._n_buf]
        self._n_buf += 1
        return row

    def _append_blank(self, flags: int) -> None:
        if self.dtype is None:
            self.n_steps += 1
            return
        row = self._next_row()
        for s in self.layout:
            row[s["name"]] = -1 if s["dtype"] == "<i4" else np.nan
        row["flags"] = flags
        self.n_steps += 1

    def _append(self, rows: dict | None, flags: int) -> None:
        if self.dtype is None or not rows:
            self._append_blank(flags | (TAP_FLAG_NONE if not flags else 0))
            return
        row = self._next_row()
        for s in self.layout:
            name = s["name"]
            a = rows.get(name)
            if a is None:
                row[name] = -1 if s["dtype"] == "<i4" else np.nan
                flags |= TAP_FLAG_PARTIAL
                continue
            a = np.asarray(a).reshape(-1)
            if a.shape[0] != s["count"]:
                row[name] = -1 if s["dtype"] == "<i4" else np.nan
                flags |= TAP_FLAG_PARTIAL
                self.note(f"{name}: {a.shape[0]} values at a step, {s['count']} expected")
                continue
            row[name] = a
        extra = set(rows) - {s["name"] for s in self.layout}
        if extra:
            self.note("streams that appeared after the first step are not kept: "
                      + ", ".join(sorted(extra)[:6]) + ("…" if len(extra) > 6 else ""))
        row["flags"] = flags
        self.n_steps += 1

    def records(self) -> np.ndarray | None:
        if self.dtype is None:
            return None
        parts = self._chunks + ([self._buf[: self._n_buf]] if self._buf is not None else [])
        return np.concatenate(parts) if parts else np.zeros(0, self.dtype)

    # ------------------------------------------------------------- trouble
    def stop(self, reason: str) -> None:
        """The Tap itself stopped (an error inside it, or its limit): every
        step from now on is flagged as stopped, and the reason kept."""
        if self.error is None:
            self.error = reason
        if self.stopped_at is None:
            self.stopped_at = self.n_steps

    def note(self, message: str) -> None:
        self.notes[message] = self.notes.get(message, 0) + 1

    def fail(self, exc: BaseException) -> None:
        if self.error is None:
            self.error = f"{type(exc).__name__}: {exc}"
            if self.stopped_at is None:
                self.stopped_at = self.n_steps
            try:
                from .. import log
                log.exception("tap-track-failed", exc, step=self.n_steps)
            except Exception:
                pass

    # -------------------------------------------------------------- saving
    def finish(self, n_steps: int) -> tuple:
        """(records, meta block) for a recording of ``n_steps`` steps:
        padded or cut to exactly that many records."""
        from ..labels import MEASURED
        while self.n_steps < n_steps:
            self._append_blank(TAP_FLAG_STOPPED if self.stopped_at is not None else TAP_FLAG_NONE)
        rec = self.records()
        if rec is not None:
            rec = rec[:n_steps]
        errors = [r.get("error") for r in self.sessions if r.get("error")]
        notes = [f"{m} (×{n})" if n > 1 else m for m, n in self.notes.items()]
        for r in self.sessions:
            notes.extend(r.get("notes") or [])
        block = {
            "streams": self.streams,
            "layout": self.layout or [],
            "record_bytes": int(self.dtype.itemsize) if self.dtype is not None else 0,
            "n_steps": int(n_steps) if rec is not None else 0,
            "flags": {str(k): v for k, v in FLAG_MEANINGS.items()},
            "rows": "outputs",
            "how": ("copied from llama.cpp's graph as it computed (the ggml evaluation "
                    "callback): for each step, the row of each tensor that belongs to the "
                    "token whose logits that step was sampled from"),
            "label": MEASURED,
            "model": self.facts,
            "unavailable": self.unavailable,
            "error": self.error or (errors[0] if errors else None),
            "stopped_at": self.stopped_at,
            "notes": sorted(set(notes)),
            "overhead_ms_per_step": (1000 * (self.seconds + sum(
                r.get("copy_seconds") or 0 for r in self.sessions)) / n_steps) if n_steps else None,
            "forward_passes": sum(r.get("forward_passes") or 0 for r in self.sessions),
        }
        return rec, block
