"""The recording format: ``<stem>.athrec-meta`` (JSON) + ``<stem>.athrec-data``.

Laid out the way SigMF lays out an RF capture (a JSON description beside a
binary file, ``global`` / ``captures`` / ``annotations``) so an RF analyst
recognises it, but it is not a SigMF recording: the binary file holds one
fixed-size record per token the model wrote, not samples.

One record, little-endian, no padding (``step_bytes = 32 + 8·k``)::

    offset  type       field
    0       int32      chosen     token id the sampler picked
    4       int32      rank       its rank in the model's own distribution
                                  (0 = the model's favourite; exact, even
                                  outside the stored candidates)
    8       float32    logprob    its log-probability, nats
    12      float32    entropy    of the whole distribution, nats
    16      float32    tail       probability outside the stored candidates
    20      uint32     flags      1 = end of generation, 2 = control token,
                                  4 = the logits held NaN or ±inf
    24      float64    t          seconds since the recording began
    32      int32[k]   ids        the k most probable tokens, best first
    32+4k   float32[k] logprobs   their log-probabilities, nats

Everything is the model's RAW distribution — read before any sampler
(penalties, grammar, top-k/p, temperature) touched it. The sampler's own
settings are in the meta file.

docs/formats/recording.md is the full specification; this module is the
reference implementation, and needs nothing but numpy.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
META_SUFFIX = ".athrec-meta"
DATA_SUFFIX = ".athrec-data"
TAP_SUFFIX = ".athrec-tap"      # optional: what the Tap copied, one record per step
HEADER_BYTES = 32
DEFAULT_K = 256

FLAG_EOG = 1
FLAG_CONTROL = 2
FLAG_NONFINITE = 4


class RecordingError(ValueError):
    """A recording that is missing, damaged, or of a version Athanor cannot read."""


def step_dtype(k: int) -> np.dtype:
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, not {k!r}")
    dt = np.dtype([("chosen", "<i4"), ("rank", "<i4"), ("logprob", "<f4"),
                   ("entropy", "<f4"), ("tail", "<f4"), ("flags", "<u4"),
                   ("t", "<f8"), ("ids", "<i4", (int(k),)), ("logprobs", "<f4", (int(k),))])
    assert dt.itemsize == HEADER_BYTES + 8 * int(k)
    return dt


LAYOUT = ("chosen:int32 rank:int32 logprob:float32 entropy:float32 tail:float32 "
          "flags:uint32 t:float64 ids:int32[k] logprobs:float32[k]; little-endian, packed")


def stems(path) -> tuple[Path, Path]:
    """(meta, data) paths for a recording named by any of its files or its stem."""
    p = Path(path)
    name = p.name
    for suffix in (META_SUFFIX, DATA_SUFFIX, TAP_SUFFIX):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            return p.with_name(base + META_SUFFIX), p.with_name(base + DATA_SUFFIX)
    return p.with_name(name + META_SUFFIX), p.with_name(name + DATA_SUFFIX)


def layout_dtype(layout: list) -> np.dtype:
    """The Tap file's record dtype, from the meta file's ``tap.layout``:
    ``flags`` (uint32), then one field per stream, a fixed count of int32
    or float32 each. Little-endian, packed."""
    fields = [("flags", "<u4")]
    for s in layout:
        dt = s["dtype"]
        if dt not in ("<i4", "<f4"):
            raise RecordingError(f"tap stream {s.get('name')!r}: dtype {dt!r} is not <i4 or <f4")
        fields.append((str(s["name"]), dt, (int(s["count"]),)))
    return np.dtype(fields)


def tap_path(meta_path) -> Path:
    """Where a recording's Tap file is (whether or not it has one)."""
    meta, _data = stems(meta_path)
    return meta.with_name(meta.name[: -len(META_SUFFIX)] + TAP_SUFFIX)


def _write_new(path: Path, raw: bytes, made: list) -> None:
    with open(path, "xb") as f:
        made.append(path)
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def write(meta_path: Path, data_path: Path, meta: dict, steps: np.ndarray,
          tap: np.ndarray | None = None) -> None:
    """Write the files, creating them exclusively: an existing recording is
    never overwritten. The data files first, the meta file last, so a meta
    file is only ever beside complete data. On failure none is left.
    ``tap``: the Tap's records, written to ``<stem>.athrec-tap`` and
    described in ``meta["tap"]``."""
    from ..util import dumps
    raw = steps.tobytes()
    meta["global"]["athrec:data_bytes"] = len(raw)
    meta["global"]["athrec:data_sha256"] = hashlib.sha256(raw).hexdigest()
    tap_raw = None
    if tap is not None and isinstance(meta.get("tap"), dict):
        tap_raw = tap.tobytes()
        meta["tap"]["file"] = tap_path(meta_path).name
        meta["tap"]["bytes"] = len(tap_raw)
        meta["tap"]["sha256"] = hashlib.sha256(tap_raw).hexdigest()
    text = dumps(meta, indent=1)
    made = []
    try:
        _write_new(data_path, raw, made)
        if tap_raw is not None:
            _write_new(tap_path(meta_path), tap_raw, made)
        with open(meta_path, "x", encoding="utf-8", newline="\n") as f:
            made.append(meta_path)
            f.write(text + "\n")
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        for p in made:
            try:
                p.unlink()
            except OSError:
                pass
        raise


def read_meta(meta_path: Path) -> dict:
    try:
        with open(meta_path, encoding="utf-8-sig") as f:
            meta = json.load(f)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise RecordingError(f"{meta_path.name}: not a readable recording ({exc})") from exc
    g = meta.get("global") if isinstance(meta, dict) else None
    if not isinstance(g, dict) or "athrec:version" not in g:
        raise RecordingError(f"{meta_path.name}: not an Athanor recording (no athrec:version)")
    version = g["athrec:version"]
    if version != FORMAT_VERSION:
        raise RecordingError(f"{meta_path.name}: recording format {version}; this Athanor "
                             f"reads format {FORMAT_VERSION}")
    return meta


def read_steps(meta: dict, data_path: Path, *, verify: bool = False) -> np.ndarray:
    g = meta["global"]
    k = int(g["athrec:k"])
    dt = step_dtype(k)
    n = int(g["athrec:n_steps"])
    size = data_path.stat().st_size
    if size != n * dt.itemsize:
        raise RecordingError(f"{data_path.name}: {size} bytes; {n} steps of {dt.itemsize} "
                             f"bytes need {n * dt.itemsize} — the file is damaged or incomplete")
    if verify and g.get("athrec:data_sha256"):
        h = hashlib.sha256()
        with open(data_path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != g["athrec:data_sha256"]:
            raise RecordingError(f"{data_path.name}: its SHA-256 does not match the meta file")
    if n == 0:
        return np.zeros(0, dt)
    return np.fromfile(data_path, dtype=dt, count=n)


def read_tap(meta: dict, meta_path: Path, *, verify: bool = False) -> np.ndarray | None:
    """The Tap's records (one per step), or None if the recording has none.
    Raises RecordingError if the meta file promises a Tap file that is
    missing or the wrong size."""
    t = meta.get("tap")
    if not isinstance(t, dict) or not t.get("layout") or not t.get("file"):
        return None
    path = Path(meta_path).with_name(Path(t["file"]).name)
    dt = layout_dtype(t["layout"])
    if dt.itemsize != int(t.get("record_bytes") or 0):
        raise RecordingError(f"{path.name}: the meta file's tap layout ({dt.itemsize} bytes) "
                             f"does not match its record size ({t.get('record_bytes')})")
    n = int(t.get("n_steps") or 0)
    try:
        size = path.stat().st_size
    except OSError:
        raise RecordingError(f"{meta_path.name}: its Tap file {path.name} is missing") from None
    if size != n * dt.itemsize:
        raise RecordingError(f"{path.name}: {size} bytes; {n} records of {dt.itemsize} bytes "
                             f"need {n * dt.itemsize} — the file is damaged or incomplete")
    if verify and t.get("sha256"):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != t["sha256"]:
            raise RecordingError(f"{path.name}: its SHA-256 does not match the meta file")
    if n == 0:
        return np.zeros(0, dt)
    return np.fromfile(path, dtype=dt, count=n)
