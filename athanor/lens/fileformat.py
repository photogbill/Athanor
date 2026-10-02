"""The lens track beside a recording: ``<stem>.athrec-lens`` + ``<stem>.athrec-lens-meta``.

A recording made with the Tap's residual stream (``--tap residual``) holds
every layer's output for every token. The lens pass reads each of those
through the model's own final norm and output matrix and keeps, per step
and per layer, what that layer "would say": its k likeliest tokens, where
the model's eventual favourite and the token actually chosen rank at that
depth, and the entropy. One fixed-size record per step, aligned with the
recording's steps, little-endian, packed::

    uint32        flags          1 = no residual at this step (the Tap had nothing)
                                 2 = the last layer's lens does not agree with the
                                     recording's favourite at this step
    int32         depth          decision depth: the first layer from which the
                                 favourite is the lens's answer at every layer up
                                 to the top; -1 if the top layer itself disagrees
    int32         first_seen     the first layer at which the favourite is within
                                 the lens's top k; -1 if never
    int32         fav            the recording's favourite (its ids[0])
    int32         chosen_depth   as depth, for the token the sampler chose
    int32[L,k]    ids            each layer's k likeliest tokens, best first
    float32[L,k]  logprobs       their log-probabilities at that layer, nats
    int32[L]      fav_rank       the favourite's rank at each layer (0 = top)
    float32[L]    fav_logprob    its log-probability at each layer
    int32[L]      chosen_rank    the chosen token's rank at each layer
    float32[L]    chosen_logprob
    float32[L]    entropy        of the layer's whole distribution, nats

``L`` layers are the ones the Tap recorded (the meta file's ``layers``, in
record order); ``k`` is the meta file's ``k``. The lens track is a DERIVED
file: computed after the fact from the Tap's rows and the model file, and
recomputable. The recording itself is never changed by it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..waterfall import fileformat as ff

FORMAT = "athlens"
FORMAT_VERSION = 1
LENS_SUFFIX = ".athrec-lens"
LENS_META_SUFFIX = ".athrec-lens-meta"

FLAG_NO_RESIDUAL = 1
FLAG_TOP_DISAGREES = 2
FLAG_MEANINGS = {FLAG_NO_RESIDUAL: "no residual at this step (the Tap had nothing)",
                 FLAG_TOP_DISAGREES: "the last layer's lens does not agree with the recording's "
                                     "favourite at this step"}

LAYOUT = ("flags:uint32 depth:int32 first_seen:int32 fav:int32 chosen_depth:int32 "
          "ids:int32[L,k] logprobs:float32[L,k] fav_rank:int32[L] fav_logprob:float32[L] "
          "chosen_rank:int32[L] chosen_logprob:float32[L] entropy:float32[L]; "
          "little-endian, packed")


def record_dtype(n_layers: int, k: int) -> np.dtype:
    L, k = int(n_layers), int(k)
    if L < 1 or k < 1:
        raise ValueError("a lens record needs at least one layer and one candidate")
    return np.dtype([("flags", "<u4"), ("depth", "<i4"), ("first_seen", "<i4"), ("fav", "<i4"),
                     ("chosen_depth", "<i4"),
                     ("ids", "<i4", (L, k)), ("logprobs", "<f4", (L, k)),
                     ("fav_rank", "<i4", (L,)), ("fav_logprob", "<f4", (L,)),
                     ("chosen_rank", "<i4", (L,)), ("chosen_logprob", "<f4", (L,)),
                     ("entropy", "<f4", (L,))])


def paths(recording_path) -> tuple[Path, Path]:
    """(lens meta, lens data) for a recording named by any of its files."""
    meta, _data = ff.stems(recording_path)
    stem = meta.name[: -len(ff.META_SUFFIX)]
    return meta.with_name(stem + LENS_META_SUFFIX), meta.with_name(stem + LENS_SUFFIX)


def _write(path: Path, raw: bytes | str, replace: bool, made: list) -> None:
    mode = "wb" if isinstance(raw, bytes) else "w"
    kw = {} if isinstance(raw, bytes) else {"encoding": "utf-8", "newline": "\n"}
    if replace:
        tmp = path.with_name(path.name + ".part")
        with open(tmp, mode, **kw) as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        made.append(path)
    else:
        with open(path, mode.replace("w", "x"), **kw) as f:
            made.append(path)
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())


def write(recording_path, meta: dict, records: np.ndarray, *, replace: bool = False) -> Path:
    """Write the lens files beside the recording; returns the lens meta
    path. Created exclusively unless ``replace`` (a lens is derived, so
    redoing it with another k or a calibrated lens is allowed when asked).
    The data file first, the meta file last; on failure nothing is left."""
    from ..util import dumps
    meta_path, data_path = paths(recording_path)
    if not replace and (meta_path.exists() or data_path.exists()):
        raise FileExistsError(f"{meta_path.name} exists; pass --replace to redo the lens")
    raw = records.tobytes()
    meta = dict(meta)
    meta["file"] = data_path.name
    meta["bytes"] = len(raw)
    meta["sha256"] = hashlib.sha256(raw).hexdigest()
    made: list = []
    try:
        _write(data_path, raw, replace, made)
        _write(meta_path, dumps(meta, indent=1) + "\n", replace, made)
    except BaseException:
        for p in made:
            try:
                p.unlink()
            except OSError:
                pass
        raise
    return meta_path


def read_meta(meta_path: Path) -> dict:
    try:
        with open(meta_path, encoding="utf-8-sig") as f:
            meta = json.load(f)
    except (OSError, ValueError) as exc:
        raise ff.RecordingError(f"{meta_path.name}: not a readable lens file ({exc})") from exc
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise ff.RecordingError(f"{meta_path.name}: not an Athanor lens file")
    if meta.get("version") != FORMAT_VERSION:
        raise ff.RecordingError(f"{meta_path.name}: lens format {meta.get('version')}; this "
                                f"Athanor reads format {FORMAT_VERSION}")
    return meta


def read(recording_path, *, verify: bool = False) -> tuple | None:
    """(meta, records) of a recording's lens track, or None when it has
    none. A lens file that is damaged or incomplete raises RecordingError."""
    meta_path, data_path = paths(recording_path)
    if not meta_path.is_file():
        return None
    meta = read_meta(meta_path)
    layers = meta.get("layers") or []
    dt = record_dtype(len(layers), int(meta.get("k") or 0))
    if dt.itemsize != int(meta.get("record_bytes") or 0):
        raise ff.RecordingError(f"{meta_path.name}: its layout ({dt.itemsize} bytes a record) does "
                                f"not match record_bytes ({meta.get('record_bytes')})")
    n = int(meta.get("n_steps") or 0)
    try:
        size = data_path.stat().st_size
    except OSError:
        raise ff.RecordingError(f"{meta_path.name}: its data file {data_path.name} is missing") \
            from None
    if size != n * dt.itemsize:
        raise ff.RecordingError(f"{data_path.name}: {size} bytes; {n} records of {dt.itemsize} "
                                f"bytes need {n * dt.itemsize} — damaged or incomplete")
    if verify and meta.get("sha256"):
        h = hashlib.sha256()
        with open(data_path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != meta["sha256"]:
            raise ff.RecordingError(f"{data_path.name}: its SHA-256 does not match the meta file")
    records = np.fromfile(data_path, dtype=dt, count=n) if n else np.zeros(0, dt)
    return meta, records


__all__ = ["FORMAT", "FORMAT_VERSION", "LENS_SUFFIX", "LENS_META_SUFFIX", "LAYOUT",
           "FLAG_NO_RESIDUAL", "FLAG_TOP_DISAGREES", "FLAG_MEANINGS", "record_dtype", "paths",
           "write", "read", "read_meta"]
