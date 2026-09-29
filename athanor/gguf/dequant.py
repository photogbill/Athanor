"""Turning stored rows back into numbers — for comparisons, not inference.

F32, F16, BF16, Q8_0, Q4_0 and Q4_1 are decoded here with numpy (their
layouts are short and fixed). Every other type is handed to gguf-py's
``gguf.quants.dequantize`` (MIT, llama.cpp's own Python package) when it is
installed; if it is not, ``dequantize_rows`` says so rather than guessing.
"""

from __future__ import annotations

import numpy as np

from .constants import GGMLType


class NoDequantizer(RuntimeError):
    pass


def _q8_0(raw: bytes, n: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)[:n]


def _q4_0(raw: bytes, n: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    qs = blocks[:, 2:]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = (qs >> 4).astype(np.int8) - 8
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (q * d).reshape(-1)[:n]


def _q4_1(raw: bytes, n: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 20)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    m = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    qs = blocks[:, 4:]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    q = np.concatenate([lo, hi], axis=1)
    return (q * d + m).reshape(-1)[:n]


def dequantize_row(raw: bytes, ggml_type: int, n: int) -> np.ndarray:
    """One row of ``n`` values."""
    t = int(ggml_type)
    if t == GGMLType.F32:
        return np.frombuffer(raw, dtype="<f4")[:n].astype(np.float32)
    if t == GGMLType.F16:
        return np.frombuffer(raw, dtype="<f2")[:n].astype(np.float32)
    if t == GGMLType.BF16:
        u = np.frombuffer(raw, dtype="<u2")[:n].astype(np.uint32) << 16
        return u.view(np.float32)
    if t == GGMLType.Q8_0:
        return _q8_0(raw, n)
    if t == GGMLType.Q4_0:
        return _q4_0(raw, n)
    if t == GGMLType.Q4_1:
        return _q4_1(raw, n)
    try:
        from gguf import quants  # gguf-py, optional
        from gguf.constants import GGMLQuantizationType
    except Exception as exc:
        raise NoDequantizer(
            f"decoding {GGMLType(t).name if t in GGMLType._value2member_map_ else t} rows "
            "needs gguf-py (pip install gguf)") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    return quants.dequantize(arr, GGMLQuantizationType(t)).reshape(-1)[:n].astype(np.float32)


def dequantize_rows(g, tensor, rows) -> np.ndarray:
    """Rows of a 2-D tensor as a float32 matrix (len(rows) × ne0)."""
    t = g.tensor(tensor) if isinstance(tensor, str) else tensor
    if getattr(g, "endian", "<") != "<":
        raise NoDequantizer("rows of a big-endian GGUF are not decoded (its header is read; "
                            "its tensor data layout is not supported here)")
    ne0 = int(t.shape[0])
    raws = g.row_bytes(t, rows)
    return np.stack([dequantize_row(r, t.ggml_type, ne0) for r in raws])
