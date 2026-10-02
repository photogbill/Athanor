"""Turning stored rows back into numbers — for comparisons and the lens, not inference.

F32, F16, BF16, the Q4/Q5/Q8 block types and the K-quants Q4_K, Q5_K and
Q6_K are decoded here with numpy, following ggml's own ``dequantize_row_*``
(``ggml-quants.c``, b11093) layout for layout — Q6_K is what every K-quant
file keeps its output matrix in, and the logit lens needs that matrix whole.
Every other type is handed to gguf-py's ``gguf.quants.dequantize`` (MIT,
llama.cpp's own Python package) when it is installed; if it is not,
``dequantize_rows`` says so rather than guessing.
"""

from __future__ import annotations

import numpy as np

from .constants import GGMLType

QK_K = 256


class NoDequantizer(RuntimeError):
    pass


def _f16(b: np.ndarray) -> np.ndarray:
    """Little-endian float16 pairs of bytes → float32."""
    return b.copy().view("<f2").astype(np.float32)


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


def _q5_0(raw: bytes, n: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 22)
    d = _f16(blocks[:, :2])
    qh = blocks[:, 2:6].copy().view("<u4").astype(np.uint32)            # [blocks, 1]
    qs = blocks[:, 6:]
    j = np.arange(16, dtype=np.uint32)
    xh_0 = ((qh >> j) << 4) & 0x10                                     # bit j
    xh_1 = (qh >> (j + 12)) & 0x10                                     # bit j+16, in place
    lo = ((qs & 0x0F) | xh_0).astype(np.int16) - 16
    hi = ((qs >> 4) | xh_1).astype(np.int16) - 16
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (q * d).reshape(-1)[:n]


def _q5_1(raw: bytes, n: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 24)
    d = _f16(blocks[:, :2])
    m = _f16(blocks[:, 2:4])
    qh = blocks[:, 4:8].copy().view("<u4").astype(np.uint32)
    qs = blocks[:, 8:]
    j = np.arange(16, dtype=np.uint32)
    xh_0 = ((qh >> j) << 4) & 0x10
    xh_1 = (qh >> (j + 12)) & 0x10
    lo = ((qs & 0x0F) | xh_0).astype(np.float32)
    hi = ((qs >> 4) | xh_1).astype(np.float32)
    q = np.concatenate([lo, hi], axis=1)
    return (q * d + m).reshape(-1)[:n]


def _scale_min_k4(sc: np.ndarray) -> tuple:
    """ggml's ``get_scale_min_k4``: the 8 six-bit scales and 8 six-bit mins
    packed in a K-quant's 12 scale bytes. ``sc`` is [blocks, 12] uint8."""
    d_lo = sc[:, 0:4] & 63
    m_lo = sc[:, 4:8] & 63
    d_hi = (sc[:, 8:12] & 0x0F) | ((sc[:, 0:4] >> 6) << 4)
    m_hi = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
    return (np.concatenate([d_lo, d_hi], axis=1).astype(np.float32),
            np.concatenate([m_lo, m_hi], axis=1).astype(np.float32))


def _q4_k(raw: bytes, n: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 144)
    nb = blocks.shape[0]
    d = _f16(blocks[:, 0:2])                                            # [nb, 1]
    dmin = _f16(blocks[:, 2:4])
    scales, mins = _scale_min_k4(blocks[:, 4:16])                      # [nb, 8]
    qs = blocks[:, 16:].reshape(nb, 4, 1, 32)
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=2).reshape(nb, 8, 32).astype(np.float32)
    y = (d * scales)[:, :, None] * q - (dmin * mins)[:, :, None]
    return y.reshape(-1)[:n]


def _q5_k(raw: bytes, n: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 176)
    nb = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    dmin = _f16(blocks[:, 2:4])
    scales, mins = _scale_min_k4(blocks[:, 4:16])
    qh = blocks[:, 16:48].reshape(nb, 1, 32)
    qs = blocks[:, 48:].reshape(nb, 4, 1, 32)
    nib = np.concatenate([qs & 0x0F, qs >> 4], axis=2).reshape(nb, 8, 32)
    high = (qh >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1    # bit s for sub-block s
    q = nib.astype(np.float32) + 16.0 * high.astype(np.float32)
    y = (d * scales)[:, :, None] * q - (dmin * mins)[:, :, None]
    return y.reshape(-1)[:n]


def _q6_k(raw: bytes, n: int) -> np.ndarray:
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 210)
    nb = blocks.shape[0]
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].copy().view(np.int8).astype(np.float32)   # [nb, 16]
    d = _f16(blocks[:, 208:210])                                          # [nb, 1]
    # two halves of 128 values; in each, four runs of 32 take the low and
    # high nibbles of ql[0:32], ql[32:64] and bit pairs 0,2,4,6 of qh[0:32]
    lo_hi = ql.reshape(nb, 2, 1, 64) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)
    lo_hi = (lo_hi & 0x0F).reshape(nb, 8, 32)
    top = qh.reshape(nb, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)
    top = (top & 0x03).reshape(nb, 8, 32)
    q = (lo_hi | (top << 4)).astype(np.int16) - 32                        # [nb, 8, 32]
    q = q.reshape(nb, 16, 16).astype(np.float32)                          # one scale per 16
    y = (d * scales)[:, :, None] * q
    return y.reshape(-1)[:n]


_NATIVE = {
    GGMLType.Q8_0: _q8_0, GGMLType.Q4_0: _q4_0, GGMLType.Q4_1: _q4_1,
    GGMLType.Q5_0: _q5_0, GGMLType.Q5_1: _q5_1,
    GGMLType.Q4_K: _q4_k, GGMLType.Q5_K: _q5_k, GGMLType.Q6_K: _q6_k,
}

#: the types decoded here without gguf-py
NATIVE_TYPES = (GGMLType.F32, GGMLType.F16, GGMLType.BF16) + tuple(_NATIVE)


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
    fn = _NATIVE.get(t)
    if fn is not None:
        return fn(raw, n)
    try:
        from gguf import quants  # gguf-py, optional
        from gguf.constants import GGMLQuantizationType
    except Exception as exc:
        raise NoDequantizer(
            f"decoding {GGMLType(t).name if t in GGMLType._value2member_map_ else t} rows "
            "needs gguf-py (pip install gguf)") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    return quants.dequantize(arr, GGMLQuantizationType(t)).reshape(-1)[:n].astype(np.float32)


def can_dequantize(ggml_type: int) -> str | None:
    """None when rows of this type can be decoded here; otherwise why not."""
    t = int(ggml_type)
    if t in NATIVE_TYPES:
        return None
    try:
        import gguf  # noqa: F401 — optional, used by dequantize_row
        return None
    except Exception:
        name = GGMLType(t).name if t in GGMLType._value2member_map_ else str(t)
        return f"decoding {name} rows needs gguf-py (pip install gguf)"


def dequantize_rows(g, tensor, rows) -> np.ndarray:
    """Rows of a 2-D tensor as a float32 matrix (len(rows) × ne0)."""
    t = g.tensor(tensor) if isinstance(tensor, str) else tensor
    if getattr(g, "endian", "<") != "<":
        raise NoDequantizer("rows of a big-endian GGUF are not decoded (its header is read; "
                            "its tensor data layout is not supported here)")
    ne0 = int(t.shape[0])
    raws = g.row_bytes(t, rows)
    return np.stack([dequantize_row(r, t.ggml_type, ne0) for r in raws])


def dequantize_row_range(g, tensor, start: int, stop: int) -> np.ndarray:
    """Rows ``start`` … ``stop-1`` of a 2-D tensor as float32 [stop-start, ne0],
    read in one piece and decoded in one go — how the lens takes an output
    matrix of 150,000 rows out of a file in seconds rather than minutes.
    Rows of a block type are whole blocks (llama.cpp requires it), so a run
    of rows decodes exactly as the rows would one by one."""
    t = g.tensor(tensor) if isinstance(tensor, str) else tensor
    if getattr(g, "endian", "<") != "<":
        raise NoDequantizer("rows of a big-endian GGUF are not decoded")
    rb = t.row_nbytes
    if rb is None:
        raise NoDequantizer(f"{t.name}: row size of type {t.type_name} unknown")
    start, stop = int(start), int(stop)
    if not 0 <= start <= stop <= t.n_rows:
        raise IndexError(f"{t.name}: rows {start}–{stop} of {t.n_rows}")
    ne0 = int(t.shape[0])
    if stop == start:
        return np.zeros((0, ne0), np.float32)
    raw = g.read_bytes(t.abs_offset + start * rb, (stop - start) * rb)
    flat = dequantize_row(raw, t.ggml_type, (stop - start) * ne0)
    return flat.reshape(stop - start, ne0)
