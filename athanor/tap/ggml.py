"""ggml from Python: just enough to copy a tensor out while llama.cpp computes it.

llama-cpp-python loads ggml but binds none of its functions. The Tap needs
five: a tensor's name, its size, a copy of its data (from whichever device
holds it — ``ggml_backend_tensor_get`` reads CPU, CUDA, Vulkan or Metal
memory alike), and ``ggml_init`` / ``ggml_new_tensor_4d`` / ``ggml_free``
for a self-test. The library is the binding's own (``libggml-base`` beside
``libllama``), so it is the one llama.cpp is already running.

A tensor's shape and strides are read straight from its C struct
(``struct ggml_tensor`` in ggml.h: ``type``, ``buffer``, ``ne[4]``,
``nb[4]`` at its start). That layout is checked, not assumed: ``Ggml()``
builds a 4-D tensor of known shape and reads it back, and refuses to run if
anything differs — a newer ggml that moved a field turns the Tap off with a
reason instead of reading the wrong bytes.
"""

from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path

import numpy as np

GGML_MAX_DIMS = 4

# ggml_type (ggml.h, b11093) -> (numpy dtype, bytes per element). Only the
# plain types: activations are never quantized.
F32, F16, I8, I16, I32, I64, F64, BF16 = 0, 1, 24, 25, 26, 27, 28, 30
TYPES = {F32: (np.float32, 4), F16: (np.float16, 2), I8: (np.int8, 1),
         I16: (np.int16, 2), I32: (np.int32, 4), I64: (np.int64, 8),
         F64: (np.float64, 8), BF16: (np.uint16, 2)}
TYPE_NAMES = {F32: "f32", F16: "f16", I8: "i8", I16: "i16", I32: "i32", I64: "i64",
              F64: "f64", BF16: "bf16"}


class TapUnavailable(RuntimeError):
    """The Tap cannot run with this binding or this model; the message says why."""


class _Tensor(ctypes.Structure):
    """``struct ggml_tensor`` up to its name (ggml.h, b11093) — every field
    the Tap reads is checked by the self-test."""


_Tensor._fields_ = [("type", ctypes.c_int),
                    ("buffer", ctypes.c_void_p),
                    ("ne", ctypes.c_int64 * GGML_MAX_DIMS),
                    ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
                    ("op", ctypes.c_int),
                    ("op_params", ctypes.c_int32 * 16),
                    ("flags", ctypes.c_int32),
                    ("src", ctypes.c_void_p * 10),
                    ("view_src", ctypes.POINTER(_Tensor)),
                    ("view_offs", ctypes.c_size_t),
                    ("data", ctypes.c_void_p),
                    ("name", ctypes.c_char * 64)]
NAME_OFFSET = _Tensor.name.offset          # 256 on 64-bit builds


class _InitParams(ctypes.Structure):
    _fields_ = [("mem_size", ctypes.c_size_t),
                ("mem_buffer", ctypes.c_void_p),
                ("no_alloc", ctypes.c_bool)]


def _library_names() -> list:
    if sys.platform == "win32":
        return ["ggml-base.dll", "libggml-base.dll", "ggml.dll", "libggml.dll"]
    if sys.platform == "darwin":
        return ["libggml-base.dylib", "libggml-base.so", "libggml.dylib", "libggml.so"]
    return ["libggml-base.so", "libggml.so"]


def _load():
    """The ggml the binding is running: its own libllama when ggml is linked
    into it, else ggml-base from the binding's library folder."""
    try:
        from llama_cpp import llama_cpp as lcc
    except Exception as exc:
        raise TapUnavailable(f"llama-cpp-python is not importable ({type(exc).__name__}: {exc})")
    tried = []
    own = getattr(lcc, "_lib", None)
    if own is not None and hasattr(own, "ggml_backend_tensor_get"):
        return own, "libllama (ggml linked in)"
    base = getattr(lcc, "_base_path", None)
    folders = [Path(base)] if base else []
    folders.append(Path(lcc.__file__).parent / "lib")
    for folder in folders:
        for name in _library_names():
            p = folder / name
            if not p.is_file():
                continue
            try:
                lib = ctypes.CDLL(str(p))
            except OSError as exc:
                tried.append(f"{p}: {exc}")
                continue
            if hasattr(lib, "ggml_backend_tensor_get"):
                return lib, str(p)
            tried.append(f"{p}: no ggml_backend_tensor_get")
    raise TapUnavailable("ggml's library was not found beside llama-cpp-python"
                         + (" (" + "; ".join(tried) + ")" if tried else ""))


class Ggml:
    """The bound functions, and a layout-checked view of a tensor."""

    def __init__(self):
        lib, where = _load()
        self.where = where
        f = lib.ggml_get_name
        f.argtypes, f.restype = [ctypes.c_void_p], ctypes.c_void_p
        self._get_name = f
        f = lib.ggml_nbytes
        f.argtypes, f.restype = [ctypes.c_void_p], ctypes.c_size_t
        self.nbytes = f
        f = lib.ggml_backend_tensor_get
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        f.restype = None
        self._get = f
        self._lib = lib
        self._self_test()

    def _self_test(self) -> None:
        """Build a tensor of known shape, name and data, and read it back
        through the struct. Raises TapUnavailable on any difference."""
        lib = self._lib
        try:
            init, new4, setn, free, get_data = (lib.ggml_init, lib.ggml_new_tensor_4d,
                                                lib.ggml_set_name, lib.ggml_free,
                                                lib.ggml_get_data)
        except AttributeError as exc:
            raise TapUnavailable(f"ggml lacks a function the self-test needs ({exc})")
        init.argtypes, init.restype = [_InitParams], ctypes.c_void_p
        new4.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_int64] * 4
        new4.restype = ctypes.c_void_p
        setn.argtypes, setn.restype = [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_void_p
        free.argtypes, free.restype = [ctypes.c_void_p], None
        get_data.argtypes, get_data.restype = [ctypes.c_void_p], ctypes.c_void_p
        ctx = init(_InitParams(1 << 16, None, False))
        if not ctx:
            raise TapUnavailable("ggml_init failed in the Tap's self-test")
        try:
            t = new4(ctx, F32, 2, 3, 5, 7)
            if not t:
                raise TapUnavailable("ggml_new_tensor_4d failed in the Tap's self-test")
            setn(t, b"athanor-self-test")
            h = _Tensor.from_address(t)
            got = (h.type, tuple(h.ne), tuple(h.nb), int(self.nbytes(t)),
                   int(self._get_name(t)) - int(t), h.name, h.data == get_data(t),
                   bool(h.view_src))
            want = (F32, (2, 3, 5, 7), (4, 8, 24, 120), 840, NAME_OFFSET,
                    b"athanor-self-test", True, False)
            if got != want or not h.data:
                raise TapUnavailable(
                    "this ggml's tensor layout is not the one Athanor reads "
                    f"(read {got}, expected {want}); the Tap is off rather than "
                    "reading the wrong bytes")
        finally:
            free(ctx)

    # ------------------------------------------------------------ reading
    @staticmethod
    def name(t) -> str:
        return ctypes.string_at(t + NAME_OFFSET).decode("utf-8", "replace")

    @staticmethod
    def head(t) -> tuple:
        """(type, ne, nb) of the tensor at address ``t``."""
        h = _Tensor.from_address(t)
        return int(h.type), tuple(h.ne), tuple(h.nb)

    @staticmethod
    def readable(t) -> bool:
        """Has llama.cpp put this tensor's data somewhere ggml can copy it
        from? (``ggml_backend_tensor_get`` ends the process if not.)"""
        h = _Tensor.from_address(t)
        if not h.data:
            return False
        if h.view_src:
            return bool(h.view_src.contents.buffer)
        return bool(h.buffer)

    def read(self, t, axis: int, rows) -> np.ndarray:
        """Rows ``rows`` (indices along ggml axis ``axis``) of tensor ``t``,
        each flattened: an array [len(rows), elements per row], float32 for
        float types and int32 for integer ones. Copies only the bytes those
        rows span. Raises ValueError for a type it does not read, a row out
        of range, or a tensor with no data."""
        typ, ne, nb = self.head(t)
        spec = TYPES.get(typ)
        if spec is None:
            raise ValueError(f"tensor type {typ} is not read by the Tap")
        if not self.readable(t):
            raise ValueError("the tensor has no data yet")
        dt, size = spec
        other = [i for i in range(GGML_MAX_DIMS) if i != axis]
        shape = [int(ne[i]) for i in reversed(other)]
        strides = [int(nb[i]) for i in reversed(other)]
        per_row = math.prod(shape)
        rows = [int(r) for r in rows]
        n = int(ne[axis])
        if any(not 0 <= r < n for r in rows):
            raise ValueError(f"row outside 0–{n - 1}")
        out = np.empty((len(rows), per_row), dtype=dt)
        if not rows or per_row == 0:
            return _plain(out, typ)
        span = size + sum((int(ne[i]) - 1) * int(nb[i]) for i in other)
        total = int(self.nbytes(t))
        step = int(nb[axis])
        if (n - 1) * step + span > total:
            raise ValueError("the tensor's strides do not fit its size")
        if len(rows) * 2 >= n:                      # most of it: one copy
            buf = np.empty(total, np.uint8)
            self._get(t, buf.ctypes.data, 0, total)
            for j, r in enumerate(rows):
                out[j] = np.ndarray(shape, dt, buffer=buf, offset=r * step,
                                    strides=strides).reshape(-1)
        else:                                       # a few rows: just those bytes
            buf = np.empty(span, np.uint8)
            for j, r in enumerate(rows):
                self._get(t, buf.ctypes.data, r * step, span)
                out[j] = np.ndarray(shape, dt, buffer=buf, strides=strides).reshape(-1)
        return _plain(out, typ)


def _plain(a: np.ndarray, typ: int) -> np.ndarray:
    if typ == BF16:
        return (a.astype(np.uint32) << 16).view(np.float32)
    if a.dtype.kind == "f":
        return a.astype(np.float32, copy=False)
    return a.astype(np.int32, copy=False)


_GGML: Ggml | None = None
_ERROR: str | None = None


def ggml() -> Ggml:
    """The process's one Ggml (loaded and self-tested once)."""
    global _GGML, _ERROR
    if _GGML is not None:
        return _GGML
    if _ERROR is not None:
        raise TapUnavailable(_ERROR)
    try:
        _GGML = Ggml()
    except TapUnavailable as exc:
        _ERROR = str(exc)
        raise
    return _GGML
