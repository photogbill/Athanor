"""Writing GGUF files — new files only, never over a model.

Two jobs:

* ``serialize_header`` turns a parsed ``GGUFFile`` (or one being built) back
  into bytes. A header read and serialized again must equal the original
  byte for byte (spike S5); ``round_trip_check`` proves it on any file
  without writing anything.
* ``GGUFWriter`` builds a new file from keys and tensors, laying the data
  out as llama.cpp's own writer does: the data section starts at the next
  multiple of the alignment, and every tensor — the last one included — is
  padded with zeros to the alignment.

Principle 2 of the plan: nothing here opens an existing path for writing.
``write_file`` refuses to overwrite and writes through ``<name>.part``.
"""

from __future__ import annotations

import hashlib
import io
import os
import struct
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable

import numpy as np

from .constants import (
    DEFAULT_ALIGNMENT,
    MAGIC,
    SCALAR_NUMPY,
    SCALAR_STRUCT,
    ValueType,
    tensor_nbytes,
)
from .read import KV, Array, GGUFError, GGUFFile, TensorInfo


def _align(x: int, a: int) -> int:
    return x + (-x % a)


class _Out:
    def __init__(self, stream: BinaryIO, endian: str):
        self.s = stream
        self.e = endian
        self.n = 0

    def raw(self, b: bytes):
        self.s.write(b)
        self.n += len(b)

    def u32(self, v: int):
        self.raw(struct.pack(self.e + "I", v))

    def u64(self, v: int):
        self.raw(struct.pack(self.e + "Q", v))

    def string(self, s: str):
        b = s.encode("utf-8", "surrogateescape")
        self.u64(len(b))
        self.raw(b)

    def value(self, vtype: ValueType, v: Any):
        if vtype == ValueType.STRING:
            self.string(v)
            return
        if vtype == ValueType.ARRAY:
            if not isinstance(v, Array):
                raise TypeError("an ARRAY value must be an Array")
            self.u32(int(v.elem_type))
            self.u64(len(v.items))
            if v.elem_type == ValueType.STRING:
                for s in v.items:
                    self.string(s)
            elif v.elem_type == ValueType.ARRAY:
                for a in v.items:
                    self.value(ValueType.ARRAY, a)
            else:
                dt = np.dtype(SCALAR_NUMPY[v.elem_type]).newbyteorder(self.e)
                self.raw(np.asarray(v.items, dtype=dt).tobytes())
            return
        dt = np.dtype(SCALAR_NUMPY[vtype]).newbyteorder(self.e)
        self.raw(np.asarray(v, dtype=dt).reshape(()).tobytes())


def serialize_header(version: int, endian: str, kvs: Iterable[KV],
                     tensors: Iterable[TensorInfo]) -> bytes:
    """Magic through the end of the tensor infos (no padding)."""
    kvs = list(kvs)
    tensors = list(tensors)
    buf = io.BytesIO()
    o = _Out(buf, endian)
    o.raw(MAGIC)
    o.u32(version)
    o.u64(len(tensors))
    o.u64(len(kvs))
    for kv in kvs:
        o.string(kv.key)
        o.u32(int(kv.type))
        o.value(kv.type, kv.value)
    for t in tensors:
        o.string(t.name)
        n_dims = t.n_dims_written if t.n_dims_written else len(t.shape)
        o.u32(n_dims)
        for d in list(t.shape)[:n_dims]:
            o.u64(int(d))
        o.u32(int(t.ggml_type))
        o.u64(int(t.offset))
    return buf.getvalue()


def round_trip_check(g: GGUFFile) -> dict:
    """Serialize the parsed header and compare it with the file's own bytes.

    Nothing is written. Returns ``{"identical": bool, "first_difference":
    int|None, "header_bytes": int}``.
    """
    ours = serialize_header(g.version, g.endian, g.kvs, g.tensors)
    theirs = g.header_bytes()
    first = None
    if ours != theirs:
        first = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b),
                     min(len(ours), len(theirs)))
    return {"identical": ours == theirs, "first_difference": first,
            "header_bytes": len(theirs)}


def copy_stream_hash(g: GGUFFile, *, chunk: int = 8 << 20) -> tuple:
    """(sha256 of the original file, sha256 of header-rewritten + data copy).

    The second hash is of what a round-trip write WOULD produce — our
    serialized header followed by the original bytes from the end of the
    header on — so a whole model can be checked without a second copy on
    disk.
    """
    orig = hashlib.sha256()
    ours = hashlib.sha256()
    ours.update(serialize_header(g.version, g.endian, g.kvs, g.tensors))
    with open(g.path, "rb") as f:
        head = f.read(g.header_end)
        orig.update(head)
        for block in iter(lambda: f.read(chunk), b""):
            orig.update(block)
            ours.update(block)
    return orig.hexdigest(), ours.hexdigest()


# ------------------------------------------------------------- new files
def _vtype_for(value: Any) -> ValueType:
    if isinstance(value, bool):
        return ValueType.BOOL
    if isinstance(value, str):
        return ValueType.STRING
    if isinstance(value, Array):
        return ValueType.ARRAY
    raise TypeError("give numbers an explicit GGUF type: GGUFWriter.add(key, value, ValueType.X)")


class GGUFWriter:
    """Build a NEW GGUF. For example::

        w = GGUFWriter()
        w.add("general.architecture", "llama")
        w.add("llama.context_length", 4096, ValueType.UINT32)
        w.add_tensor("token_embd.weight", (8, 4), GGMLType.F32, data)
        w.write_file("new.gguf")
    """

    def __init__(self, *, version: int = 3, endian: str = "<",
                 alignment: int | None = None):
        self.version = version
        self.endian = endian
        self.kvs: list = []
        self._keys: set = set()
        self._tensors: list = []  # (TensorInfo, source)
        self._alignment = DEFAULT_ALIGNMENT
        if alignment:
            self.add("general.alignment", alignment, ValueType.UINT32)

    @property
    def alignment(self) -> int:
        """The layout's alignment — set only through ``general.alignment``
        (constructor or ``add``), so the value written and the value used
        can never differ."""
        return self._alignment

    @alignment.setter
    def alignment(self, value) -> None:
        raise AttributeError("set the alignment with add('general.alignment', n, ValueType.UINT32) "
                             "or GGUFWriter(alignment=n) — the key and the layout must agree")

    @staticmethod
    def _check_alignment(value: Any, vtype: ValueType) -> int:
        # gguf.cpp refuses any other type, and the layout needs a power of two
        if vtype != ValueType.UINT32:
            raise GGUFError(f"general.alignment must be UINT32 (llama.cpp refuses {vtype.name})")
        a = int(np.asarray(value).reshape(()))
        if a <= 0 or a & (a - 1):
            raise GGUFError(f"general.alignment {a} is not a power of two")
        return a

    def add(self, key: str, value: Any, vtype: ValueType | None = None,
            elem_type: ValueType | None = None) -> None:
        if key in self._keys:
            raise KeyError(f"duplicate key {key!r}")
        if vtype is None:
            if isinstance(value, (list, tuple, np.ndarray)) and elem_type is not None:
                vtype = ValueType.ARRAY
            else:
                vtype = _vtype_for(value)
        if vtype == ValueType.ARRAY and not isinstance(value, Array):
            if elem_type is None:
                raise TypeError(f"{key}: give the array's element type")
            if elem_type in (ValueType.STRING, ValueType.ARRAY):
                value = Array(elem_type, list(value))
            else:
                dt = np.dtype(SCALAR_NUMPY[elem_type]).newbyteorder(self.endian)
                value = Array(elem_type, np.asarray(value, dtype=dt))
        if key == "general.alignment":
            # the key IS the layout: the value written and the value used
            # must be the same number, or llama.cpp refuses the file
            self._alignment = self._check_alignment(value, vtype)
        self.kvs.append(KV(key, vtype, value))
        self._keys.add(key)

    def add_tensor(self, name: str, shape, ggml_type: int,
                   data: bytes | np.ndarray | Callable[[], Iterable[bytes]]) -> None:
        """``shape`` in ggml order (ne0 first). ``data`` is the raw bytes, an
        array whose bytes are the data, or a callable yielding chunks."""
        nbytes = tensor_nbytes(ggml_type, shape)
        if nbytes is None:
            raise GGUFError(f"{name}: cannot size type {ggml_type} with shape {shape}")
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data).tobytes()
        if isinstance(data, (bytes, bytearray, memoryview)) and len(data) != nbytes:
            raise GGUFError(f"{name}: {len(data)} bytes given, {nbytes} expected")
        info = TensorInfo(name, tuple(int(d) for d in shape), int(ggml_type), 0, len(shape))
        self._tensors.append((info, data))

    def _layout(self) -> list:
        off = 0
        infos = []
        for info, _src in self._tensors:
            info.offset = off
            off = _align(off + info.nbytes, self.alignment)
            infos.append(info)
        return infos

    def write(self, stream: BinaryIO) -> int:
        infos = self._layout()
        head = serialize_header(self.version, self.endian, self.kvs, infos)
        o = _Out(stream, self.endian)
        o.raw(head)
        if infos:
            o.raw(b"\0" * (_align(o.n, self.alignment) - o.n))
        data_start = o.n
        for info, src in self._tensors:
            assert o.n - data_start == info.offset
            written = 0
            chunks = [src] if isinstance(src, (bytes, bytearray, memoryview)) else src()
            for chunk in chunks:
                o.raw(bytes(chunk))
                written += len(chunk)
            if written != info.nbytes:
                raise GGUFError(f"{info.name}: wrote {written} bytes, expected {info.nbytes}")
            o.raw(b"\0" * (_align(o.n - data_start, self.alignment) - (o.n - data_start)))
        return o.n

    def write_file(self, path: str | os.PathLike) -> Path:
        """Write to a path that does not exist, and never over one that does.

        The data goes to ``<name>.part`` (created exclusively — an existing
        part file is someone else's and is left alone), then the part file
        is given the final name by an operation that FAILS if that name
        exists by then: a hard link on POSIX, a rename on Windows. There is
        no moment at which an existing file can be replaced. A failed write
        removes its part file.
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} exists — Athanor never overwrites a file")
        part = path.with_name(path.name + ".part")
        try:
            f = open(part, "xb")
        except FileExistsError:
            raise FileExistsError(f"{part} exists — another write in progress, or left by one; "
                                  "not touched") from None
        try:
            with f:
                self.write(f)
            _publish(part, path)
        except BaseException:
            try:
                part.unlink()
            except OSError:
                pass
            raise
        return path


def _publish(part: Path, path: Path) -> None:
    """Give ``part`` the name ``path`` only if ``path`` does not exist."""
    if os.name == "nt":
        os.rename(part, path)          # Windows: fails if the target exists
        return
    try:
        os.link(part, path)            # POSIX: fails if the target exists
    except FileExistsError:
        raise FileExistsError(f"{path} appeared while writing — not overwritten") from None
    except OSError:
        # a file system without hard links (FAT, exFAT, some network mounts):
        # create the target EXCLUSIVELY and copy into it — slower, but still
        # never over an existing file
        _copy_exclusive(part, path)
        os.unlink(part)
        return
    os.unlink(part)


def _copy_exclusive(src: Path, dst: Path, chunk: int = 8 << 20) -> None:
    try:
        out = open(dst, "xb")
    except FileExistsError:
        raise FileExistsError(f"{dst} appeared while writing — not overwritten") from None
    try:
        with out, open(src, "rb") as inp:
            for block in iter(lambda: inp.read(chunk), b""):
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
    except BaseException:
        try:
            dst.unlink()        # ours: created by the open above
        except OSError:
            pass
        raise
