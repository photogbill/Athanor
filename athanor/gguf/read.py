"""A complete GGUF reader.

Everything in the header is read — every key and every array, however long
(a vocabulary is ~130k strings; ATK's own metadata reader stops at 4,096 on
purpose, Athanor does not) — and every tensor's name, shape, type and place.
Tensor DATA is not read until asked for.

Strings are decoded as UTF-8 with ``surrogateescape``, so a key or token that
is not valid UTF-8 survives a read and a write byte for byte.

Layout facts this follows (llama.cpp b11093, ``ggml/src/gguf.cpp``):
little-endian unless the version reads byte-swapped; 64-bit counts and
lengths (versions 2 and 3); the tensor data section starts at the first
multiple of ``general.alignment`` (default 32) after the tensor infos, and
is padded only when the file holds at least one tensor.
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .constants import (
    BLOCK,
    DEFAULT_ALIGNMENT,
    GGMLType,
    MAGIC,
    SCALAR_NUMPY,
    SCALAR_SIZE,
    SCALAR_STRUCT,
    SUPPORTED_VERSIONS,
    ValueType,
    row_nbytes,
    tensor_nbytes,
    type_name,
)


BLOCK_TYPES = {int(t) for t in BLOCK}


class GGUFError(ValueError):
    """The file is not a GGUF this reader can read, and why."""


@dataclass
class Array:
    """A GGUF array value: its element type and its items.

    Numeric items are a numpy array (exact dtype, so they write back
    exactly); strings are a list of str; nested arrays a list of Array.
    """

    elem_type: ValueType
    items: Any

    def __len__(self) -> int:
        return len(self.items)

    def tolist(self) -> list:
        if isinstance(self.items, np.ndarray):
            if self.elem_type == ValueType.BOOL:
                return [bool(x) for x in self.items]
            return self.items.tolist()
        if self.elem_type == ValueType.ARRAY:
            return [a.tolist() for a in self.items]
        return list(self.items)


@dataclass
class KV:
    key: str
    type: ValueType
    value: Any  # numpy scalar, bool-as-uint8 scalar, str, or Array

    def plain(self) -> Any:
        """The value as plain Python (numeric arrays stay numpy)."""
        v = self.value
        if isinstance(v, Array):
            if v.elem_type == ValueType.STRING or v.elem_type == ValueType.ARRAY:
                return v.tolist()
            if v.elem_type == ValueType.BOOL:
                return v.items.astype(bool)
            return v.items
        if self.type == ValueType.BOOL:
            return bool(v)
        if isinstance(v, np.generic):
            return v.item()
        return v


@dataclass
class TensorInfo:
    name: str
    shape: tuple          # ggml order: ne0 (the row length) first
    ggml_type: int
    offset: int           # relative to the data section
    n_dims_written: int   # as stored, for an exact round trip
    data_offset: int = 0  # absolute position of the data section

    @property
    def type_name(self) -> str:
        return type_name(self.ggml_type)

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def nbytes(self) -> int | None:
        return tensor_nbytes(self.ggml_type, self.shape)

    @property
    def abs_offset(self) -> int:
        return self.data_offset + self.offset

    @property
    def n_rows(self) -> int:
        return self.n_elements // int(self.shape[0]) if self.shape else 0

    @property
    def row_nbytes(self) -> int | None:
        return row_nbytes(self.ggml_type, int(self.shape[0])) if self.shape else None


@dataclass
class GGUFFile:
    path: Path
    version: int
    endian: str                    # "<" or ">"
    kvs: list = field(default_factory=list)      # [KV] in file order
    tensors: list = field(default_factory=list)  # [TensorInfo] in file order
    header_end: int = 0            # end of the tensor infos
    data_offset: int = 0           # start of the tensor data section
    file_size: int = 0
    alignment: int = DEFAULT_ALIGNMENT

    # ------------------------------------------------------------- lookups
    def __post_init__(self):
        self._kv_index = {}
        self._tensor_index = {}

    def _reindex(self):
        self._kv_index = {kv.key: kv for kv in self.kvs}
        self._tensor_index = {t.name: t for t in self.tensors}

    def kv(self, key: str) -> KV | None:
        return self._kv_index.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        kv = self._kv_index.get(key)
        return default if kv is None else kv.plain()

    def __contains__(self, key: str) -> bool:
        return key in self._kv_index

    def keys(self) -> list:
        return [kv.key for kv in self.kvs]

    def tensor(self, name: str) -> TensorInfo | None:
        return self._tensor_index.get(name)

    @property
    def architecture(self) -> str | None:
        return self.get("general.architecture")

    def arch_get(self, suffix: str, default: Any = None) -> Any:
        """``<architecture>.<suffix>`` — e.g. ``context_length``."""
        arch = self.architecture
        return default if not arch else self.get(f"{arch}.{suffix}", default)

    # ---------------------------------------------------------- data access
    def read_bytes(self, start: int, n: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(start)
            data = f.read(n)
        if len(data) != n:
            raise GGUFError(f"{self.path.name}: file ends {n - len(data)} bytes early")
        return data

    def tensor_bytes(self, t: TensorInfo | str) -> bytes:
        t = self.tensor(t) if isinstance(t, str) else t
        if t is None:
            raise KeyError("no such tensor")
        if t.nbytes is None:
            raise GGUFError(f"{t.name}: size of type {t.type_name} unknown to this reader")
        return self.read_bytes(t.abs_offset, t.nbytes)

    def row_bytes(self, t: TensorInfo | str, rows) -> list:
        """Raw bytes of the given rows (e.g. token ids of token_embd)."""
        t = self.tensor(t) if isinstance(t, str) else t
        rb = t.row_nbytes
        if rb is None:
            raise GGUFError(f"{t.name}: row size of type {t.type_name} unknown")
        out = []
        with open(self.path, "rb") as f:
            for r in rows:
                if not 0 <= r < t.n_rows:
                    raise IndexError(f"{t.name}: row {r} of {t.n_rows}")
                f.seek(t.abs_offset + r * rb)
                out.append(f.read(rb))
        return out

    def header_bytes(self) -> bytes:
        """The header through the end of the tensor infos."""
        return self.read_bytes(0, self.header_end)

    def header_sha256(self) -> str:
        """Identifies the metadata, tokenizer and tensor layout — cheap."""
        return hashlib.sha256(self.header_bytes()).hexdigest()

    def file_sha256(self, chunk: int = 8 << 20) -> str:
        """The whole file's SHA-256. Slow on a large model (it reads it all)."""
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "file_size": self.file_size,
            "version": self.version,
            "endian": "little" if self.endian == "<" else "big",
            "alignment": self.alignment,
            "n_kv": len(self.kvs),
            "n_tensors": len(self.tensors),
            "header_bytes": self.header_end,
            "data_offset": self.data_offset,
        }


# --------------------------------------------------------------- parsing
class _NeedMore(Exception):
    pass


class _Parser:
    def __init__(self, buf: bytes, endian: str, file_size: int):
        self.buf = buf
        self.pos = 0
        self.e = endian
        self.file_size = file_size

    def take(self, n: int) -> bytes:
        end = self.pos + n
        if end > self.file_size:
            # a damaged length: say so now, rather than reading the whole
            # file into memory looking for bytes that are not there
            raise GGUFError(f"a field claims {n:,} bytes at offset {self.pos:,}, past the end "
                            "of the file — damaged or truncated")
        if end > len(self.buf):
            raise _NeedMore
        b = self.buf[self.pos:end]
        self.pos = end
        return b

    def scalar(self, fmt: str):
        size = struct.calcsize(fmt)
        if self.pos + size > self.file_size:
            raise GGUFError("the file ends inside its header — truncated")
        if self.pos + size > len(self.buf):
            raise _NeedMore
        v = struct.unpack_from(self.e + fmt, self.buf, self.pos)[0]
        self.pos += size
        return v

    def u32(self) -> int:
        return self.scalar("I")

    def u64(self) -> int:
        return self.scalar("Q")

    def string(self) -> str:
        n = self.u64()
        if n > (1 << 30):
            raise GGUFError(f"string length {n} is not plausible — file damaged?")
        return self.take(n).decode("utf-8", "surrogateescape")

    def value(self, vtype: ValueType):
        if vtype == ValueType.STRING:
            return self.string()
        if vtype == ValueType.ARRAY:
            code = self.u32()
            try:
                elem = ValueType(code)
            except ValueError:
                raise GGUFError(f"an array has element type {code}, which GGUF does not define "
                                "— damaged?") from None
            n = self.u64()
            if n > (1 << 30):
                raise GGUFError(f"array length {n} is not plausible — file damaged?")
            if elem == ValueType.STRING:
                return Array(elem, [self.string() for _ in range(n)])
            if elem == ValueType.ARRAY:
                return Array(elem, [self.value(ValueType.ARRAY) for _ in range(n)])
            size = SCALAR_SIZE[elem]
            raw = self.take(n * size)
            dt = np.dtype(SCALAR_NUMPY[elem]).newbyteorder(self.e)
            return Array(elem, np.frombuffer(raw, dtype=dt).copy())
        size = SCALAR_SIZE[vtype]
        raw = self.take(size)
        dt = np.dtype(SCALAR_NUMPY[vtype]).newbyteorder(self.e)
        return np.frombuffer(raw, dtype=dt)[0]


def _align(x: int, a: int) -> int:
    return x + (-x % a)


def _parse(buf: bytes, path: Path, file_size: int) -> GGUFFile:
    if file_size < 24:
        raise GGUFError(f"{path.name}: {file_size} bytes is too short to be a GGUF")
    if buf[:4] != MAGIC:
        raise GGUFError(f"{path.name}: not a GGUF file (magic {buf[:4]!r})")
    version = struct.unpack_from("<I", buf, 4)[0]
    endian = "<"
    if version & 0xFFFF == 0 and version != 0:
        endian = ">"
        version = struct.unpack_from(">I", buf, 4)[0]
    if version == 1:
        raise GGUFError(f"{path.name}: GGUF version 1 (32-bit counts) is not supported")
    if version not in SUPPORTED_VERSIONS:
        raise GGUFError(f"{path.name}: GGUF version {version} is not one this reader knows")
    p = _Parser(buf, endian, file_size)
    p.pos = 8
    n_tensors = p.u64()
    n_kv = p.u64()
    g = GGUFFile(path=path, version=version, endian=endian, file_size=file_size)
    for _ in range(n_kv):
        key = p.string()
        try:
            vtype = ValueType(p.u32())
        except ValueError as exc:
            raise GGUFError(f"{path.name}: key {key!r} has an unknown value type") from exc
        g.kvs.append(KV(key, vtype, p.value(vtype)))
    for _ in range(n_tensors):
        name = p.string()
        n_dims = p.u32()
        if n_dims > 4:
            raise GGUFError(f"{path.name}: tensor {name!r} has {n_dims} dimensions")
        shape = tuple(p.u64() for _ in range(n_dims))
        gtype = p.u32()
        offset = p.u64()
        g.tensors.append(TensorInfo(name, shape, gtype, offset, n_dims))
    g.header_end = p.pos
    g._reindex()
    akv = g.kv("general.alignment")
    if akv is not None and akv.type != ValueType.UINT32:
        # gguf.cpp refuses the file in this case; so do we
        raise GGUFError(f"{path.name}: general.alignment is {akv.type.name}, llama.cpp "
                        "requires UINT32")
    align = akv.plain() if akv is not None else DEFAULT_ALIGNMENT
    if align <= 0 or align & (align - 1):
        raise GGUFError(f"{path.name}: general.alignment {align!r} is not a power of two")
    g.alignment = align
    g.data_offset = _align(g.header_end, align) if g.tensors else g.header_end
    for t in g.tensors:
        t.data_offset = g.data_offset
    return g


def read(path: str | os.PathLike, *, first_chunk: int = 32 << 20) -> GGUFFile:
    """Read a GGUF's whole header. Tensor data stays on disk."""
    path = Path(path)
    size = path.stat().st_size
    n = min(first_chunk, size)
    with open(path, "rb") as f:
        while True:
            f.seek(0)
            buf = f.read(n)
            try:
                return _parse(buf, path, size)
            except _NeedMore:
                if n >= size:
                    raise GGUFError(f"{path.name}: file ends inside its header") from None
                n = min(n * 4, size)


def iter_ggufs(folder: str | os.PathLike) -> Iterator[Path]:
    """Every ``*.gguf`` under a folder, sorted, without following links."""
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(".gguf"):
                yield Path(root) / name


def validate(g: GGUFFile) -> list:
    """Structural problems with the tensor layout, as plain sentences."""
    problems = []
    names = set()
    expected = 0  # gguf.cpp: each tensor starts where the previous one's padding ends
    for t in g.tensors:
        if t.name in names:
            problems.append(f"tensor name {t.name!r} appears twice")
        names.add(t.name)
        if t.offset % g.alignment:
            problems.append(f"{t.name}: offset {t.offset} is not aligned to {g.alignment}")
        nb = t.nbytes
        if nb is None:
            if t.ggml_type in BLOCK_TYPES:
                per = BLOCK[GGMLType(t.ggml_type)][0]
                problems.append(f"{t.name}: row length {t.shape[0] if t.shape else 0} is not a "
                                f"whole number of {t.type_name} blocks ({per})")
            else:
                problems.append(f"{t.name}: type {t.type_name} unknown to llama.cpp b11093 "
                                "(size unchecked)")
            expected = None
            continue
        if expected is not None and t.offset != expected:
            problems.append(f"{t.name}: offset {t.offset}, but llama.cpp expects {expected} "
                            "(tensors must be packed in order) — llama.cpp will refuse the file")
        if expected is not None:
            expected = _align(t.offset + nb, g.alignment)
        if t.abs_offset + nb > g.file_size:
            problems.append(f"{t.name}: data runs past the end of the file")
    spans = sorted((t.offset, t.offset + (t.nbytes or 0), t.name) for t in g.tensors)
    for (a0, a1, an), (b0, _b1, bn) in zip(spans, spans[1:]):
        if b0 < a1:
            problems.append(f"{an} and {bn} overlap")
    return problems
