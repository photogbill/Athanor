"""GGUF and ggml constants, as defined in llama.cpp b11093.

Sources (read, not recalled): ``ggml/include/ggml.h`` (``enum ggml_type``),
``ggml/include/gguf.h`` (``enum gguf_type``) and gguf-py's
``GGML_QUANT_SIZES``. A type id this table does not know is kept, reported
and round-tripped; only its byte size is unknown.
"""

from __future__ import annotations

from enum import IntEnum

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
SUPPORTED_VERSIONS = (2, 3)


class ValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# struct format character and numpy dtype character for each scalar type
# (endianness is added by the caller).
SCALAR_STRUCT = {
    ValueType.UINT8: "B",
    ValueType.INT8: "b",
    ValueType.UINT16: "H",
    ValueType.INT16: "h",
    ValueType.UINT32: "I",
    ValueType.INT32: "i",
    ValueType.FLOAT32: "f",
    ValueType.BOOL: "?",
    ValueType.UINT64: "Q",
    ValueType.INT64: "q",
    ValueType.FLOAT64: "d",
}

SCALAR_NUMPY = {
    ValueType.UINT8: "u1",
    ValueType.INT8: "i1",
    ValueType.UINT16: "u2",
    ValueType.INT16: "i2",
    ValueType.UINT32: "u4",
    ValueType.INT32: "i4",
    ValueType.FLOAT32: "f4",
    ValueType.BOOL: "u1",   # one byte, 0 or 1; kept as raw bytes to round-trip
    ValueType.UINT64: "u8",
    ValueType.INT64: "i8",
    ValueType.FLOAT64: "f8",
}

SCALAR_SIZE = {
    ValueType.UINT8: 1,
    ValueType.INT8: 1,
    ValueType.UINT16: 2,
    ValueType.INT16: 2,
    ValueType.UINT32: 4,
    ValueType.INT32: 4,
    ValueType.FLOAT32: 4,
    ValueType.BOOL: 1,
    ValueType.UINT64: 8,
    ValueType.INT64: 8,
    ValueType.FLOAT64: 8,
}


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40
    Q1_0 = 41
    Q2_0 = 42


_QK_K = 256

# (elements per block, bytes per block) — gguf-py's GGML_QUANT_SIZES, b11093.
BLOCK = {
    GGMLType.F32: (1, 4),
    GGMLType.F16: (1, 2),
    GGMLType.Q4_0: (32, 2 + 16),
    GGMLType.Q4_1: (32, 2 + 2 + 16),
    GGMLType.Q5_0: (32, 2 + 4 + 16),
    GGMLType.Q5_1: (32, 2 + 2 + 4 + 16),
    GGMLType.Q8_0: (32, 2 + 32),
    GGMLType.Q8_1: (32, 2 + 2 + 32),
    GGMLType.Q2_K: (256, 2 + 2 + _QK_K // 16 + _QK_K // 4),
    GGMLType.Q3_K: (256, 2 + _QK_K // 4 + _QK_K // 8 + 12),
    GGMLType.Q4_K: (256, 2 + 2 + _QK_K // 2 + 12),
    GGMLType.Q5_K: (256, 2 + 2 + _QK_K // 2 + _QK_K // 8 + 12),
    GGMLType.Q6_K: (256, 2 + _QK_K // 2 + _QK_K // 4 + _QK_K // 16),
    GGMLType.Q8_K: (256, 4 + _QK_K + _QK_K // 8),
    GGMLType.IQ2_XXS: (256, 2 + _QK_K // 4),
    GGMLType.IQ2_XS: (256, 2 + _QK_K // 4 + _QK_K // 32),
    GGMLType.IQ3_XXS: (256, 2 + _QK_K // 4 + _QK_K // 8),
    GGMLType.IQ1_S: (256, 2 + _QK_K // 8 + _QK_K // 16),
    GGMLType.IQ4_NL: (32, 2 + 16),
    GGMLType.IQ3_S: (256, 2 + _QK_K // 4 + _QK_K // 8 + _QK_K // 32 + 4),
    GGMLType.IQ2_S: (256, 2 + _QK_K // 4 + _QK_K // 16),
    GGMLType.IQ4_XS: (256, 2 + 2 + _QK_K // 2 + _QK_K // 64),
    GGMLType.I8: (1, 1),
    GGMLType.I16: (1, 2),
    GGMLType.I32: (1, 4),
    GGMLType.I64: (1, 8),
    GGMLType.F64: (1, 8),
    GGMLType.IQ1_M: (256, _QK_K // 8 + _QK_K // 16 + _QK_K // 32),
    GGMLType.BF16: (1, 2),
    GGMLType.TQ1_0: (256, 2 + 4 * 13),
    GGMLType.TQ2_0: (256, 2 + 64),
    GGMLType.MXFP4: (32, 1 + 16),
    GGMLType.NVFP4: (64, 4 + 32),
    GGMLType.Q1_0: (128, 2 + 16),
    GGMLType.Q2_0: (64, 2 + 16),
}


def type_name(ggml_type: int) -> str:
    """The ggml name of a tensor type id, or ``type<N>`` for one b11093 lacks."""
    try:
        return GGMLType(ggml_type).name
    except ValueError:
        return f"type{ggml_type}"


def tensor_nbytes(ggml_type: int, shape) -> int | None:
    """Bytes a tensor occupies, or None when the type's block size is unknown.

    ``shape`` is in ggml order (ne0 first); ne0 must be a whole number of
    blocks, as ggml requires.
    """
    try:
        per_block, block_bytes = BLOCK[GGMLType(ggml_type)]
    except (ValueError, KeyError):
        return None
    n = 1
    for d in shape:
        n *= int(d)
    if not shape or int(shape[0]) % per_block:
        return None
    return n // per_block * block_bytes


def row_nbytes(ggml_type: int, ne0: int) -> int | None:
    """Bytes in one row of ``ne0`` elements, or None when unknown."""
    return tensor_nbytes(ggml_type, (ne0,))
