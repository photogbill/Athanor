"""Test fixtures: synthetic GGUFs written by Athanor itself, and the real
vocabularies llama.cpp tests with.

* ``tiny_gguf(...)`` — a small file with a made-up vocabulary and a few
  tensors. Structure only; llama.cpp cannot run it, and no test asks it to.
* ``VOCABS`` — the three vocab-only GGUFs in tests/data/vocab (MIT, from
  llama.cpp), always present.
* ``extra_vocab(name)`` — more of llama.cpp's set when ATHANOR_TEST_VOCAB_DIR
  points at a llama.cpp ``models/`` folder.
* ``REAL_MODEL`` — ATHANOR_TEST_MODEL, a real model of yours, if set.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from athanor.gguf import Array, GGMLType, GGUFWriter, ValueType

# Tests never write into the real per-user data folder: the notebook and the
# log of a test run go to a scratch folder (set before anything reads it).
if not os.environ.get("ATHANOR_TEST_KEEP_DATA"):
    import atexit
    import shutil
    _scratch = tempfile.mkdtemp(prefix="athanor-tests-")
    os.environ["ATHANOR_DATA"] = _scratch
    atexit.register(shutil.rmtree, _scratch, ignore_errors=True)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VOCAB_DIR = DATA / "vocab"
VOCABS = {
    "llama-spm": VOCAB_DIR / "ggml-vocab-llama-spm.gguf",
    "phi-3": VOCAB_DIR / "ggml-vocab-phi-3.gguf",
    "gpt-neox": VOCAB_DIR / "ggml-vocab-gpt-neox.gguf",
}
REAL_MODEL = os.environ.get("ATHANOR_TEST_MODEL") or None


def extra_vocab(name: str) -> Path | None:
    d = os.environ.get("ATHANOR_TEST_VOCAB_DIR")
    if not d:
        return None
    p = Path(d) / f"ggml-vocab-{name}.gguf"
    return p if p.is_file() else None


def all_extra_vocabs() -> list:
    d = os.environ.get("ATHANOR_TEST_VOCAB_DIR")
    return sorted(Path(d).glob("ggml-vocab-*.gguf")) if d and Path(d).is_dir() else []


def have_llama() -> bool:
    try:
        import llama_cpp  # noqa: F401
        return True
    except Exception:
        return False


HAVE_LLAMA = have_llama()
needs_llama = unittest.skipUnless(HAVE_LLAMA, "llama-cpp-python is not installed here")


def tiny_tokens(n: int = 64) -> list:
    return ["<unk>", "<s>", "</s>"] + [f"tok{i}" for i in range(n - 3)]


def tiny_gguf(path, *, n_vocab: int = 64, n_embd: int = 32, emb: np.ndarray | None = None,
              emb_type: int = GGMLType.F32, emb_rows: int | None = None,
              name: str = "Tiny", ctx: int = 4096, template: str | None = None,
              tokens: list | None = None, extra: list | None = None,
              alignment: int | None = None, pre: str | None = None,
              tokenizer_model: str = "llama") -> Path:
    """Write a small synthetic GGUF and return its path."""
    tokens = tokens or tiny_tokens(n_vocab)
    rows = emb_rows or len(tokens)
    rng = np.random.default_rng(7)
    if emb is None:
        emb = rng.standard_normal((rows, n_embd)).astype(np.float32)
    w = GGUFWriter(alignment=alignment)
    w.add("general.architecture", "llama")
    w.add("general.name", name)
    w.add("general.file_type", 0, ValueType.UINT32)
    w.add("llama.context_length", ctx, ValueType.UINT32)
    w.add("llama.embedding_length", n_embd, ValueType.UINT32)
    w.add("llama.block_count", 1, ValueType.UINT32)
    w.add("llama.attention.head_count", 4, ValueType.UINT32)
    w.add("llama.attention.head_count_kv", 2, ValueType.UINT32)
    w.add("tokenizer.ggml.model", tokenizer_model)
    if pre is not None:
        w.add("tokenizer.ggml.pre", pre)
    w.add("tokenizer.ggml.tokens", tokens, ValueType.ARRAY, ValueType.STRING)
    w.add("tokenizer.ggml.token_type", [2, 3, 3] + [1] * (len(tokens) - 3),
          ValueType.ARRAY, ValueType.INT32)
    w.add("tokenizer.ggml.scores", np.zeros(len(tokens), np.float32),
          ValueType.ARRAY, ValueType.FLOAT32)
    w.add("tokenizer.ggml.bos_token_id", 1, ValueType.UINT32)
    w.add("tokenizer.ggml.eos_token_id", 2, ValueType.UINT32)
    if template:
        w.add("tokenizer.chat_template", template)
    for key, value, vtype, *elem in (extra or []):
        w.add(key, value, vtype, *(elem or []))
    if emb_type == GGMLType.F32:
        data = emb.astype(np.float32)
    elif emb_type == GGMLType.F16:
        data = emb.astype(np.float16)
    elif emb_type == GGMLType.Q8_0:
        data = quantize_q8_0(emb)
    else:
        raise ValueError("tiny_gguf writes F32, F16 or Q8_0 embeddings")
    w.add_tensor("token_embd.weight", (n_embd, rows), emb_type, data)
    w.add_tensor("blk.0.attn_norm.weight", (n_embd,), GGMLType.F32,
                 np.ones(n_embd, np.float32))
    w.add_tensor("output_norm.weight", (n_embd,), GGMLType.F32, np.ones(n_embd, np.float32))
    return w.write_file(path)


def quantize_q8_0(x: np.ndarray) -> bytes:
    """Reference Q8_0 (ggml's quantize_row_q8_0_ref): per 32 values, d =
    max|x| / 127 stored as f16, q = round(x / d)."""
    flat = x.astype(np.float32).reshape(-1, 32)
    amax = np.abs(flat).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0)
    q = np.round(flat * inv[:, None]).astype(np.int8)
    out = bytearray()
    for di, qi in zip(d.astype(np.float16), q):
        out += di.tobytes() + qi.tobytes()
    return bytes(out)


class TempDir:
    def __enter__(self):
        self._t = tempfile.TemporaryDirectory()
        return Path(self._t.name)

    def __exit__(self, *exc):
        self._t.cleanup()
