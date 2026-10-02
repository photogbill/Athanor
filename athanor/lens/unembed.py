"""The unembedding: a model's own way from a residual vector to its logits.

A transformer's layers all write into one running vector, the residual
stream, and the model turns that vector into token scores exactly once, at
the top: the final norm, then the output matrix. Because every layer's
output lives in the same space, the same two steps can be applied to the
output of layer 12, or 20, and read as "what the model would say if it
stopped here" — the logit lens (nostalgebraist, 2020). This module is those
two steps, taken from the GGUF itself:

* the final norm's weight (``output_norm.weight``), its epsilon from the
  file's metadata, and its kind — RMS (Llama, Qwen, Mistral, Gemma, …) or
  a LayerNorm with a bias (GPT-2, GPT-NeoX, Phi-2);
* the output matrix (``output.weight``, or ``token_embd.weight`` when the
  model ties them), dequantized once to float32 and kept on disk under the
  model's fingerprint, so the second lens pass on a model costs no decoding;
* Gemma's final logit soft-capping, when the file declares it; and
  ``output.bias`` when the model has one.

Nothing here is a measurement: it is the file's own numbers, decoded. The
check that they are the RIGHT numbers — that the last layer's output, put
through them, gives llama.cpp's own logits — is made against every
recording the lens is run on (``athanor.lens.run``) and by
``athanor tap probe``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .. import __version__
from ..gguf import read as read_gguf
from ..gguf.dequant import NoDequantizer, can_dequantize, dequantize_row, dequantize_row_range
from ..util import file_identity, utc_now

CACHE_FILE = "unembed.f32"
CACHE_META = "unembed.json"
CACHE_VERSION = 1
ROWS_PER_CHUNK = 8192


class LensUnavailable(RuntimeError):
    """This model's unembedding cannot be built here; the message says why."""


def fingerprint(header_sha256: str, size: int) -> str:
    """How a lens cache names its model: the GGUF header's SHA-256 (metadata,
    tokenizer and tensor layout) and the file's size. A different quant of
    the same model has a different header, so a different cache."""
    return f"{str(header_sha256)[:24]}-{int(size)}"


def default_dir() -> Path:
    from ..host import get_host
    return get_host().data_dir() / "lenses"


class Unembedding:
    """The final norm and the output matrix of one model, ready to apply.

    ``W`` is [n_vocab, n_embd] float32, memory-mapped from the cache;
    ``norm_w`` the norm's weight; ``norm_b`` its bias (LayerNorm only);
    ``eps`` its epsilon; ``kind`` ``"rms"`` or ``"layer"``. ``softcap`` is
    Gemma's final logit cap (None otherwise); ``bias`` the output bias."""

    def __init__(self, info: dict, W: np.ndarray):
        self.info = info
        self.W = W
        self.n_vocab, self.n_embd = int(W.shape[0]), int(W.shape[1])
        self.kind = info["norm"]["kind"]
        self.eps = float(info["norm"]["eps"])
        self.norm_w = np.asarray(info["norm"]["weight"], np.float32)
        b = info["norm"].get("bias")
        self.norm_b = np.asarray(b, np.float32) if b is not None else None
        ob = info["output"].get("bias")
        self.bias = np.asarray(ob, np.float32) if ob is not None else None
        self.softcap = info["output"].get("softcap")
        if self.norm_w.shape != (self.n_embd,):
            raise LensUnavailable(f"the norm weight has {self.norm_w.shape[0]} values; the "
                                  f"output matrix is {self.n_embd} wide")

    @property
    def model(self) -> dict:
        return self.info.get("model") or {}

    @property
    def fingerprint(self) -> str:
        return self.info["fingerprint"]

    # ------------------------------------------------------------ the maths
    def norm(self, x: np.ndarray) -> np.ndarray:
        """The final norm applied to rows of residual vectors (float32)."""
        x = np.asarray(x, np.float32)
        if self.kind == "rms":
            ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
            y = (x / np.sqrt(ms + self.eps)).astype(np.float32) * self.norm_w
        else:
            mu = np.mean(x.astype(np.float64), axis=-1, keepdims=True)
            var = np.mean((x.astype(np.float64) - mu) ** 2, axis=-1, keepdims=True)
            y = ((x - mu) / np.sqrt(var + self.eps)).astype(np.float32) * self.norm_w
            if self.norm_b is not None:
                y = y + self.norm_b
        return y

    def logits(self, h: np.ndarray, *, normed: bool = False, rows_per_chunk: int = 32768,
               out: np.ndarray | None = None) -> np.ndarray:
        """[n, n_vocab] float32 logits for residual rows ``h`` [n, n_embd]
        (``normed``: already through the norm). The output matrix is
        streamed in chunks of rows so a 150,000 × 5,000 matrix never has to
        be in memory twice."""
        h = np.asarray(h, np.float32)
        single = h.ndim == 1
        if single:
            h = h[None, :]
        n = self.norm(h) if not normed else h
        V = self.n_vocab
        res = out if out is not None else np.empty((n.shape[0], V), np.float32)
        for start in range(0, V, rows_per_chunk):
            stop = min(V, start + rows_per_chunk)
            np.matmul(n, np.asarray(self.W[start:stop]).T, out=res[:, start:stop])
        if self.bias is not None:
            res += self.bias
        if self.softcap:
            c = float(self.softcap)
            np.multiply(np.tanh(res / c), c, out=res)
        return res[0] if single else res

    def logits_for(self, h: np.ndarray, ids, *, normed: bool = False) -> np.ndarray:
        """[n, len(ids)]: the logits of a few tokens only — the cheap form
        for a live reading, when the candidates are already known."""
        h = np.asarray(h, np.float32)
        n = self.norm(h) if not normed else h
        ids = np.asarray(list(ids), np.int64)
        res = n @ np.asarray(self.W[ids]).T
        if self.bias is not None:
            res = res + self.bias[ids]
        if self.softcap:
            c = float(self.softcap)
            res = np.tanh(res / c) * c
        return res

    def describe(self) -> dict:
        o, nm = self.info["output"], self.info["norm"]
        return {"fingerprint": self.fingerprint, "model": self.model,
                "n_vocab": self.n_vocab, "n_embd": self.n_embd,
                "output": {k: o.get(k) for k in ("tensor", "type", "tied", "softcap")}
                | {"bias": o.get("bias") is not None},
                "norm": {"kind": nm["kind"], "eps": nm["eps"], "tensor": nm.get("tensor"),
                         "bias": nm.get("bias") is not None},
                "cache": self.info.get("cache"), "decode_seconds": self.info.get("decode_seconds")}


# ------------------------------------------------------------- the file
def _norm_spec(g, arch: str) -> dict:
    """What the final norm is, from the file."""
    t = g.tensor("output_norm.weight")
    if t is None:
        raise LensUnavailable("the file has no output_norm.weight; this lens knows the "
                              "llama-style final norm only")
    try:
        w = dequantize_row(g.tensor_bytes(t), t.ggml_type, int(t.shape[0]))
    except NoDequantizer as exc:
        raise LensUnavailable(f"output_norm.weight could not be read ({exc})") from exc
    bias_t = g.tensor("output_norm.bias")
    rms_eps = g.get(f"{arch}.attention.layer_norm_rms_epsilon")
    ln_eps = g.get(f"{arch}.attention.layer_norm_epsilon")
    if bias_t is not None or (rms_eps is None and ln_eps is not None):
        kind, eps = "layer", ln_eps if ln_eps is not None else rms_eps
    else:
        kind, eps = "rms", rms_eps
    if eps is None:
        raise LensUnavailable(f"the file declares no norm epsilon for {arch} "
                              "(no attention.layer_norm_rms_epsilon or layer_norm_epsilon)")
    spec = {"kind": kind, "eps": float(eps), "tensor": t.name, "type": t.type_name,
            "weight": w.astype(np.float32).tolist(), "bias": None}
    if bias_t is not None:
        try:
            b = dequantize_row(g.tensor_bytes(bias_t), bias_t.ggml_type, int(bias_t.shape[0]))
        except NoDequantizer as exc:
            raise LensUnavailable(f"output_norm.bias could not be read ({exc})") from exc
        spec["bias"] = b.astype(np.float32).tolist()
    return spec


def _output_spec(g, arch: str) -> tuple:
    out_t = g.tensor("output.weight")
    tied = out_t is None
    if tied:
        out_t = g.tensor("token_embd.weight")
    if out_t is None:
        raise LensUnavailable("the file has neither output.weight nor token_embd.weight")
    if len(out_t.shape) != 2:
        raise LensUnavailable(f"{out_t.name} is not a matrix ({list(out_t.shape)})")
    why = can_dequantize(out_t.ggml_type)
    if why:
        raise LensUnavailable(f"{out_t.name} is {out_t.type_name}: {why}")
    spec = {"tensor": out_t.name, "type": out_t.type_name, "tied": tied, "bias": None,
            "softcap": None}
    bias_t = g.tensor("output.bias")
    if bias_t is not None:
        try:
            b = dequantize_row(g.tensor_bytes(bias_t), bias_t.ggml_type, int(bias_t.shape[0]))
        except NoDequantizer as exc:
            raise LensUnavailable(f"output.bias could not be read ({exc})") from exc
        spec["bias"] = b.astype(np.float32).tolist()
    cap = g.get(f"{arch}.final_logit_softcapping")
    if cap:
        spec["softcap"] = float(cap)
    return out_t, spec


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(model_path, *, cache_dir=None, progress=None) -> Unembedding:
    """Decode the model's unembedding and write it to the cache (or load it
    when the cache already holds this model). ``progress(done_rows,
    total_rows)`` is called as the matrix is decoded."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"no model at {model_path}")
    g = read_gguf(model_path)
    header = g.header_sha256()
    identity = file_identity(model_path, header_sha256=header)
    fp = fingerprint(header, identity["size"])
    folder = Path(cache_dir) if cache_dir else default_dir()
    here = folder / fp
    cached = load(here, expect=fp)
    if cached is not None:
        cached.info.setdefault("model", identity)
        return cached
    arch = g.architecture or ""
    norm = _norm_spec(g, arch)
    out_t, output = _output_spec(g, arch)
    V, E = out_t.n_rows, int(out_t.shape[0])
    if len(norm["weight"]) != E:
        raise LensUnavailable(f"output_norm.weight has {len(norm['weight'])} values but "
                              f"{out_t.name} is {E} wide")
    here.mkdir(parents=True, exist_ok=True)
    tmp = here / (CACHE_FILE + ".part")
    started = time.perf_counter()
    try:
        with open(tmp, "wb") as f:
            done = 0
            while done < V:
                stop = min(V, done + ROWS_PER_CHUNK)
                block = dequantize_row_range(g, out_t, done, stop)
                f.write(np.ascontiguousarray(block, dtype="<f4").tobytes())
                done = stop
                if progress is not None:
                    progress(done, V)
            f.flush()
            os.fsync(f.fileno())
        seconds = time.perf_counter() - started
        info = {
            "version": CACHE_VERSION,
            "athanor": __version__,
            "fingerprint": fp,
            "model": identity,
            "arch": arch,
            "n_vocab": V, "n_embd": E,
            "dtype": "<f4", "file": CACHE_FILE,
            "bytes": V * E * 4,
            "sha256": _sha256_file(tmp),
            "norm": norm,
            "output": output,
            "decode_seconds": round(seconds, 3),
            "created": utc_now(),
            "how": ("the file's own output_norm and output matrix, dequantized to float32 by "
                    "Athanor (ggml's layouts) — not a measurement; checked against llama.cpp's "
                    "logits whenever the lens runs"),
        }
        os.replace(tmp, here / CACHE_FILE)
        with open(here / (CACHE_META + ".part"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(info, f, indent=1)
            f.write("\n")
        os.replace(here / (CACHE_META + ".part"), here / CACHE_META)
    except BaseException:
        for p in (tmp, here / (CACHE_META + ".part")):
            try:
                p.unlink()
            except OSError:
                pass
        raise
    from .. import log
    log.event("lens-unembed-built", model=str(model_path), fingerprint=fp, n_vocab=V, n_embd=E,
              seconds=round(seconds, 3), output_type=output["type"])
    return load(here, expect=fp)


def load(folder, *, expect: str | None = None) -> Unembedding | None:
    """The cached unembedding in ``folder``, or None if there is none (or it
    is damaged: a damaged cache is ignored and rebuilt, never trusted)."""
    folder = Path(folder)
    meta, data = folder / CACHE_META, folder / CACHE_FILE
    if not meta.is_file() or not data.is_file():
        return None
    try:
        with open(meta, encoding="utf-8") as f:
            info = json.load(f)
        if info.get("version") != CACHE_VERSION:
            return None
        if expect is not None and info.get("fingerprint") != expect:
            return None
        V, E = int(info["n_vocab"]), int(info["n_embd"])
        if data.stat().st_size != V * E * 4:
            return None
        W = np.memmap(data, dtype="<f4", mode="r", shape=(V, E))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    info["cache"] = str(folder)
    return Unembedding(info, W)


def for_recording(rec, *, model_path=None, cache_dir=None, progress=None) -> Unembedding:
    """The unembedding for the model a recording was made with: from the
    cache by the recording's own fingerprint when it is there, else built
    from the model file (the recording's path, or ``model_path``). A model
    file that is not the one recorded is refused unless it was named
    explicitly — and then the mismatch is written into the result."""
    m = rec.model or {}
    folder = Path(cache_dir) if cache_dir else default_dir()
    fp = fingerprint(m["header_sha256"], m["size"]) if m.get("header_sha256") and m.get("size") \
        else None
    if fp is not None and model_path is None:
        cached = load(folder / fp, expect=fp)
        if cached is not None:
            cached.info.setdefault("model", m)
            return cached
    path = Path(model_path) if model_path else Path(m.get("path") or "")
    if not str(path) or not path.is_file():
        raise LensUnavailable(
            "the model file is needed once, to decode its output matrix, and "
            + (f"{path} is not there" if str(path) else "the recording does not say where it was")
            + " — name it with --model")
    u = build(path, cache_dir=folder, progress=progress)
    if fp is not None and u.fingerprint != fp:
        if model_path is None:
            raise LensUnavailable(f"{path.name} is not the file this recording was made with "
                                  f"(the fingerprints differ) — name the right one with --model")
        u.info["model_mismatch"] = (f"{path.name} is not the file the recording names; the lens "
                                    "was read through this file's norm and output matrix")
    return u


__all__ = ["Unembedding", "LensUnavailable", "build", "load", "for_recording", "fingerprint",
           "default_dir"]
