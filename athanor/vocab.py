"""A model's tokenizer, loaded by llama.cpp itself — never re-implemented.

``Vocab(path)`` loads a GGUF with ``vocab_only=True``: the vocabulary and
metadata, no weights, no GPU. Everything here goes through llama.cpp's own
functions, so what Athanor reports is what llama.cpp will do.

llama.cpp's load log is captured (not printed) — it is evidence: a missing
pre-tokenizer, for instance, is announced there and nowhere else. The
binding's own logger is put back after every load.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading
import time
from pathlib import Path

from .labels import MEASURED, Figure


class LlamaUnavailable(RuntimeError):
    """llama-cpp-python is not installed or its library would not load."""


class ModelLoadError(RuntimeError):
    """llama.cpp refused a file; ``log`` holds everything it said."""

    def __init__(self, path, log, what: str = "the model"):
        self.path = str(path)
        self.log = log
        tail = "".join(t for _lvl, t in log[-8:]).strip()
        super().__init__(f"llama.cpp could not load {what} {Path(path).name}"
                         + (f":\n{tail}" if tail else " (it printed nothing)"))


class VocabLoadError(ModelLoadError):
    def __init__(self, path, log):
        super().__init__(path, log, what="the vocabulary of")


VOCAB_TYPES = {0: "none", 1: "SPM", 2: "BPE", 3: "WPM", 4: "UGM", 5: "RWKV",
               6: "PLAMO2", 7: "TEST"}

ATTR_NAMES = (
    (1 << 0, "unknown"), (1 << 1, "unused"), (1 << 2, "normal"), (1 << 3, "control"),
    (1 << 4, "user_defined"), (1 << 5, "byte"), (1 << 6, "normalized"),
    (1 << 7, "lstrip"), (1 << 8, "rstrip"), (1 << 9, "single_word"),
)
ATTR_CONTROL = 1 << 3
ATTR_USER_DEFINED = 1 << 4
ATTR_UNUSED = 1 << 1
ATTR_BYTE = 1 << 5

_lock = threading.RLock()
_backend_ready = False
_capture_list = None
_capture_cb = None


def llama():
    """Import the binding, or say plainly why not."""
    try:
        import llama_cpp  # noqa: F401
    except Exception as exc:  # ImportError, or OSError from the shared library
        raise LlamaUnavailable(
            "llama-cpp-python is not available: " + f"{type(exc).__name__}: {exc}. "
            "Athanor uses the llama.cpp you already have — install a build of "
            "llama-cpp-python into this Python (see docs/INTEGRATING.md).") from exc
    return llama_cpp


def _ensure_backend(lc):
    global _backend_ready
    if not _backend_ready:
        lc.llama_backend_init()
        _backend_ready = True


def _capture_callback(lc):
    global _capture_cb
    if _capture_cb is None:
        @lc.llama_log_callback
        def cb(level, text, user_data):  # noqa: ARG001
            if _capture_list is not None:
                s = text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text)
                if level == LOG_CONT and _capture_list:
                    lvl, prev = _capture_list[-1]
                    _capture_list[-1] = (lvl, prev + s)
                else:
                    _capture_list.append((int(level), s))
        _capture_cb = cb
    return _capture_cb


def _restore_logger(lc):
    """The binding's own default logger (used only when the previous one
    cannot be read back)."""
    try:
        from llama_cpp import _logger
        lc.llama_log_set(_logger.llama_log_callback, ctypes.c_void_p(0))
    except Exception:
        lc.llama_log_set(ctypes.cast(None, lc.llama_log_callback), ctypes.c_void_p(0))


def _get_logger(lc):
    """Whatever logger is installed now — a host's (ATK sets its own), the
    binding's, or none — so it can be put back exactly."""
    get = getattr(lc, "llama_log_get", None)
    if get is None:
        return None
    try:
        cb = lc.llama_log_callback()
        ud = ctypes.c_void_p()
        get(ctypes.byref(cb), ctypes.byref(ud))
        return (cb, ud)
    except Exception:
        return None


@contextlib.contextmanager
def _capturing(lc, lines: list):
    """Route llama.cpp's log into ``lines`` for the block, then put back the
    logger that was there before — not a default of our choosing."""
    global _capture_list
    with _lock:
        saved = _get_logger(lc)
        prev = _capture_list
        _capture_list = lines
        lc.llama_log_set(_capture_callback(lc), ctypes.c_void_p(0))
        try:
            yield lines
        finally:
            _capture_list = prev
            if saved is None:
                _restore_logger(lc)
            else:
                lc.llama_log_set(saved[0], saved[1])


# ggml_log_level in b11093 (ggml.h). NOTE: llama-cpp-python ea3b56b's own
# _logger.py still maps the OLD order (1 info, 2 warn, 3 error, 4 debug), so
# with verbose=False the binding prints llama.cpp's WARNINGS and hides its
# ERRORS. Athanor captures by the header's order.
LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CONT = 1, 2, 3, 4, 5
LOG_LEVELS = {0: "none", LOG_DEBUG: "debug", LOG_INFO: "info", LOG_WARN: "warn",
              LOG_ERROR: "error"}


def _initial_capacity(nbytes: int) -> int:
    """First guess at how many tokens a text of ``nbytes`` can make (a token
    is at least one byte; +8 for BOS/EOS). The retry path handles the rest."""
    return nbytes + 8


class Vocab:
    """``with Vocab("model.gguf") as v: v.tokenize("hello")``"""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._lc = lc = llama()
        self._model = None
        self._vh = None
        self.log: list = []
        if not self.path.is_file():
            raise FileNotFoundError(str(self.path))
        from . import log
        st = self.path.stat()
        ident = {"path": str(self.path), "size": st.st_size, "mtime": st.st_mtime}
        log.breadcrumb("vocab-load", **ident)
        with _lock:
            _ensure_backend(lc)
            params = lc.llama_model_default_params()
            params.vocab_only = True
            params.use_mmap = True
            params.n_gpu_layers = 0
            with _capturing(lc, self.log):
                t0 = time.perf_counter()
                try:
                    model = lc.llama_model_load_from_file(os.fsencode(str(self.path)), params)
                finally:
                    self.load_seconds = time.perf_counter() - t0
        full_log = [(LOG_LEVELS.get(l, str(l)), t.rstrip("\n")) for l, t in self.log]
        if not model:
            log.event("vocab-load-failed", seconds=round(self.load_seconds, 4),
                      llama_cpp_log=full_log, **ident)
            raise VocabLoadError(self.path, self.log)
        self._model = model
        self._vh = lc.llama_model_get_vocab(model)
        self.n_tokens = int(lc.llama_vocab_n_tokens(self._vh))
        log.event("vocab-load", seconds=round(self.load_seconds, 4), n_tokens=self.n_tokens,
                  vocab_type=self.type, add_bos=self.add_bos, add_eos=self.add_eos,
                  llama_cpp_log=full_log, **ident)

    # ---------------------------------------------------------- lifecycle
    def close(self) -> None:
        if self._model:
            model, self._model, self._vh = self._model, None, None
            self._lc.llama_model_free(model)

    @property
    def _vocab(self):
        """The native vocabulary handle — refused once closed, because a
        freed handle crashes llama.cpp (or silently reads freed memory)."""
        if not self._model:
            raise RuntimeError("this Vocab is closed")
        return self._vh

    @property
    def closed(self) -> bool:
        return not self._model

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @contextlib.contextmanager
    def capture(self):
        """Capture what llama.cpp logs during a block (instead of printing it).

        ``with v.capture() as log: v.tokenize(...)`` — ``log`` is a list of
        (level, text). llama.cpp's own warnings are evidence, and should not
        leak onto a host's console. The host's own logger is put back after.
        """
        lines: list = []
        try:
            with _capturing(self._lc, lines):
                yield lines
        finally:
            if lines:
                from . import log
                log.event("llama-log", path=str(self.path),
                          lines=[(LOG_LEVELS.get(l, str(l)), t.rstrip("\n")) for l, t in lines])

    # ------------------------------------------------------------ the facts
    @property
    def type(self) -> str:
        return VOCAB_TYPES.get(int(self._lc.llama_vocab_type(self._vocab)), "?")

    @property
    def add_bos(self) -> bool:
        return bool(self._lc.llama_vocab_get_add_bos(self._vocab))

    @property
    def add_eos(self) -> bool:
        return bool(self._lc.llama_vocab_get_add_eos(self._vocab))

    def special_ids(self) -> dict:
        """The ids llama.cpp resolved for each special role (None if unset)."""
        lc, v = self._lc, self._vocab
        out = {}
        for role in ("bos", "eos", "eot", "sep", "nl", "pad", "mask", "fim_pre",
                     "fim_suf", "fim_mid", "fim_pad", "fim_rep", "fim_sep"):
            fn = getattr(lc, f"llama_vocab_{role}", None)
            if fn is None:
                continue
            tid = int(fn(v))
            out[role] = tid if 0 <= tid < self.n_tokens else None
        return out

    def eog_ids(self) -> list:
        """Every token llama.cpp treats as end-of-generation."""
        f = self._lc.llama_vocab_is_eog
        return [i for i in range(self.n_tokens) if f(self._vocab, i)]

    def log_lines(self, min_level: int = LOG_WARN) -> list:
        """Captured load messages at or above a level (default: warnings and
        errors), as (level name, text)."""
        return [(LOG_LEVELS.get(l, str(l)), t.rstrip()) for l, t in self.log
                if LOG_DEBUG <= l <= LOG_ERROR and l >= min_level]

    def log_text(self) -> str:
        return "".join(t for _l, t in self.log)

    def load_time(self) -> Figure:
        return Figure(round(self.load_seconds, 4), MEASURED, "s", "vocab-only load, wall clock")

    # ------------------------------------------------------- per token
    def _check(self, tid) -> int:
        """llama.cpp answers an id outside the vocabulary with a C++
        exception nothing catches — the whole process ends (a host with it).
        So the id is checked here first."""
        if isinstance(tid, bool) or not isinstance(tid, (int,)) and not hasattr(tid, "__index__"):
            raise TypeError(f"a token id must be an integer, not {type(tid).__name__}")
        tid = int(tid)
        if not 0 <= tid < self.n_tokens:
            raise IndexError(f"token id {tid} is outside this vocabulary (0–{self.n_tokens - 1})")
        if not self._model:
            raise RuntimeError("this Vocab is closed")
        return tid

    def text(self, tid: int) -> str:
        """The token's string as stored in the vocabulary."""
        b = self._lc.llama_vocab_get_text(self._vocab, self._check(tid))
        return b.decode("utf-8", "surrogateescape") if isinstance(b, bytes) else str(b)

    def attr(self, tid: int) -> int:
        return int(self._lc.llama_vocab_get_attr(self._vocab, self._check(tid)))

    def attr_names(self, tid: int) -> list:
        a = self.attr(tid)
        return [name for bit, name in ATTR_NAMES if a & bit]

    def is_control(self, tid: int) -> bool:
        return bool(self._lc.llama_vocab_is_control(self._vocab, self._check(tid)))

    def is_eog(self, tid: int) -> bool:
        return bool(self._lc.llama_vocab_is_eog(self._vocab, self._check(tid)))

    def piece(self, tid: int, *, special: bool = True) -> bytes:
        """The bytes this token produces (llama_token_to_piece)."""
        tid = self._check(tid)
        lc = self._lc
        n = 64
        while True:
            buf = ctypes.create_string_buffer(n)
            got = lc.llama_token_to_piece(self._vocab, tid, buf, n, 0, special)
            if got >= 0:
                return buf.raw[:got]
            n = -got

    # ----------------------------------------------------- text <-> ids
    def tokenize(self, text: str | bytes, *, add_special: bool = False,
                 parse_special: bool = True) -> list:
        """llama_tokenize. ``add_special`` adds BOS/EOS as the model is
        configured to; ``parse_special`` turns ``<|im_end|>``-style text into
        its control token (as a chat template's render needs)."""
        lc = self._lc
        if not self._model:
            raise RuntimeError("this Vocab is closed")
        b = text.encode("utf-8", "surrogateescape") if isinstance(text, str) else bytes(text)
        n_max = _initial_capacity(len(b))
        while True:
            arr = (lc.llama_token * n_max)()
            got = lc.llama_tokenize(self._vocab, b, len(b), arr, n_max, add_special, parse_special)
            if got >= 0:
                return list(arr[:got])
            n_max = -got

    def detokenize(self, ids, *, remove_special: bool = False,
                   unparse_special: bool = True) -> str:
        lc = self._lc
        ids = [self._check(i) for i in ids]
        arr = (lc.llama_token * max(1, len(ids)))(*ids)
        n = max(64, len(ids) * 16)
        while True:
            buf = ctypes.create_string_buffer(n)
            got = lc.llama_detokenize(self._vocab, arr, len(ids), buf, n, remove_special,
                                      unparse_special)
            if got >= 0:
                return buf.raw[:got].decode("utf-8", "replace")
            n = -got

    def _live_model(self):
        if not self._model:
            raise RuntimeError("this Vocab is closed")
        return self._model

    def meta(self, key: str) -> str | None:
        """A metadata value as llama.cpp parsed it (string form)."""
        lc = self._lc
        size = 4096
        while True:
            buf = ctypes.create_string_buffer(size)
            n = lc.llama_model_meta_val_str(self._live_model(), key.encode(), buf, size)
            if n < 0:
                return None
            if n < size:  # snprintf semantics: n is the full length
                return buf.raw[:n].decode("utf-8", "replace")
            size = n + 1
