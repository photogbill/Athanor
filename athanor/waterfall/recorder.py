"""The Recorder: one step per token, from the model's raw logits.

Engine-agnostic — it is handed a logits vector and the token that was
chosen, and needs nothing else (``attach`` feeds it from llama-cpp-python;
any other runtime can feed it the same way). The work per step is a few
passes over the vocabulary (a partial sort for the top k, one exp for the
normaliser and the entropy), timed and reported as the recording's own
overhead.
"""

from __future__ import annotations

import codecs
import math
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import fileformat as ff

CHUNK = 256


class Recorder:
    """Collects steps; ``close()`` resolves the token texts (needs the model
    still loaded); ``save()`` writes the files (needs nothing)."""

    def __init__(self, *, n_vocab: int, k: int = ff.DEFAULT_K,
                 piece: Callable[[int], bytes] | None = None,
                 is_eog: Callable[[int], bool] | None = None,
                 is_control: Callable[[int], bool] | None = None,
                 clock: Callable[[], float] = time.perf_counter):
        if not isinstance(n_vocab, (int, np.integer)) or n_vocab < 1:
            raise ValueError(f"n_vocab must be a positive integer, not {n_vocab!r}")
        self.n_vocab = int(n_vocab)
        self.k = min(int(k), self.n_vocab)
        self.dtype = ff.step_dtype(self.k)
        self._piece, self._is_eog, self._is_control = piece, is_eog, is_control
        self._clock = clock
        self.t0 = clock()
        self.started = datetime.now(timezone.utc)
        self._chunks: list = []
        self._buf = np.zeros(CHUNK, self.dtype)
        self._n_buf = 0
        self.n_steps = 0
        self.prompt_ids: list | None = None
        self.meta: dict = {}           # the host's: settings, messages, host, model
        self.error: str | None = None  # set once; recording stops, generation does not
        self.overhead_s = 0.0
        self.closed = False
        self.annotations: list = []    # marks made while recording (segments)
        self._raw: dict = {}           # token id -> bytes, filled by resolve()
        self.tap_track = None          # athanor.tap.track.TapTrack, when the Tap records too

    # ------------------------------------------------------------ recording
    def begin(self, prompt_ids=None) -> None:
        if prompt_ids is not None:
            self.prompt_ids = [int(i) for i in prompt_ids]

    def fail(self, exc: BaseException) -> None:
        """Stop recording (the generation goes on) and keep the reason."""
        if self.error is None:
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                from .. import log
                log.exception("waterfall-record-failed", exc, step=self.n_steps)
            except Exception:
                pass

    def step(self, logits, chosen: int, t: float | None = None) -> None:
        """One token: the raw logits it was chosen from, and the choice."""
        now = self._clock() if t is None else t
        start = self._clock()
        x = np.asarray(logits, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.n_vocab:
            raise ValueError(f"logits have {x.shape[0]} entries; the vocabulary has {self.n_vocab}")
        flags = 0
        finite = np.isfinite(x)
        if not finite.all():
            flags |= ff.FLAG_NONFINITE
            x = np.where(np.isnan(x), -np.inf, x).astype(np.float32)
        row = self._next_row()
        m = float(x.max())
        if not math.isfinite(m):            # nothing is possible: record, don't divide
            row["chosen"], row["rank"] = int(chosen), -1
            row["logprob"] = row["entropy"] = np.nan
            row["tail"] = 1.0
            row["ids"] = np.arange(self.k)
            row["logprobs"] = -np.inf
        else:
            d = x - m                                       # <= 0; -inf where masked
            e = np.exp(d)                                   # float32, -inf -> 0
            s = float(e.sum(dtype=np.float64))              # pairwise, in float64
            lse = m + math.log(s)
            if flags & ff.FLAG_NONFINITE:
                d = np.where(e > 0, d, 0).astype(np.float32)
            # H = lse - sum(p x) = log s - sum(e (x - m)) / s; every term has
            # one sign, and the sum runs in float64
            entropy = math.log(s) - float((e * d).sum(dtype=np.float64)) / s
            n = self.n_vocab
            if self.k < n:                                  # the k best; at a tie, lower ids
                v = np.partition(x, n - self.k)[n - self.k]
                above = np.flatnonzero(x > v)
                idx = np.concatenate([above, np.flatnonzero(x == v)[: self.k - len(above)]])
            else:
                idx = np.arange(n)
            order = np.lexsort((idx, -x[idx]))              # best first; ties by id
            ids = idx[order]
            lp = x[ids].astype(np.float64) - lse
            row["ids"] = ids
            row["logprobs"] = lp
            row["tail"] = max(0.0, 1.0 - float(np.exp(lp).sum()))
            row["entropy"] = max(0.0, entropy)
            c = int(chosen)
            if 0 <= c < n:
                xc = x[c]
                row["chosen"] = c
                row["rank"] = int(np.count_nonzero(x > xc))
                row["logprob"] = float(xc) - lse
            else:
                row["chosen"], row["rank"], row["logprob"] = c, -1, np.nan
        c = int(chosen)
        if 0 <= c < self.n_vocab:
            if self._is_eog is not None and self._is_eog(c):
                flags |= ff.FLAG_EOG
            if self._is_control is not None and self._is_control(c):
                flags |= ff.FLAG_CONTROL
        row["flags"] = flags
        row["t"] = now - self.t0
        self.n_steps += 1
        self.overhead_s += self._clock() - start

    def _next_row(self):
        if self._n_buf == len(self._buf):
            self._chunks.append(self._buf)
            self._buf = np.zeros(CHUNK, self.dtype)
            self._n_buf = 0
        row = self._buf[self._n_buf]
        self._n_buf += 1
        return row

    def steps(self) -> np.ndarray:
        parts = self._chunks + [self._buf[: self._n_buf]]
        return np.concatenate(parts) if parts else np.zeros(0, self.dtype)

    def mark(self, label: str, **fields) -> None:
        """An annotation starting at the next step (e.g. a new segment)."""
        a = {"athrec:step_start": self.n_steps, "athrec:label": label}
        a.update({f"athrec:{k}": v for k, v in fields.items()})
        self.annotations.append(a)

    # ------------------------------------------------------- the token texts
    def resolve(self) -> None:
        """Look up the text of every token seen so far and not yet known.
        Needs the model loaded; ``attach`` calls it on leaving each block,
        then ``detach``es, so nothing can reach a model that has since been
        unloaded. Safe to call any number of times; a failure is kept."""
        if self._piece is None:
            return
        try:
            steps = self.steps()
            want = set(np.unique(steps["ids"]).tolist()) if len(steps) else set()
            want |= set(int(c) for c in steps["chosen"])
            if self.prompt_ids:
                want |= set(self.prompt_ids)
            for tid in sorted(want - self._raw.keys()):
                self._raw[tid] = self._piece(tid) if 0 <= tid < self.n_vocab else b""
        except Exception as exc:
            self.fail(exc)

    def detach(self) -> None:
        """Forget the functions that reach into the model."""
        self._piece = self._is_eog = self._is_control = None

    def close(self) -> None:
        """Resolve, and mark the recording finished (no more steps expected)."""
        self.resolve()
        self.closed = True

    def _texts(self, steps) -> tuple:
        raw = self._raw
        pieces = {tid: b.decode("utf-8", "backslashreplace") for tid, b in raw.items()}
        text, spans = _join([raw.get(int(c), b"") for c in steps["chosen"]])
        prompt = None
        if self.prompt_ids is not None and self.prompt_ids and all(
                t in raw for t in self.prompt_ids):
            prompt = b"".join(raw[t] for t in self.prompt_ids).decode("utf-8", "replace")
        return pieces, text, spans, prompt

    @property
    def reply_text(self) -> str:
        return self._texts(self.steps())[1]

    # --------------------------------------------------------------- saving
    def save(self, directory, *, name: str | None = None, finish_reason: str | None = None,
             meta: dict | None = None, host_reply: str | None = None) -> Path:
        """Write the recording into ``directory``; returns the meta file's path.

        ``host_reply`` is the text the host showed for this generation: if it
        is not empty and no step was recorded, the runtime generated without
        passing through the hook, and the recording says so."""
        from .. import __version__
        from ..labels import MEASURED
        from ..util import utc_now, versions
        self.resolve()          # a no-op once detached from the model
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        info = dict(self.meta)
        if meta:
            info.update(meta)
        model = _identify(info.pop("model", None) or {})
        stem = _stem(self.started, name or model.get("name") or "recording")
        meta_path = directory / (stem + ff.META_SUFFIX)
        data_path = directory / (stem + ff.DATA_SUFFIX)
        steps = self.steps()
        n = len(steps)
        pieces, reply_text, spans, prompt_text = self._texts(steps)
        if n == 0 and host_reply and self.error is None:
            self.error = ("no step was recorded although the model replied: this runtime "
                          "did not sample through Llama.sample")
        elapsed = float(steps["t"][-1]) if n else 0.0
        first = float(steps["t"][0]) if n else None
        rate = (n - 1) / (elapsed - first) if n > 1 and elapsed > first else None
        doc = {
            "global": {
                "athrec:version": ff.FORMAT_VERSION,
                "athrec:k": self.k,
                "athrec:step_bytes": self.dtype.itemsize,
                "athrec:layout": ff.LAYOUT,
                "athrec:n_steps": n,
                "athrec:n_vocab": self.n_vocab,
                "athrec:created": utc_now(),
                "athrec:started": self.started.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "athrec:recorder": f"athanor {__version__}",
                "athrec:versions": versions(),
                "athrec:label": MEASURED,
                "athrec:how": ("the model's raw logits, read from llama.cpp at each sampling "
                               "step before any sampler touched them; top-k kept, the rest "
                               "summed as 'tail'"),
                "athrec:model": model,
                "athrec:settings": info.pop("settings", {}),
                "athrec:host": info.pop("host", None),
                "athrec:error": self.error,
            },
            "captures": [{"athrec:step_start": 0, "athrec:datetime": self.started.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ")}],
            "annotations": (list(info.pop("annotations", None) or []) + list(self.annotations)
                            + _thinking([pieces.get(int(c), "") for c in steps["chosen"]])),
            "prompt": {"n_tokens": info.pop("prompt_tokens", None) or (
                           len(self.prompt_ids) if self.prompt_ids is not None else None),
                       "ids": self.prompt_ids or None, "text": prompt_text,
                       "note": info.pop("prompt_note", None)},
            "messages": info.pop("messages", None),
            "reply": {"text": reply_text, "spans": spans,
                      "finish_reason": finish_reason, "n_steps": n},
            "timing": {"prompt_seconds": first, "total_seconds": elapsed,
                       "tokens_per_second": rate,
                       "overhead_ms_per_step": (1000 * self.overhead_s / n) if n else None,
                       "label": MEASURED},
            "pieces": {str(k): v for k, v in sorted(pieces.items())},
        }
        tap_records = None
        if self.tap_track is not None:
            tap_records, doc["tap"] = self.tap_track.finish(n)
        if info:
            doc["extra"] = info
        ff.write(meta_path, data_path, doc, steps, tap=tap_records)
        return meta_path


def _join(chunks: list) -> tuple[str, list]:
    """The reply as text, and each token's [start, end) in it. A token that
    ends inside a multi-byte character owns nothing; the next one owns the
    whole character."""
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    parts, spans, pos = [], [], 0
    for b in chunks:
        s = dec.decode(b)
        parts.append(s)
        spans.append([pos, pos + len(s)])
        pos += len(s)
    rest = dec.decode(b"", final=True)
    if rest:
        parts.append(rest)
        if spans:
            spans[-1][1] += len(rest)
    return "".join(parts), spans


THINK_MARKERS = (("<think>", "</think>"), ("[THINK]", "[/THINK]"),
                 ("<thinking>", "</thinking>"))


def _thinking(pieces: list) -> list:
    """Annotations for the stretches where the model was thinking, found by
    its markers. A closing marker with no opening one in the reply means the
    template opened the thought in the prompt, so it began at step 0."""
    out = []
    for open_m, close_m in THINK_MARKERS:
        start = None
        seen_any = False
        for i, p in enumerate(pieces):
            s = p.strip()
            if s == open_m and start is None:
                start, seen_any = i, True
            elif s == close_m:
                begin = start if start is not None else (0 if not seen_any and not out else None)
                if begin is not None:
                    out.append({"athrec:step_start": begin, "athrec:step_count": i - begin + 1,
                                "athrec:label": "thinking", "athrec:markers": [open_m, close_m]})
                start, seen_any = None, True
        if start is not None:           # still thinking when the reply was cut off
            out.append({"athrec:step_start": start, "athrec:step_count": len(pieces) - start,
                        "athrec:label": "thinking", "athrec:markers": [open_m, close_m],
                        "athrec:unfinished": True})
        if out:
            break
    return out


def _identify(model: dict) -> dict:
    """The model's file identity (size, date, header SHA-256) when only its
    path is known. Never fails the save."""
    path = model.get("path")
    if not path or "size" in model:
        return dict(model)
    out = dict(model)
    try:
        from ..gguf import read
        from ..util import file_identity
        try:
            header = read(path).header_sha256()
        except Exception:
            header = None
        out.update(file_identity(path, header_sha256=header))
    except OSError as exc:
        out["identity_error"] = f"{type(exc).__name__}: {exc}"
    return out


_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def _stem(started: datetime, name: str) -> str:
    slug = _SLUG.sub("-", Path(str(name)).stem).strip("-._")[:48] or "recording"
    return f"{started.strftime('%Y%m%d-%H%M%S')}-{slug}-{secrets.token_hex(3)}"
