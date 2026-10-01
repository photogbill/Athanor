"""Reading a recording back: what the player, the command line and any
other program use. Needs only numpy."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import fileformat as ff


#: architectures whose router hands llama.cpp logits, mixed by a softmax over
#: the chosen experts (LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT)
LOGIT_ROUTERS = {"gpt-oss"}


class ExpertRouting:
    """A mixture-of-experts model's routing, step by step, from the Tap's
    ``experts`` streams (M28). Arrays are [steps, layers, …]; -1 / NaN where
    the Tap had nothing for a step.

    ``ids``: the experts each layer's router chose for the token.
    ``weights``: the router's value for each chosen expert (llama.cpp's
    ``ffn_moe_weights``, before normalising), or None.
    ``probs``: the router's value for every expert (``ffn_moe_probs``: a
    softmax for Mixtral and Qwen, a sigmoid for DeepSeek V3, raw logits for
    gpt-oss), or None."""

    def __init__(self, layers: list, ids, weights=None, probs=None, n_expert: int | None = None,
                 arch: str | None = None):
        self.layers = list(layers)
        self.arch = arch
        self._share = None
        self.ids = ids
        self.weights = weights
        self.probs = probs
        seen = int(ids.max()) + 1 if ids.size and ids.max() >= 0 else 0
        width = probs.shape[2] if probs is not None else 0
        self.n_expert = int(n_expert or max(seen, width))
        self.n_used = int(ids.shape[2]) if ids.ndim == 3 else 0

    @property
    def n_steps(self) -> int:
        return int(self.ids.shape[0])

    def share(self) -> np.ndarray | None:
        """How much each chosen expert counted, summing to 1 per step and
        layer: the chosen weights normalised — or, for a router that gives
        logits (gpt-oss; or any router with a negative value), a softmax
        over them, which is how llama.cpp mixes those."""
        if self.weights is None:
            return None
        if self._share is not None:
            return self._share
        w = self.weights.astype(np.float64)
        finite = np.isfinite(w)
        logits = self.arch in LOGIT_ROUTERS or bool(finite.any() and (w[finite] < 0).any())
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            if logits:
                e = np.exp(w - np.nanmax(np.where(finite, w, -np.inf), axis=2, keepdims=True))
                out = e / np.nansum(e, axis=2, keepdims=True)
            else:
                out = w / np.nansum(w, axis=2, keepdims=True)
        self._share = out.astype(np.float32)
        return self._share

    def grid(self, step: int) -> dict:
        """One step, for drawing: ``chosen`` [layers, experts] as the share
        each chosen expert got (0 elsewhere; NaN for an expert that was used
        but whose share is unknown), and ``router`` [layers, experts] (the
        router's value for every expert, or None)."""
        L, X = len(self.layers), self.n_expert
        chosen = np.zeros((L, X), np.float32)
        share = self.share()
        ids = self.ids[step]
        for li in range(L):
            for j, e in enumerate(ids[li]):
                if 0 <= e < X:
                    chosen[li, e] = share[step, li, j] if share is not None else 1.0
                    if share is not None and not np.isfinite(chosen[li, e]):
                        chosen[li, e] = np.nan
        router = self.probs[step] if self.probs is not None else None
        return {"chosen": chosen, "router": router}

    def usage(self, steps=None, *, weighted: bool = False) -> np.ndarray:
        """[layers, experts]: the fraction of the given steps (default all)
        at which each expert was chosen — or, ``weighted``, the mean share
        it got."""
        idx = np.arange(self.n_steps) if steps is None else np.asarray(list(steps), np.int64)
        L, X = len(self.layers), self.n_expert
        out = np.zeros((L, X), np.float64)
        if not len(idx):
            return out.astype(np.float32)
        ids = self.ids[idx]                               # [s, L, used]
        share = self.share()
        w = share[idx] if (weighted and share is not None) else np.ones(ids.shape, np.float64)
        valid = (ids >= 0) & (ids < X)
        n_valid_steps = np.maximum(valid.any(axis=2).sum(axis=0), 1)   # per layer
        for li in range(L):
            e = ids[:, li, :][valid[:, li, :]]
            ww = np.nan_to_num(w[:, li, :][valid[:, li, :]].astype(np.float64))
            np.add.at(out[li], e, ww)
            out[li] /= n_valid_steps[li]
        return out.astype(np.float32)


class TapData:
    """What the Tap recorded beside each step of a recording."""

    def __init__(self, block: dict, records: np.ndarray):
        self.block = block
        self.records = records
        self.layout = list(block.get("layout") or [])
        self.names = [s["name"] for s in self.layout]
        self.model = dict(block.get("model") or {})

    def __len__(self) -> int:
        return len(self.records)

    @property
    def flags(self) -> np.ndarray:
        return self.records["flags"]

    def stream(self, name: str) -> np.ndarray:
        """[steps, count] for one stream (``"ffn_moe_topk-3"``)."""
        return self.records[name]

    def kinds(self) -> list:
        return sorted({s["kind"] for s in self.layout})

    def stacked(self, kind: str):
        """(layers, array [steps, layers, count]) for every layer's stream of
        one kind (``"l_out"``, ``"ffn_moe_topk"``); (None, None) if absent."""
        rows = sorted((s["layer"], s["name"]) for s in self.layout
                      if s["kind"] == kind and s["layer"] is not None)
        if not rows:
            return None, None
        counts = {self.records[name].shape[1] for _l, name in rows}
        if len(counts) != 1:
            return None, None
        return [l for l, _n in rows], np.stack([self.records[n] for _l, n in rows], axis=1)

    def experts(self) -> ExpertRouting | None:
        """The routing of a mixture-of-experts model, if the Tap recorded it."""
        layers, ids = self.stacked("ffn_moe_topk")
        if layers is None:
            return None

        def same_layers(kind):
            ls, arr = self.stacked(kind)
            return arr if ls == layers else None

        return ExpertRouting(layers, ids, same_layers("ffn_moe_weights"),
                             same_layers("ffn_moe_probs"), self.model.get("n_expert"),
                             self.model.get("arch"))


class Recording:
    """A recording, loaded. ``steps`` is the structured array described in
    fileformat; everything else comes from the meta file."""

    def __init__(self, meta_path: Path, data_path: Path, meta: dict, steps: np.ndarray):
        self.meta_path, self.data_path = meta_path, data_path
        self.meta = meta
        self.steps = steps
        g = meta["global"]
        self.k = int(g["athrec:k"])
        self.n_vocab = int(g.get("athrec:n_vocab") or 0)
        reply = meta.get("reply") or {}
        self.text = reply.get("text") or ""
        self.spans = [tuple(s) for s in (reply.get("spans") or [])]
        self.finish_reason = reply.get("finish_reason")
        self.pieces = {int(k): v for k, v in (meta.get("pieces") or {}).items()}
        self._tap = False              # not read yet
        self._verify = False

    # ------------------------------------------------------------- shape
    def __len__(self) -> int:
        return len(self.steps)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def model(self) -> dict:
        return self.meta["global"].get("athrec:model") or {}

    @property
    def settings(self) -> dict:
        return self.meta["global"].get("athrec:settings") or {}

    @property
    def error(self) -> str | None:
        return self.meta["global"].get("athrec:error")

    @property
    def tap(self) -> TapData | None:
        """What the Tap recorded beside each step, or None (read on first use;
        a missing or damaged Tap file raises RecordingError here, not when
        the recording is opened)."""
        if self._tap is False:
            block = self.meta.get("tap")
            records = ff.read_tap(self.meta, self.meta_path, verify=self._verify)
            self._tap = TapData(block, records) if records is not None else None
        return self._tap

    @property
    def tap_info(self) -> dict | None:
        """The meta file's account of the Tap (streams, errors, notes), even
        when it recorded nothing."""
        t = self.meta.get("tap")
        return t if isinstance(t, dict) else None

    @property
    def title(self) -> str:
        return self.model.get("name") or self.meta_path.name

    # ------------------------------------------------------------- steps
    def piece(self, token_id: int) -> str:
        return self.pieces.get(int(token_id), f"<{int(token_id)}>")

    def span(self, step: int) -> tuple[int, int]:
        if 0 <= step < len(self.spans):
            return self.spans[step]
        return (len(self.text), len(self.text))

    def step_at_char(self, pos: int) -> int | None:
        """The step whose text covers character ``pos`` (for clicks)."""
        import bisect
        starts = [s for s, _e in self.spans]
        i = bisect.bisect_right(starts, pos) - 1
        while i >= 0 and self.spans[i][0] == self.spans[i][1]:
            i -= 1                     # an empty span (half a character) is never the target
        if i < 0:
            return None
        return i

    def candidates(self, step: int, n: int = 20) -> list[dict]:
        """The n most probable tokens at a step, best first, with the one
        chosen marked."""
        row = self.steps[step]
        chosen = int(row["chosen"])
        out = []
        for tid, lp in zip(row["ids"][:n], row["logprobs"][:n]):
            tid = int(tid)
            out.append({"id": tid, "piece": self.piece(tid), "logprob": float(lp),
                        "p": math.exp(float(lp)) if math.isfinite(float(lp)) else 0.0,
                        "chosen": tid == chosen})
        return out

    def chosen(self, step: int) -> dict:
        row = self.steps[step]
        lp = float(row["logprob"])
        return {"step": step, "id": int(row["chosen"]), "piece": self.piece(int(row["chosen"])),
                "rank": int(row["rank"]), "logprob": lp,
                "p": math.exp(lp) if math.isfinite(lp) else None,
                "entropy_bits": float(row["entropy"]) / math.log(2),
                "tail": float(row["tail"]), "t": float(row["t"]), "flags": int(row["flags"])}

    def doubts(self, below: float = 0.5) -> np.ndarray:
        """The moments of doubt: steps where the sampler took something other
        than the model's favourite, or where even the token taken had less
        than ``below`` of the probability. Sorted, as step numbers."""
        s = self.steps
        if not len(s):
            return np.zeros(0, np.int64)
        lp = s["logprob"].astype(np.float64)
        with np.errstate(invalid="ignore"):
            weak = np.isfinite(lp) & (lp < math.log(below))
        return np.flatnonzero((s["rank"] > 0) | weak)

    def summary(self) -> dict:
        s = self.steps
        n = len(s)
        off_favourite = int(np.count_nonzero(s["rank"] > 0)) if n else 0
        return {
            "kind": "recording",
            "path": str(self.meta_path),
            "model": self.model,
            "n_steps": n,
            "k": self.k,
            "finish_reason": self.finish_reason,
            "settings": self.settings,
            "timing": self.meta.get("timing"),
            "mean_entropy_bits": (float(np.nanmean(s["entropy"]) / math.log(2))
                                  if n and np.isfinite(s["entropy"]).any() else None),
            "not_the_favourite": off_favourite,
            "moments_of_doubt": int(len(self.doubts())),
            "error": self.error,
            "tap": _tap_summary(self.tap_info),
            "text": self.text,
        }


def _tap_summary(t: dict | None) -> dict | None:
    if not t:
        return None
    kinds = sorted({s["kind"] for s in t.get("layout") or []})
    return {"streams": t.get("streams"), "recorded": kinds, "steps": t.get("n_steps"),
            "error": t.get("error"), "unavailable": t.get("unavailable"),
            "stopped_at": t.get("stopped_at"), "notes": t.get("notes"),
            "overhead_ms_per_step": t.get("overhead_ms_per_step"), "model": t.get("model")}


def read(path, *, verify: bool = False) -> Recording:
    """A recording, named by its meta file, its data file, or its stem."""
    meta_path, data_path = ff.stems(path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"no recording at {meta_path}")
    meta = ff.read_meta(meta_path)
    if not data_path.is_file():
        raise ff.RecordingError(f"{meta_path.name}: its data file {data_path.name} is missing")
    steps = ff.read_steps(meta, data_path, verify=verify)
    rec = Recording(meta_path, data_path, meta, steps)
    rec._verify = verify
    if verify:
        rec.tap                        # noqa: B018 — check the Tap file now, too
    return rec


#: list_recordings' cache: path -> ((meta mtime, meta size, data size), row).
#: A meta file carries the prompt, the messages and every candidate's text —
#: tens of milliseconds to parse — and a host lists the folder every time
#: its tab is shown, so an unchanged file is parsed once.
_ROWS: dict = {}


def _listing_row(p: Path) -> dict:
    st = p.stat()
    data = ff.stems(p)[1]
    try:
        data_size = data.stat().st_size
    except OSError:
        data_size = None
    key = (st.st_mtime_ns, st.st_size, data_size)
    hit = _ROWS.get(str(p))
    if hit is not None and hit[0] == key:
        return dict(hit[1])
    meta = ff.read_meta(p)
    g = meta["global"]
    model = g.get("athrec:model") or {}
    reply = meta.get("reply") or {}
    text = reply.get("text") or ""
    error = g.get("athrec:error")
    n = int(g.get("athrec:n_steps") or 0)
    want = n * int(g.get("athrec:step_bytes") or ff.step_dtype(int(g["athrec:k"])).itemsize)
    if data_size is None:
        error = f"its data file {data.name} is missing"
    elif data_size != want:
        error = (f"its data file is damaged or incomplete ({data_size} bytes; "
                 f"{n} steps need {want})")
    tap = meta.get("tap") if isinstance(meta.get("tap"), dict) else None
    row = {"path": str(p), "created": g.get("athrec:created"),
           "started": g.get("athrec:started"), "model": model.get("name"),
           "n_steps": g.get("athrec:n_steps"), "finish_reason": reply.get("finish_reason"),
           "preview": text[:160], "error": error,
           "tap": sorted({s["kind"] for s in (tap or {}).get("layout") or []}) or None}
    _ROWS[str(p)] = (key, row)
    return dict(row)


def list_recordings(directory) -> list[dict]:
    """Every recording in a folder, newest first, from the meta files (and
    the data files' sizes). Damaged ones are listed with their error rather
    than left out. Safe to call from a worker thread."""
    d = Path(directory)
    if not d.is_dir():
        return []
    out = []
    for p in d.glob("*" + ff.META_SUFFIX):
        try:
            out.append(_listing_row(p))
        except Exception as exc:
            out.append({"path": str(p), "created": None, "started": None, "model": None,
                        "n_steps": None, "finish_reason": None, "preview": "",
                        "error": f"{type(exc).__name__}: {exc}", "tap": None})
    out.sort(key=lambda r: (r.get("started") or r.get("created") or "", r["path"]), reverse=True)
    return out
