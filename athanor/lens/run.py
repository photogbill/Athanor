"""The lens pass: every layer's output in a recording, read as logits.

    from athanor import lens
    lens.run("recordings/20261002-…-Qwen3-Coder.athrec-meta")
    rec = athanor.waterfall.read(path); rec.lens.depth      # a layer number per token

For every step the Tap recorded the residual stream at (``l_out-N`` for
each layer N), each layer's vector goes through the model's own final norm
and output matrix (``athanor.lens.unembed``) and becomes a full
distribution over the vocabulary — what the model would say if it stopped
at that layer. Kept per step and layer: the k likeliest tokens, where the
model's eventual favourite and the chosen token rank, the entropy; per
step, the DECISION DEPTH — the first layer from which the favourite is
the lens's answer all the way to the top (M30) — and the first layer at
which it appears among the k.

Honesty: the last layer's reading is compared with the recording's own
logits at every step (the favourite must agree; the log-probabilities of
the recorded candidates must match). That check decides the label: MEASURED
when the lens reproduces the model's output at the top, EXPERIMENTAL with
the numbers when it does not — a wrong norm, an architecture this
unembedding does not know, or a tapped layer that is not the last one.

Its limit, in the plan's words: a lens shows what can be READ OUT of a
layer in the output's own terms, not that the model "thinks in" those
words there. Early layers read poorly through the plain lens; the
calibrated (tuned) lens is the next step on this same track.
"""

from __future__ import annotations

import time

import numpy as np

from .. import __version__, log
from ..labels import EXPERIMENTAL, MEASURED
from ..tap.track import TAP_FLAG_NONE, TAP_FLAG_PARTIAL, TAP_FLAG_STOPPED
from ..util import utc_now, versions
from ..waterfall.reading import Recording, read as read_recording
from . import fileformat as lf
from .reading import LensData
from .unembed import LensUnavailable, for_recording

DEFAULT_K = 8
#: the last layer's lens must name the recording's favourite at this share
#: of the steps for the pass to be MEASURED
AGREEMENT_FLOOR = 0.98
CHECK_CANDIDATES = 16


def parse_layers(spec) -> list | None:
    """``"0-11"``, ``"3,7,20-23"`` or a list of ints → sorted layer numbers;
    None for all."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, (list, tuple, set, range)):
        return sorted({int(x) for x in spec})
    out: set = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                raise ValueError(f"layer range {part!r} runs backwards")
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _topk_sorted(logits: np.ndarray, k: int) -> tuple:
    """(ids [L, k], values [L, k]) best first; ties by the lower id — the
    recording's own rule."""
    L, V = logits.shape
    if k >= V:
        part = np.tile(np.arange(V), (L, 1))
    else:
        part = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(logits, part, axis=1)
    ids = np.empty_like(part)
    out_vals = np.empty_like(vals)
    for li in range(L):
        order = np.lexsort((part[li], -vals[li]))
        ids[li] = part[li][order]
        out_vals[li] = vals[li][order]
    return ids, out_vals


def _run_back(top_is: np.ndarray) -> int:
    """The first index from which ``top_is`` holds through the end; -1 if
    it does not hold at the end."""
    L = len(top_is)
    if L == 0 or not top_is[-1]:
        return -1
    i = L - 1
    while i > 0 and top_is[i - 1]:
        i -= 1
    return i


def run(recording, *, model_path=None, k: int = DEFAULT_K, layers=None, cache_dir=None,
        replace: bool = False, progress=None) -> dict:
    """Run the lens over a recording that carries the residual stream and
    write its lens track beside it. Returns the lens summary (``LensData.summary()``
    plus ``path``). ``progress(done, total, stage)`` is called while the
    output matrix is decoded (stage ``"unembedding"``) and per step
    (``"lens"``)."""
    started = time.perf_counter()
    rec = recording if isinstance(recording, Recording) else read_recording(recording)
    tap = rec.tap
    if tap is None:
        info = rec.tap_info or {}
        why = info.get("unavailable") or info.get("error") or "it was recorded without the Tap"
        raise LensUnavailable(f"{rec.meta_path.name} has no Tap track ({why}); record with "
                              "--tap residual")
    lays, R = tap.stacked("l_out")
    if lays is None:
        raise LensUnavailable(f"{rec.meta_path.name}'s Tap recorded {', '.join(tap.kinds())}, "
                              "not the residual stream (l_out-*); record with --tap residual")
    want = parse_layers(layers)
    if want is not None:
        keep = [i for i, l in enumerate(lays) if l in want]
        missing = sorted(set(want) - set(lays))
        if not keep:
            raise LensUnavailable(f"none of layers {want} is in the recording ({lays[0]}–{lays[-1]})")
        lays = [lays[i] for i in keep]
        R = R[:, keep, :]
    else:
        missing = []
    n_layer_model = tap.model.get("n_layer")
    n, L, E = R.shape
    if n != rec.n_steps:
        raise LensUnavailable(f"the Tap track has {n} steps, the recording {rec.n_steps}")

    def unembed_progress(done, total):
        if progress is not None:
            progress(done, total, "unembedding")

    u = for_recording(rec, model_path=model_path, cache_dir=cache_dir, progress=unembed_progress)
    if u.n_embd != E:
        raise LensUnavailable(f"the recorded residual vectors are {E} wide; {u.model.get('name')}'s "
                              f"output matrix is {u.n_embd} wide — not the same model")
    if rec.n_vocab and u.n_vocab != rec.n_vocab:
        raise LensUnavailable(f"the recording's vocabulary has {rec.n_vocab:,} tokens; this file's "
                              f"output matrix {u.n_vocab:,} — not the same model")
    V = u.n_vocab
    k = max(1, min(int(k), V))
    dt = lf.record_dtype(L, k)
    out = np.zeros(n, dt)
    at_top = n_layer_model is not None and lays[-1] == int(n_layer_model) - 1
    # running figures
    agree = np.zeros(L, np.int64)
    fav_lp_sum = np.zeros(L, np.float64)
    ent_sum = np.zeros(L, np.float64)
    counted = 0
    chk_n = chk_agree = 0
    chk_abs: list = []
    buf = np.empty((L, V), np.float32)
    dbuf = np.empty((L, V), np.float32)
    steps = rec.steps
    tap_flags = tap.flags
    lens_started = time.perf_counter()
    for s in range(n):
        row = out[s]
        H = np.asarray(R[s], np.float32)
        bad = bool(tap_flags[s] & (TAP_FLAG_NONE | TAP_FLAG_STOPPED | TAP_FLAG_PARTIAL))
        if bad or not np.isfinite(H).all():
            row["flags"] = lf.FLAG_NO_RESIDUAL
            row["depth"] = row["first_seen"] = row["chosen_depth"] = -1
            row["fav"] = int(steps["ids"][s][0])
            row["ids"] = -1
            row["logprobs"] = np.nan
            row["fav_rank"] = row["chosen_rank"] = -1
            row["fav_logprob"] = row["chosen_logprob"] = row["entropy"] = np.nan
            if progress is not None:
                progress(s + 1, n, "lens")
            continue
        logits = u.logits(H, out=buf)                                # [L, V]
        m = logits.max(axis=1, keepdims=True)
        np.subtract(logits, m, out=dbuf)
        e = np.exp(dbuf)
        ssum = e.sum(axis=1, dtype=np.float64)                      # [L]
        log_s = np.log(ssum)
        lse = m[:, 0].astype(np.float64) + log_s
        entropy = log_s - (e * dbuf).sum(axis=1, dtype=np.float64) / ssum
        fav = int(steps["ids"][s][0])
        chosen = int(steps["chosen"][s])
        ids, vals = _topk_sorted(logits, k)
        row["ids"] = ids
        row["logprobs"] = vals - lse[:, None]
        row["entropy"] = np.maximum(entropy, 0.0)
        row["fav"] = fav
        lf_ = logits[:, fav]
        fav_rank = (logits > lf_[:, None]).sum(axis=1)
        row["fav_rank"] = fav_rank
        row["fav_logprob"] = lf_ - lse
        top_is_fav = ids[:, 0] == fav
        depth = _run_back(top_is_fav)
        row["depth"] = depth
        seen = np.flatnonzero(fav_rank < k)
        row["first_seen"] = int(seen[0]) if len(seen) else -1
        flags = 0
        if depth < 0:
            flags |= lf.FLAG_TOP_DISAGREES
        if 0 <= chosen < V:
            lc = logits[:, chosen]
            row["chosen_rank"] = (logits > lc[:, None]).sum(axis=1)
            row["chosen_logprob"] = lc - lse
            row["chosen_depth"] = _run_back(ids[:, 0] == chosen)
        else:
            row["chosen_rank"] = -1
            row["chosen_logprob"] = np.nan
            row["chosen_depth"] = -1
        row["flags"] = flags
        agree += top_is_fav
        fav_lp_sum += row["fav_logprob"]
        ent_sum += row["entropy"]
        counted += 1
        if at_top:
            mc = min(CHECK_CANDIDATES, rec.k)
            rid = steps["ids"][s][:mc].astype(np.int64)
            rlp = steps["logprobs"][s][:mc].astype(np.float64)
            fin = np.isfinite(rlp) & (rid >= 0) & (rid < V)
            if fin.any():
                mine = logits[-1, rid[fin]].astype(np.float64) - lse[-1]
                chk_abs.append(np.abs(mine - rlp[fin]))
            chk_n += 1
            chk_agree += int(top_is_fav[-1])
        if progress is not None:
            progress(s + 1, n, "lens")
    lens_seconds = time.perf_counter() - lens_started

    notes: list = []
    if missing:
        notes.append(f"layers asked for but not in the recording: {missing}")
    if counted < n:
        notes.append(f"{n - counted} of {n} steps had no residual (the Tap had nothing there)")
    if "model_mismatch" in u.info:
        notes.append(u.info["model_mismatch"])
    check: dict = {"made": bool(at_top and chk_n), "steps_compared": chk_n}
    if at_top and chk_n:
        allabs = np.concatenate(chk_abs) if chk_abs else np.zeros(0)
        check.update({
            "favourite_agrees": chk_agree,
            "favourite_agreement": chk_agree / chk_n,
            "candidates_compared": int(allabs.size),
            "max_abs_logprob_error": float(allabs.max()) if allabs.size else None,
            "mean_abs_logprob_error": float(allabs.mean()) if allabs.size else None,
            "agreement_floor": AGREEMENT_FLOOR,
            "how": ("the last recorded layer's lens against the recording's own logits at every "
                    "step: does its favourite agree, and how far are its log-probabilities from "
                    "the recorded ones over the recording's top candidates"),
        })
        check["reproduces"] = check["favourite_agreement"] >= AGREEMENT_FLOOR
        if not check["reproduces"]:
            notes.append(f"the last layer's lens names the recording's favourite at only "
                         f"{100 * check['favourite_agreement']:.1f}% of steps: this unembedding "
                         "does not reproduce the model's output (a norm or architecture this lens "
                         "does not know?) — read every layer with that in mind")
    else:
        check["reproduces"] = None
        why = ("the model's layer count is unknown" if n_layer_model is None else
               f"the last layer (l_out-{int(n_layer_model) - 1}) is not in the recording")
        check["why_not"] = why
        notes.append(f"the lens could not be checked against the logits: {why}")
    label = MEASURED if check.get("reproduces") else EXPERIMENTAL
    cnt = max(counted, 1)
    # the text of the tokens the layers name that the recording never ranked
    from .pieces import resolve
    seen = set(np.unique(out["ids"]).tolist()) - set(rec.pieces) - {-1}
    pieces, pieces_from = resolve(u.model.get("path") or rec.model.get("path"), seen) \
        if seen else ({}, "the recording holds every piece")
    if seen and not pieces:
        notes.append(f"the text of {len(seen)} tokens the layers name could not be looked up "
                     f"({pieces_from}); they show as <id>")
    meta = {
        "format": lf.FORMAT,
        "version": lf.FORMAT_VERSION,
        "kind": "logit",
        "recording": rec.meta_path.name,
        "k": k,
        "layers": [int(x) for x in lays],
        "n_layer": n_layer_model,
        "n_steps": n,
        "n_vocab": V,
        "record_bytes": int(dt.itemsize),
        "layout": lf.LAYOUT,
        "flags": {str(f): m for f, m in lf.FLAG_MEANINGS.items()},
        "depth_rule": ("depth: the first layer (as an index into 'layers') from which the "
                       "recording's favourite is this lens's top token at every layer up to the "
                       "last; -1 when the last layer itself disagrees. first_seen: the first "
                       "layer at which the favourite is within the lens's k. chosen_depth: as "
                       "depth, for the token the sampler took"),
        "how": ("each recorded layer output (l_out-N, from the Tap) through the model's own "
                "final norm and output matrix, in float32 numpy, as a full distribution over the "
                "vocabulary — the logit lens (nostalgebraist 2020)"),
        "label": label,
        "check": check,
        "by_layer": {
            "agreement": (agree / cnt).round(5).tolist(),
            "fav_logprob_mean": (fav_lp_sum / cnt).round(5).tolist(),
            "entropy_mean": (ent_sum / cnt).round(5).tolist(),
        },
        "unembedding": u.describe(),
        "model": rec.model,
        "pieces": {str(t): s for t, s in sorted(pieces.items())},
        "pieces_from": pieces_from,
        "seconds": round(lens_seconds, 3),
        "ms_per_step": round(1000 * lens_seconds / n, 3) if n else None,
        "created": utc_now(),
        "athanor": __version__,
        "versions": versions(),
        "notes": notes,
    }
    path = lf.write(rec.meta_path, meta, out, replace=replace)
    log.event("lens-run", recording=str(rec.meta_path), steps=n, layers=L, k=k, label=label,
              seconds=round(time.perf_counter() - started, 3),
              agreement=check.get("favourite_agreement"))
    data = LensData(meta, out)
    summary = data.summary()
    summary["lens_kind"] = summary.pop("kind")
    summary["path"] = str(path)
    summary["kind"] = "lens"
    summary["recording"] = str(rec.meta_path)
    summary["model"] = rec.model
    summary["label"] = label
    summary["seconds_total"] = round(time.perf_counter() - started, 3)
    return summary


__all__ = ["run", "parse_layers", "DEFAULT_K", "AGREEMENT_FLOOR"]
