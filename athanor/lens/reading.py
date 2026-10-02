"""Reading a lens track back: what the player, the command line and any
other program use. Needs only numpy."""

from __future__ import annotations

import math

import numpy as np

from . import fileformat as lf


class LensData:
    """A recording's lens track, loaded. Arrays are [steps, layers, …];
    ``layers`` names the model layer each column is (``l_out-N``'s N)."""

    def __init__(self, meta: dict, records: np.ndarray):
        self.meta = meta
        self.records = records
        self.layers = [int(x) for x in (meta.get("layers") or [])]
        self.k = int(meta.get("k") or 0)
        self.n_layer = meta.get("n_layer")          # the model's, from the Tap's facts
        #: the text of tokens the layers name that the recording never ranked
        self.pieces = {int(k): v for k, v in (meta.get("pieces") or {}).items()}

    def piece(self, token_id: int, fallback=None) -> str:
        """The text of a token: the recording's (``fallback``, usually
        ``Recording.piece``) when it knows it, else the lens's own list,
        else ``<id>``."""
        tid = int(token_id)
        if fallback is not None:
            s = fallback(tid)
            if s is not None and s != f"<{tid}>":
                return s
        return self.pieces.get(tid, f"<{tid}>")

    def unresolved(self, known=None) -> set:
        """Token ids the layers name whose text neither the lens's list nor
        ``known`` (the recording's ``pieces``) holds."""
        ids = set(np.unique(self.records["ids"]).tolist()) - {-1}
        return ids - set(self.pieces) - set(known or ())

    def add_pieces(self, more: dict) -> None:
        """Texts looked up later (``athanor.lens.pieces.resolve``)."""
        self.pieces.update({int(k): v for k, v in more.items()})

    def __len__(self) -> int:
        return len(self.records)

    @property
    def n_steps(self) -> int:
        return len(self.records)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def kind(self) -> str:
        return self.meta.get("kind") or "logit"

    @property
    def label(self) -> str:
        return self.meta.get("label") or "EXPERIMENTAL"

    @property
    def check(self) -> dict:
        return self.meta.get("check") or {}

    # ------------------------------------------------------------- columns
    @property
    def flags(self) -> np.ndarray:
        return self.records["flags"]

    @property
    def depth(self) -> np.ndarray:
        """Decision depth per step, as a LAYER NUMBER (-1: undecided)."""
        return self._to_layer(self.records["depth"])

    @property
    def first_seen(self) -> np.ndarray:
        return self._to_layer(self.records["first_seen"])

    @property
    def chosen_depth(self) -> np.ndarray:
        return self._to_layer(self.records["chosen_depth"])

    def _to_layer(self, col: np.ndarray) -> np.ndarray:
        """Record columns hold INDICES into ``layers``; readers want the
        layer's number."""
        out = np.full(col.shape, -1, np.int64)
        ok = col >= 0
        if ok.any():
            lay = np.asarray(self.layers, np.int64)
            out[ok] = lay[col[ok]]
        return out

    def depth_index(self) -> np.ndarray:
        """Decision depth per step as an index into ``layers`` (-1: undecided)."""
        return self.records["depth"].astype(np.int64)

    @property
    def fav_rank(self) -> np.ndarray:
        return self.records["fav_rank"]

    @property
    def fav_logprob(self) -> np.ndarray:
        return self.records["fav_logprob"]

    @property
    def chosen_rank(self) -> np.ndarray:
        return self.records["chosen_rank"]

    @property
    def chosen_logprob(self) -> np.ndarray:
        return self.records["chosen_logprob"]

    @property
    def entropy(self) -> np.ndarray:
        return self.records["entropy"]

    def top(self, step: int) -> tuple:
        """(ids [L, k], logprobs [L, k]) — what each layer would say at a step."""
        r = self.records[step]
        return r["ids"], r["logprobs"]

    # --------------------------------------------------------------- views
    def column(self, step: int, piece=None, n: int = 3) -> list:
        """One step, layer by layer, for reading: each layer's ``n`` likeliest
        tokens with their probabilities, and the favourite's rank there."""
        r = self.records[step]
        out = []
        for li, layer in enumerate(self.layers):
            cands = []
            for tid, lp in zip(r["ids"][li][:n], r["logprobs"][li][:n]):
                tid, lp = int(tid), float(lp)
                cands.append({"id": tid, "piece": self.piece(tid, piece) if tid >= 0 else None,
                              "p": math.exp(lp) if math.isfinite(lp) else 0.0})
            out.append({"layer": layer, "candidates": cands,
                        "fav_rank": int(r["fav_rank"][li]),
                        "fav_p": _p(r["fav_logprob"][li]),
                        "chosen_rank": int(r["chosen_rank"][li]),
                        "chosen_p": _p(r["chosen_logprob"][li]),
                        "entropy_bits": float(r["entropy"][li]) / math.log(2)
                        if np.isfinite(r["entropy"][li]) else None})
        return out

    def grid(self, what: str = "fav_logprob") -> np.ndarray:
        """[steps, layers] of one quantity (``fav_logprob``, ``fav_rank``,
        ``chosen_logprob``, ``chosen_rank``, ``entropy``) — the picture: time
        down, depth across."""
        return np.asarray(self.records[what])

    def agreement_by_layer(self) -> np.ndarray:
        """Per layer, the fraction of steps at which that layer's lens answer
        IS the model's eventual favourite — how far each depth can be read
        with the plain lens."""
        ok = (self.flags & lf.FLAG_NO_RESIDUAL) == 0
        if not ok.any():
            return np.full(self.n_layers, np.nan, np.float32)
        return (self.fav_rank[ok] == 0).mean(axis=0).astype(np.float32)

    def depth_histogram(self) -> np.ndarray:
        """Count of steps decided at each layer (by ``layers`` index); the
        undecided are not counted."""
        d = self.records["depth"]
        h = np.zeros(self.n_layers, np.int64)
        np.add.at(h, d[d >= 0], 1)
        return h

    def summary(self) -> dict:
        d = self.depth
        ok = d >= 0
        return {
            "kind": self.kind,
            "label": self.label,
            "k": self.k,
            "layers": self.layers,
            "n_steps": self.n_steps,
            "check": self.check,
            "depth": {
                "median": float(np.median(d[ok])) if ok.any() else None,
                "mean": float(d[ok].mean()) if ok.any() else None,
                "undecided": int((~ok).sum()),
                "at_last_layer": int((d == self.layers[-1]).sum()) if self.layers else 0,
                "histogram": self.depth_histogram().tolist(),
            },
            "agreement_by_layer": [None if not np.isfinite(x) else round(float(x), 4)
                                   for x in self.agreement_by_layer()],
            "unembedding": self.meta.get("unembedding"),
            "seconds": self.meta.get("seconds"),
            "notes": self.meta.get("notes") or [],
        }


def _p(lp) -> float | None:
    lp = float(lp)
    return math.exp(lp) if math.isfinite(lp) else None


def read(recording_path, *, verify: bool = False) -> LensData | None:
    got = lf.read(recording_path, verify=verify)
    if got is None:
        return None
    meta, records = got
    return LensData(meta, records)


__all__ = ["LensData", "read"]
