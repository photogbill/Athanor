"""Compare — two files (plan §3.3).

Question: how do these two differ — and are they the same family?

Tokenizer, metadata and tensors side by side, from the headers alone. Then
lineage: when the vocabularies and shapes match, sampled embedding rows are
decoded and compared numerically — a fine-tune keeps its base's embeddings
very close; unrelated models do not. And across a library, which models
share a vocabulary at all (the precondition for a draft model, a LoRA, or a
tokenizer transplant).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .. import gguf
from ..gguf.dequant import NoDequantizer, dequantize_rows
from ..labels import DECLARED, ESTIMATE, MEASURED, Figure
from ..util import file_identity

_TOKENIZER_ARRAYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type",
                     "tokenizer.ggml.scores", "tokenizer.ggml.merges")


def _digest(v) -> str:
    h = hashlib.sha256()
    if isinstance(v, gguf.Array):
        if isinstance(v.items, np.ndarray):
            h.update(v.items.tobytes())
        else:
            for s in v.tolist():
                h.update(repr(s).encode("utf-8", "surrogateescape"))
                h.update(b"\0")
    else:
        h.update(repr(v).encode("utf-8", "surrogateescape"))
    return h.hexdigest()[:16]


def tokenizer_diff(a, b, *, show: int = 20) -> dict:
    ta = a.get("tokenizer.ggml.tokens") or []
    tb = b.get("tokenizer.ggml.tokens") or []
    same_list = ta == tb
    diffs = []
    n_diff = 0
    for i in range(min(len(ta), len(tb))):
        if ta[i] != tb[i]:
            n_diff += 1
            if len(diffs) < show:
                diffs.append({"id": i, "a": ta[i], "b": tb[i]})
    sa, sb = set(ta), set(tb)
    only_a = sorted(sa - sb)
    only_b = sorted(sb - sa)
    specials = {}
    keys = {k for k in a.keys() + b.keys()
            if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
    for k in sorted(keys):
        va, vb = a.get(k), b.get(k)
        if va != vb:
            specials[k[len("tokenizer.ggml."):]] = {"a": va, "b": vb}
    arrays = {}
    for k in _TOKENIZER_ARRAYS:
        ka, kb = a.kv(k), b.kv(k)
        arrays[k] = {"a": _digest(ka.value) if ka else None, "b": _digest(kb.value) if kb else None}
        arrays[k]["identical"] = arrays[k]["a"] == arrays[k]["b"]
    union = len(sa | sb)
    return {
        "model": {"a": a.get("tokenizer.ggml.model"), "b": b.get("tokenizer.ggml.model")},
        "pre": {"a": a.get("tokenizer.ggml.pre"), "b": b.get("tokenizer.ggml.pre")},
        "n_tokens": {"a": len(ta), "b": len(tb)},
        "identical": same_list and all(x["identical"] for x in arrays.values()),
        "same_token_list": same_list,
        "ids_that_differ": n_diff,
        "first_differences": diffs,
        "only_in_a": {"count": len(only_a), "sample": only_a[:show]},
        "only_in_b": {"count": len(only_b), "sample": only_b[:show]},
        "overlap": Figure(round(len(sa & sb) / union, 4) if union else None, DECLARED, "share",
                          "tokens (by string) in both ÷ tokens in either").to_dict(),
        "special_ids_differ": specials,
        "arrays": arrays,
        "label": DECLARED,
    }


def metadata_diff(a, b) -> dict:
    ka, kb = set(a.keys()), set(b.keys())
    skip = set(_TOKENIZER_ARRAYS)
    changed = []
    for k in sorted((ka & kb) - skip):
        va, vb = a.kv(k), b.kv(k)
        if va.type != vb.type or _digest(va.value) != _digest(vb.value):
            pa, pb = va.plain(), vb.plain()
            if isinstance(pa, (list, np.ndarray)) or isinstance(pb, (list, np.ndarray)):
                pa = f"array of {len(pa)}"
                pb = f"array of {len(pb)}"
            elif isinstance(pa, str) and len(pa) > 200:
                pa = pa[:200] + "…"
            if isinstance(pb, str) and len(pb) > 200:
                pb = pb[:200] + "…"
            changed.append({"key": k, "a": pa, "b": pb})
    return {"only_in_a": sorted(ka - kb), "only_in_b": sorted(kb - ka), "changed": changed,
            "label": DECLARED}


def tensor_diff(a, b, *, show: int = 40) -> dict:
    na = {t.name: t for t in a.tensors}
    nb = {t.name: t for t in b.tensors}
    shape, qtype = [], []
    for name in sorted(set(na) & set(nb)):
        x, y = na[name], nb[name]
        if tuple(x.shape) != tuple(y.shape):
            shape.append({"tensor": name, "a": list(x.shape), "b": list(y.shape)})
        if x.ggml_type != y.ggml_type:
            qtype.append({"tensor": name, "a": x.type_name, "b": y.type_name})
    return {
        "n_tensors": {"a": len(na), "b": len(nb)},
        "only_in_a": sorted(set(na) - set(nb))[:show],
        "only_in_b": sorted(set(nb) - set(na))[:show],
        "n_only_in_a": len(set(na) - set(nb)),
        "n_only_in_b": len(set(nb) - set(na)),
        "shape_differs": shape[:show], "n_shape_differs": len(shape),
        "type_differs": qtype[:show], "n_type_differs": len(qtype),
        "same_architecture": a.architecture == b.architecture,
        "label": DECLARED,
    }


def _shared_rows(a, b, n_rows_a: int, n_rows_b: int, sample: int) -> list:
    """Pairs (row in A, row in B) for the same token, spread across the
    vocabulary. Matched by token STRING, so a fine-tune that appended tokens
    (or reordered some) is compared on the tokens both files share."""
    ta = a.get("tokenizer.ggml.tokens") or []
    tb = b.get("tokenizer.ggml.tokens") or []
    if ta and tb:
        where_b = {}
        for j, tok in enumerate(tb):
            where_b.setdefault(tok, j)
        pairs = [(i, where_b[tok]) for i, tok in enumerate(ta)
                 if tok in where_b and i < n_rows_a and where_b[tok] < n_rows_b]
    else:
        pairs = [(i, i) for i in range(min(n_rows_a, n_rows_b))]
    if not pairs:
        return []
    pick = np.unique(np.linspace(0, len(pairs) - 1, num=min(sample, len(pairs))).astype(int))
    return [pairs[k] for k in pick]


def lineage(a, b, *, sample: int = 64, tensor: str = "token_embd.weight") -> dict:
    """Are these built on the same base? Sampled rows, decoded and compared."""
    out = {"tensor": tensor, "label": MEASURED}
    ta, tb = a.tensor(tensor), b.tensor(tensor)
    if ta is None or tb is None:
        for alt in ("output.weight",):
            if alt != tensor and a.tensor(alt) is not None and b.tensor(alt) is not None:
                return lineage(a, b, sample=sample, tensor=alt)
        out.update(verdict=None, why=f"{tensor} is missing from one file")
        return out
    if ta.shape[0] != tb.shape[0] or len(ta.shape) != len(tb.shape):
        out.update(verdict="not the same base",
                   why=f"row width {ta.shape[0]} vs {tb.shape[0]} — a different model shape",
                   verdict_label=DECLARED)
        return out
    pairs = _shared_rows(a, b, ta.n_rows, tb.n_rows, sample)
    if not pairs:
        out.update(verdict="not the same base", why="the two vocabularies share no tokens",
                   verdict_label=DECLARED)
        return out
    rows_a = [p for p, _ in pairs]
    rows_b = [q for _, q in pairs]
    out["rows_compared"] = len(pairs)
    out["matched_by"] = "token string"
    if ta.n_rows != tb.n_rows:
        out["vocabulary_rows"] = {"a": ta.n_rows, "b": tb.n_rows}
    if ta.ggml_type == tb.ggml_type:
        same = sum(x == y for x, y in zip(a.row_bytes(ta, rows_a), b.row_bytes(tb, rows_b)))
        out["rows_byte_identical"] = same
    try:
        xa = dequantize_rows(a, ta, rows_a)
        xb = dequantize_rows(b, tb, rows_b)
    except NoDequantizer as exc:
        out.update(verdict=None, why=str(exc))
        return out
    na = np.linalg.norm(xa, axis=1)
    nb = np.linalg.norm(xb, axis=1)
    ok = (na > 0) & (nb > 0)
    if not ok.any():
        out.update(verdict=None, why="the sampled rows are all zero")
        return out
    cos = np.full(len(pairs), np.nan)
    cos[ok] = (xa[ok] * xb[ok]).sum(axis=1) / (na[ok] * nb[ok])
    rel = np.linalg.norm(xa - xb, axis=1) / np.maximum(na, 1e-12)
    out["cosine"] = {"mean": float(np.nanmean(cos)), "min": float(np.nanmin(cos)),
                     "max": float(np.nanmax(cos))}
    out["relative_difference"] = {"mean": float(np.mean(rel)), "max": float(np.max(rel))}
    m = out["cosine"]["mean"]
    if out.get("rows_byte_identical") == len(pairs):
        verdict = "identical rows"
    elif m > 0.98:
        verdict = "same base (fine-tune, merge or requantization)"
    elif m > 0.5:
        verdict = "related but heavily changed"
    else:
        verdict = "not the same weights"
    out["verdict"] = verdict
    out["verdict_label"] = ESTIMATE
    out["how"] = ("cosine similarity of the same token's row in both files (matched by token "
                  "string), over rows spread across the vocabulary; thresholds 0.98 / 0.5 are "
                  "judgment, the numbers are measured")
    return out


def compare(path_a: str, path_b: str, *, lineage_rows: int = 64) -> dict:
    a, b = gguf.read(path_a), gguf.read(path_b)
    tok = tokenizer_diff(a, b)
    ten = tensor_diff(a, b)
    lin = lineage(a, b, sample=lineage_rows) if tok["same_token_list"] or ten["same_architecture"] \
        else {"verdict": None, "why": "different vocabularies and architectures — not compared"}
    return {
        "kind": "compare",
        "a": file_identity(path_a, a.header_sha256()),
        "b": file_identity(path_b, b.header_sha256()),
        "same_header": a.header_sha256() == b.header_sha256(),
        "tokenizer": tok,
        "metadata": metadata_diff(a, b),
        "tensors": ten,
        "lineage": lin,
    }


def overlap_matrix(paths: list) -> dict:
    """Vocabulary overlap (by token string) between every pair in a library."""
    sets, names, lists = [], [], []
    for p in paths:
        g = gguf.read(p)
        toks = g.get("tokenizer.ggml.tokens") or []
        lists.append(toks)
        sets.append(set(toks))
        names.append(Path(p).name)
    n = len(paths)
    matrix = [[None] * n for _ in range(n)]
    identical = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            u = len(sets[i] | sets[j])
            matrix[i][j] = round(len(sets[i] & sets[j]) / u, 4) if u else None
            identical[i][j] = lists[i] == lists[j]
    return {"kind": "overlap", "label": DECLARED, "models": names, "overlap": matrix,
            "identical_vocabulary": identical,
            "note": "A draft model, a LoRA or a tokenizer transplant needs the SAME vocabulary "
                    "(identical ids), not just overlap."}
