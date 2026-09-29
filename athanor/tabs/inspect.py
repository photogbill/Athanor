"""Inspect — a model file's anatomy and its tokenizer's health (plan §3.1).

Question: what is this file, really — and will its tokenizer and template
work together?

Everything the header declares is labelled DECLARED. Everything llama.cpp
was run to find out is MEASURED. Arithmetic on either is ESTIMATE.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .. import gguf, templates
from ..gguf.read import BLOCK_TYPES
from ..labels import DECLARED, ESTIMATE, MEASURED, Figure, Finding
from ..util import file_identity, human_bytes
from .common import FTYPES, TOKEN_TYPES

_LAYER_RE = re.compile(r"^blk\.(\d+)\.")
_CTX_IN_NAME = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([kKmM])(?:[-_ ]?(?:ctx|context|tok))?\b")


# llama.cpp's banner around the missing-pre-tokenizer warning; the warning
# itself is kept, the decoration is not
_BANNER = {"GENERATION QUALITY WILL BE DEGRADED!", "CONSIDER REGENERATING THE MODEL"}


def _fig(v, label, unit=None, how=None):
    return Figure(v, label, unit, how).to_dict()


def anatomy(g) -> dict:
    """What the header says the model is, and what its tensors add up to."""
    arch = g.architecture
    ag = g.arch_get
    n_params = sum(t.n_elements for t in g.tensors)
    known = [t for t in g.tensors if t.nbytes is not None]
    data_bytes = sum(t.nbytes for t in known)
    by_type = Counter()
    bytes_by_type = Counter()
    for t in g.tensors:
        by_type[t.type_name] += 1
        bytes_by_type[t.type_name] += t.nbytes or 0
    layers = defaultdict(int)
    other = []
    for t in g.tensors:
        m = _LAYER_RE.match(t.name)
        if m:
            layers[int(m.group(1))] += 1
        else:
            other.append(t.name)
    ft = g.get("general.file_type")
    head_count = ag("attention.head_count")
    head_kv = ag("attention.head_count_kv", head_count)
    n_embd = ag("embedding_length")
    n_layer = ag("block_count")
    key_len = ag("attention.key_length")
    val_len = ag("attention.value_length")
    if isinstance(head_count, (list, tuple)) or hasattr(head_count, "tolist"):
        head_count = None  # per-layer arrays: leave the estimate out
    if isinstance(head_kv, (list, tuple)) or hasattr(head_kv, "tolist"):
        head_kv = None
    kv_per_token = None
    if n_layer and head_kv and (key_len or (n_embd and head_count)):
        kl = key_len or n_embd // head_count
        vl = val_len or kl
        kv_per_token = int(n_layer) * int(head_kv) * (int(kl) + int(vl)) * 2
    out = {
        "architecture": _fig(arch, DECLARED),
        "name": _fig(g.get("general.name"), DECLARED),
        "basename": _fig(g.get("general.basename"), DECLARED),
        "finetune": _fig(g.get("general.finetune"), DECLARED),
        "size_label": _fig(g.get("general.size_label"), DECLARED),
        "license": _fig(g.get("general.license"), DECLARED),
        "file_type": _fig(FTYPES.get(ft, ft) if ft is not None else None, DECLARED),
        "context_length": _fig(ag("context_length"), DECLARED, "tokens",
                               "what the header says the model was trained for"),
        "embedding_length": _fig(n_embd, DECLARED),
        "block_count": _fig(n_layer, DECLARED),
        "head_count": _fig(head_count, DECLARED),
        "head_count_kv": _fig(head_kv, DECLARED),
        "expert_count": _fig(ag("expert_count"), DECLARED),
        "rope_freq_base": _fig(ag("rope.freq_base"), DECLARED),
        "parameters": _fig(n_params, DECLARED, None, "counted from the tensor shapes in the header"),
        "tensor_bytes": _fig(data_bytes, DECLARED, "bytes"),
        "bits_per_weight": _fig(round(data_bytes * 8 / n_params, 3) if n_params else None,
                                ESTIMATE, "bits", "tensor bytes × 8 ÷ parameters"),
        "kv_cache_per_token": _fig(kv_per_token, ESTIMATE, "bytes",
                                   "layers × KV heads × (key + value length) × 2 bytes (f16 cache)"),
        "tensor_types": [{"type": k, "tensors": by_type[k], "bytes": bytes_by_type[k]}
                         for k in sorted(by_type, key=lambda k: -bytes_by_type[k])],
        "layers_with_tensors": len(layers),
        "tensors_per_layer": sorted(set(layers.values())),
        "non_layer_tensors": other,
    }
    return out


def tokenizer_declared(g) -> dict:
    tokens = g.get("tokenizer.ggml.tokens") or []
    types = g.get("tokenizer.ggml.token_type")
    hist = {}
    if types is not None and len(types):
        c = Counter(int(x) for x in types)
        hist = {TOKEN_TYPES.get(k, str(k)): v for k, v in sorted(c.items())}
    specials = {}
    for key in g.keys():
        if key.startswith("tokenizer.ggml.") and key.endswith("_token_id"):
            role = key[len("tokenizer.ggml."):-len("_token_id")]
            tid = g.get(key)
            specials[role] = {"id": tid, "text": tokens[tid] if isinstance(tid, int)
                              and 0 <= tid < len(tokens) else None}
    own = templates.model_templates(g)
    return {
        "model": _fig(g.get("tokenizer.ggml.model"), DECLARED),
        "pre": _fig(g.get("tokenizer.ggml.pre"), DECLARED),
        "n_tokens": _fig(len(tokens), DECLARED),
        "n_merges": _fig(len(g.get("tokenizer.ggml.merges") or []), DECLARED),
        "has_scores": "tokenizer.ggml.scores" in g,
        "token_types": hist,
        "special_ids": specials,
        "add_bos": _fig(g.get("tokenizer.ggml.add_bos_token"), DECLARED),
        "add_eos": _fig(g.get("tokenizer.ggml.add_eos_token"), DECLARED),
        "chat_templates": {name: {"chars": len(text),
                                  "detected_format": templates.detect(text) or None}
                           for name, text in own.items()},
    }


def tokenizer_measured(vocab) -> dict:
    ids = vocab.special_ids()
    eog = vocab.eog_ids()
    return {
        "type": _fig(vocab.type, MEASURED, None, "llama.cpp's vocab type"),
        "n_tokens": _fig(vocab.n_tokens, MEASURED),
        "special_ids": {k: {"id": v, "text": vocab.text(v) if v is not None else None}
                        for k, v in ids.items() if v is not None},
        "end_of_generation": [{"id": i, "text": vocab.text(i)} for i in eog[:64]],
        "n_end_of_generation": len(eog),
        "add_bos": _fig(vocab.add_bos, MEASURED),
        "add_eos": _fig(vocab.add_eos, MEASURED),
        "load_seconds": vocab.load_time().to_dict(),
        "llama_cpp_warnings": [t for _lvl, t in vocab.log_lines()],
    }


def _num_in_name(text: str):
    """Context lengths a filename claims — '128k', '1M', '32K-ctx'."""
    out = []
    for m in _CTX_IN_NAME.finditer(text or ""):
        value, unit = float(m.group(1)), m.group(2)
        if unit in "mM" and value > 10:
            continue  # "70m", "135M" are parameter counts, not contexts
        n = value * (1024 if unit in "kK" else 1024 * 1024)
        if n >= 2048:
            out.append((m.group(0), int(n)))
    return out


def _size_label_params(label: str):
    m = re.fullmatch(r"\s*(?:(\d+)x)?(\d+(?:\.\d+)?)\s*([BbMm])\s*", str(label or ""))
    if not m:
        return None
    n = float(m.group(2)) * (1e9 if m.group(3) in "Bb" else 1e6)
    return n * (int(m.group(1)) if m.group(1) else 1)


def is_embedder(g) -> bool:
    """An embedding model (it pools; it does not generate)."""
    arch = (g.architecture or "").lower()
    return (g.arch_get("pooling_type") is not None
            or arch in {"bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3",
                        "t5encoder", "neo-bert", "modern-bert", "eurobert"})


def checks(g, vocab=None, *, llama_tried: bool = False) -> list:
    """The health checks. Each returns a Finding; none guesses."""
    f = []
    # structure
    problems = gguf.validate(g)
    if problems:
        for p in problems:
            f.append(Finding("structure", "problem", p, MEASURED))
    elif g.tensors:
        f.append(Finding("structure", "ok", f"{len(g.tensors)} tensors, aligned, in the file, "
                         "not overlapping", MEASURED))
    rt = gguf.round_trip_check(g)
    f.append(Finding("header-round-trip", "ok" if rt["identical"] else "warn",
                     "Athanor's writer reproduces this header byte for byte" if rt["identical"]
                     else f"Athanor's writer differs from this header at byte {rt['first_difference']}",
                     MEASURED, rt))
    unknown = sorted({t.type_name for t in g.tensors if t.ggml_type not in BLOCK_TYPES})
    if unknown:
        f.append(Finding("tensor-types", "warn", "tensor types unknown to llama.cpp "
                         f"b11093: {', '.join(unknown)}", DECLARED))

    model = g.get("tokenizer.ggml.model")
    pre = g.get("tokenizer.ggml.pre")
    tokens = g.get("tokenizer.ggml.tokens") or []
    is_projector = g.get("general.type") == "mmproj" or any(k.startswith("clip.") for k in g.keys())
    embedder = is_embedder(g)

    # pre-tokenizer (llama-vocab.cpp: an EMPTY one warns and falls back to
    # 'default'; an explicit 'default' is accepted silently)
    if model == "gpt2" and not pre:
        f.append(Finding("pre-tokenizer", "problem",
                         "a BPE tokenizer with no pre-tokenizer named — llama.cpp falls back to "
                         "'default' and warns that generation quality will be degraded",
                         DECLARED, {"tokenizer.ggml.pre": pre}))
    elif model == "gpt2" and pre == "default":
        f.append(Finding("pre-tokenizer", "info",
                         "pre-tokenizer set explicitly to 'default' — llama.cpp accepts it; right "
                         "only if the model was trained with that pre-tokenization",
                         DECLARED, {"tokenizer.ggml.pre": pre}))
    elif model:
        f.append(Finding("pre-tokenizer", "ok" if pre else "info",
                         f"pre-tokenizer '{pre}'" if pre else f"{model} tokenizer; no pre-tokenizer "
                         "key (normal for this type)", DECLARED, {"tokenizer.ggml.pre": pre}))

    # vocabulary vs embedding and output rows: llama.cpp creates these with
    # exactly one row per token and refuses a file whose shapes differ
    # (llama-model-loader.cpp, check_tensor_dims)
    for tname, what in (("token_embd.weight", "embedding"), ("output.weight", "output")):
        t = g.tensor(tname)
        if t is None or not tokens or len(t.shape) < 2:
            continue
        rows = int(t.shape[1])
        if rows == len(tokens):
            f.append(Finding("vocab-rows", "ok", f"{rows} {what} rows for {len(tokens)} tokens",
                             DECLARED, {"tensor": tname}))
        else:
            f.append(Finding("vocab-rows", "problem",
                             f"{rows} {what} rows for {len(tokens)} tokens — llama.cpp needs one row "
                             "per token and will refuse to load this file", DECLARED,
                             {"tensor": tname, "rows": rows, "tokens": len(tokens)}))

    # the name against the header
    ctx = g.arch_get("context_length")
    names = " ".join(str(x) for x in (g.path.name, g.get("general.name") or ""))
    for text, n in _num_in_name(names):
        if isinstance(ctx, int) and abs(n - ctx) / max(ctx, 1) > 0.1:
            f.append(Finding("name-vs-header", "warn",
                             f"the name says '{text}' but the header declares a context of {ctx:,} "
                             "tokens (and neither says what your card can hold — the Context tab "
                             "measures that)", DECLARED, {"name": text, "context_length": ctx}))
    label_n = _size_label_params(g.get("general.size_label"))
    n_params = sum(t.n_elements for t in g.tensors)
    if label_n and n_params and abs(n_params - label_n) / label_n > 0.2 and not is_projector:
        f.append(Finding("name-vs-header", "info",
                         f"size label {g.get('general.size_label')!r} vs {n_params / 1e9:.2f}B "
                         "parameters counted in the header", DECLARED))

    # the model's own chat template
    own = templates.model_templates(g)
    if embedder:
        f.append(Finding("embedder", "info",
                         "an embedding model — it does not generate, so the chat-template and "
                         "end-of-generation checks do not apply", DECLARED,
                         {"pooling_type": g.arch_get("pooling_type")}))
    elif not own and tokens and not is_projector:
        f.append(Finding("chat-template", "warn",
                         "the file carries no chat template — a runner will fall back to a default "
                         "or its own guess (Mistral's official GGUFs are like this); choose one in "
                         "the Template tab", DECLARED))
    elif own:
        t = own.get("default") or next(iter(own.values()))
        uses_bos = "bos_token" in t
        f.append(Finding("chat-template", "info",
                         f"{len(own)} chat template(s); the default looks like llama.cpp's "
                         f"'{templates.detect(t) or 'unrecognised'}'", DECLARED,
                         {"names": list(own), "writes_bos": uses_bos}))

    # llama.cpp's own view
    if vocab is None:
        if tokens and not llama_tried:
            f.append(Finding("llama-cpp", "skipped",
                             "llama.cpp was not run, so the checks that need it were skipped",
                             MEASURED))
        return f
    for lvl, text in vocab.log_lines():
        body = re.sub(r"^\w+:\s*", "", text.strip()).strip()
        if not body.strip("* ") or body in _BANNER:
            continue
        f.append(Finding("llama-cpp-log", "problem" if lvl == "error" else "warn",
                         f"llama.cpp {lvl}: {body}", MEASURED, {"line": text.strip()}))
    if vocab.n_tokens != len(tokens):
        f.append(Finding("vocab-size", "warn",
                         f"llama.cpp sees {vocab.n_tokens} tokens; the header lists {len(tokens)}",
                         MEASURED))
    ids = vocab.special_ids()
    if ids.get("eos") is None and not embedder:
        f.append(Finding("eos", "problem", "llama.cpp found no EOS token", MEASURED))
    if own and not embedder:
        from .template import analyse
        a = analyse("gguf", model=str(g.path), vocab=vocab)
        for x in a["findings"]:
            if x["check"] in ("marker-is-special", "turn-end", "bos-once", "renders", "tokenizer"):
                x = dict(x)
                x["check"] = "own-template/" + x["check"]
                f.append(Finding(**{k: x[k] for k in ("check", "status", "message", "label",
                                                        "evidence")}))
    return f


def projector_pairing(text_g, proj_g) -> list:
    """Does this mmproj belong to this text model?"""
    out = []
    n_embd = text_g.arch_get("embedding_length")
    proj_dim = proj_g.get("clip.vision.projection_dim") or proj_g.get("clip.projection_dim")
    if proj_dim is None:
        mm = [t for t in proj_g.tensors if t.name.startswith("mm.") and len(t.shape) == 2]
        if mm:
            proj_dim = int(mm[-1].shape[1])
    if n_embd and proj_dim:
        out.append(Finding("projector-width", "ok" if int(proj_dim) == int(n_embd) else "problem",
                           f"projector output {proj_dim} vs text model embedding {n_embd}",
                           DECLARED, {"projection": proj_dim, "embedding_length": n_embd}))
    else:
        out.append(Finding("projector-width", "skipped",
                           "could not read both widths from the headers", DECLARED))
    tn = (text_g.get("general.basename") or text_g.get("general.name") or "").lower()
    pn = (proj_g.get("general.basename") or proj_g.get("general.name") or "").lower()
    if tn and pn:
        same = tn.split()[0] in pn or pn.split()[0] in tn
        out.append(Finding("projector-name", "ok" if same else "info",
                           f"text model '{tn}', projector '{pn}'", DECLARED))
    return out


def inspect(path: str, *, run_llama: bool = True, full_hash: bool = False,
            projector: str | None = None) -> dict:
    """Inspect one GGUF. ``run_llama`` loads the vocabulary through llama.cpp
    (no weights, no GPU) for the checks that need it."""
    g = gguf.read(path)
    vocab = None
    load_problem = None
    if run_llama and (g.get("tokenizer.ggml.tokens") is not None):
        from ..vocab import LlamaUnavailable, Vocab
        try:
            vocab = Vocab(path)
        except LlamaUnavailable as exc:
            run_llama = False
            load_problem = Finding("llama-cpp", "skipped",
                                   f"the checks that need llama.cpp were skipped — {exc}", MEASURED)
        except Exception as exc:
            load_problem = Finding("llama-cpp", "problem",
                                   f"llama.cpp could not load this file's vocabulary: {exc}",
                                   MEASURED, {"log": getattr(exc, "log", None)})
    try:
        result = {
            "kind": "inspect",
            "file": file_identity(path, g.header_sha256(), g.file_sha256() if full_hash else None),
            "gguf": g.summary(),
            "anatomy": anatomy(g),
            "tokenizer": {"declared": tokenizer_declared(g),
                          "measured": tokenizer_measured(vocab) if vocab else None},
        }
        findings = checks(g, vocab, llama_tried=load_problem is not None)
        if load_problem is not None:
            findings.append(load_problem)
        if projector:
            pg = gguf.read(projector)
            result["projector"] = file_identity(projector, pg.header_sha256())
            findings.extend(projector_pairing(g, pg))
        order = {"problem": 0, "warn": 1, "info": 2, "skipped": 3, "ok": 4}
        findings.sort(key=lambda x: order[x.status])
        result["findings"] = [x.to_dict() for x in findings]
        result["summary"] = dict(Counter(x.status for x in findings))
        result["size_human"] = human_bytes(g.file_size)
        return result
    finally:
        if vocab is not None:
            vocab.close()


def metadata(path: str, *, max_items: int = 16) -> list:
    """Every key, its type and a preview (arrays shortened for display)."""
    g = gguf.read(path)
    out = []
    for kv in g.kvs:
        v = kv.value
        if isinstance(v, gguf.Array):
            items = v.tolist()
            preview = items[:max_items]
            out.append({"key": kv.key, "type": f"array[{v.elem_type.name.lower()}]",
                        "length": len(items), "preview": preview})
        else:
            pv = kv.plain()
            if isinstance(pv, str) and len(pv) > 400:
                pv = pv[:400] + f"… ({len(kv.plain())} chars)"
            out.append({"key": kv.key, "type": kv.type.name.lower(), "value": pv})
    return out
