"""Template — the prompt microscope (plan §3.4).

Question: what does the model actually see?

A conversation rendered through any template (the model's own, any of the
55, a file, or text), the render tokenized by the model's own tokenizer and
every token classed (BOS, EOS, end-of-generation, control, user-defined,
byte, text). Then the things that go wrong when a template and a model are
mixed and matched:

* a marker the template writes that this vocabulary does not have as ONE
  special token. That is always wrong for pipe-style markers
  (``<|im_start|>``, ``<start_of_turn>``, ``</s>``), whose own models always
  have them as special tokens. For bracket markers like ``[INST]`` it
  depends on the model: Llama 2 and Mistral v0.1–v0.2 were trained on them
  as ordinary text;
* the text the template writes to END a reply, when its first token (after
  the template's whitespace) is not one llama.cpp stops on — replies that
  never end;
* the wrong number of BOS tokens, counted the way each runner builds the
  prompt: llama-cpp-python's chat path, llama.cpp's server, and a naive
  runner.

HOW THE RENDER IS MADE — as the runners make it: the model's BOS and EOS
text go to the template (llama-cpp-python and llama.cpp's server both pass
them whatever the tokenizer's add-BOS setting), and the render is tokenized
with special-token parsing ON and add-special OFF, as llama-cpp-python's chat
path tokenizes it. The BOS check then counts what each runner would send
against what the model expects (one BOS if its tokenizer adds one, else
none).
"""

from __future__ import annotations

import difflib

from .. import gguf, log, templates
from ..labels import MEASURED, Finding
from ..util import file_identity
from .common import C_WHITESPACE, classify_marker, marker_candidates, turn_end


def _special_texts(g, vocab) -> dict:
    """BOS / EOS token text, from llama.cpp if loaded, else the header."""
    out = {"bos": "", "eos": ""}
    if vocab is not None:
        ids = vocab.special_ids()
        for role in ("bos", "eos"):
            if ids.get(role) is not None:
                out[role] = vocab.text(ids[role])
        return out
    if g is not None:
        toks = g.get("tokenizer.ggml.tokens") or []
        for role in ("bos", "eos"):
            i = g.get(f"tokenizer.ggml.{role}_token_id")
            if isinstance(i, int) and 0 <= i < len(toks):
                out[role] = toks[i]
    return out


def classify_tokens(vocab, ids) -> list:
    """Each token of a render with its piece and its kind."""
    special = vocab.special_ids()
    rows = []
    for i in ids:
        names = vocab.attr_names(i)
        if i == special.get("bos"):
            kind = "bos"
        elif i == special.get("eos"):
            kind = "eos"
        elif vocab.is_eog(i):
            kind = "eog"
        elif "control" in names:
            kind = "control"
        elif "user_defined" in names:
            kind = "user_defined"
        elif "byte" in names:
            kind = "byte"
        else:
            kind = "text"
        rows.append({"id": i, "piece": vocab.piece(i).decode("utf-8", "replace"), "kind": kind})
    return rows


def segments(rows) -> list:
    """Runs of plain text merged, so a render reads as markers and text."""
    out = []
    for r in rows:
        if r["kind"] == "text" and out and out[-1]["kind"] == "text":
            out[-1]["text"] += r["piece"]
            out[-1]["n_tokens"] += 1
        else:
            out.append({"kind": r["kind"], "text": r["piece"], "n_tokens": 1,
                        **({"id": r["id"]} if r["kind"] != "text" else {})})
    return out


# Formats whose own models were trained on their bracket markers as ORDINARY
# text, and formats whose own models have them as control tokens.
TEXT_MARKER_FORMATS = {"llama2", "llama2-sys", "llama2-sys-bos", "llama2-sys-strip", "mistral-v1"}
CONTROL_MARKER_FORMATS = {"mistral-v3", "mistral-v3-tekken", "mistral-v7", "mistral-v7-tekken"}


def _adds_bos(g, vocab) -> bool:
    if vocab is not None:
        return vocab.add_bos
    v = g.get("tokenizer.ggml.add_bos_token") if g is not None else None
    return bool(v)


def _leading(ids, tid) -> int:
    n = 0
    for i in ids:
        if i != tid:
            break
        n += 1
    return n


def analyse(spec: str = "gguf", *, model: str | None = None, messages: list | None = None,
            add_generation_prompt: bool = True, vocab=None) -> dict:
    """Render ``spec`` for ``model`` and look at it through the model's tokenizer.

    ``vocab`` may be an open ``Vocab`` (the caller keeps ownership); if not
    given and ``model`` is, one is opened when llama.cpp is available.
    """
    sample = messages is None
    messages = list(messages or templates.SAMPLE_CONVERSATION)
    g = gguf.read(model) if model else None
    t = templates.resolve(spec, gguf=g)
    own_text = (templates.model_templates(g).get("default") or "") if g is not None else ""
    findings = []
    own_vocab = False
    if vocab is None and model is not None:
        from ..vocab import LlamaUnavailable, Vocab
        try:
            vocab = Vocab(model)
            own_vocab = True
        except LlamaUnavailable as exc:
            findings.append(Finding("tokenizer", "skipped",
                                    f"the render was not tokenized — {exc}", MEASURED))
        except Exception as exc:
            findings.append(Finding("tokenizer", "problem",
                                    f"llama.cpp could not load this model's vocabulary: {exc}",
                                    MEASURED))
    try:
        sp = _special_texts(g, vocab)
        adds_bos = _adds_bos(g, vocab)
        bos_passed = sp["bos"]
        result = {
            "kind": "template",
            "template": {k: t[k] for k in ("name", "source", "label")},
            "template_chars": len(t["text"]),
            "detected_format": templates.detect(t["text"]) or None,
            "model": file_identity(model, g.header_sha256()) if model else None,
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": sp["bos"], "eos_token": sp["eos"],
            "bos_token_passed": bos_passed,
        }
        # the format the markers belong to: a library template's own name (the
        # detector reads markers, and calls several library texts 'llama2'),
        # else what llama.cpp's detector says
        det = result["detected_format"]
        fmt = t["name"].split("/", 1)[1] if t["source"] == "library" else det
        result["format"] = fmt
        findings.append(Finding(
            "detected-format", "info",
            f"llama.cpp would call this template '{det}'" if det
            else "llama.cpp's detector does not recognise this template's markers",
            MEASURED, {"format": det}))

        def _render(msgs):
            return templates.render(t["text"], msgs, add_generation_prompt=add_generation_prompt,
                                    bos_token=bos_passed, eos_token=sp["eos"])
        try:
            text = _render(messages)
        except Exception as exc:
            has_system = any(m.get("role") == "system" for m in messages)
            text = None
            if sample and has_system:
                # many templates (Gemma 2, old Mistral) refuse a system role
                # on purpose; that is a property, not a fault — try without
                try:
                    messages = [m for m in messages if m.get("role") != "system"]
                    text = _render(messages)
                    result["messages"] = messages
                    findings.append(Finding(
                        "renders", "info",
                        f"this template refuses a system message ({exc}); rendered without one",
                        MEASURED, {"error": str(exc)}))
                except Exception:
                    text = None
            if text is None:
                result["render"] = None
                findings.append(Finding("renders", "problem",
                                        f"the template will not render this conversation: {exc}",
                                        MEASURED, {"error": str(exc)}))
                result["findings"] = [f.to_dict() for f in findings]
                return result
        result["render"] = text
        result["render_chars"] = len(text)
        markers = marker_candidates(t["text"])
        end = turn_end(t["text"], bos_token=bos_passed, eos_token=sp["eos"])
        result["turn_end"] = end
        if vocab is None:
            result["tokens"] = None
            result["markers"] = [{"marker": m} for m in markers]
            result["findings"] = [f.to_dict() for f in findings]
            return result

        log.breadcrumb("template-tokenize", model=str(model), template=t["name"],
                       render_chars=len(text))
        ids = vocab.tokenize(text, add_special=False, parse_special=True)
        rows = classify_tokens(vocab, ids)
        result["n_tokens"] = len(ids)
        result["tokens"] = rows
        result["segments"] = segments(rows)

        # 1. markers that are not one special token here
        classified = [classify_marker(vocab, m) for m in markers]
        result["markers"] = classified
        in_render = [c for c in classified if c["marker"] in text]
        bad = 0
        for c in in_render:
            if c["special"]:
                continue
            if c["n_tokens"] == 1:
                findings.append(Finding(
                    "marker-is-special", "warn",
                    f"'{c['marker']}' is one token here, but an ordinary one, not a control token",
                    MEASURED, {"marker": c["marker"], "ids": c["ids"]}))
                bad += 1
            elif c["always_special_style"]:
                findings.append(Finding(
                    "marker-is-special", "problem",
                    f"'{c['marker']}' is split into {c['n_tokens']} tokens here — markers of this "
                    "kind are always special tokens in their own models, so this model has never "
                    "seen it as a unit", MEASURED, {"marker": c["marker"], "ids": c["ids"]}))
                bad += 1
            elif fmt in CONTROL_MARKER_FORMATS:
                findings.append(Finding(
                    "marker-is-special", "problem",
                    f"'{c['marker']}' is ordinary text here ({c['n_tokens']} tokens), but the "
                    f"'{fmt}' format's own models have it as a control token", MEASURED,
                    {"marker": c["marker"], "ids": c["ids"]}))
                bad += 1
            elif fmt in TEXT_MARKER_FORMATS:
                findings.append(Finding(
                    "marker-is-special", "info",
                    f"'{c['marker']}' is ordinary text here ({c['n_tokens']} tokens), as the "
                    f"'{fmt}' format was trained", MEASURED,
                    {"marker": c["marker"], "ids": c["ids"]}))
            elif c["marker"] in own_text:
                continue  # plain text, and the model's own template writes it the same way
            else:
                findings.append(Finding(
                    "marker-is-special", "warn",
                    f"'{c['marker']}' is ordinary text here ({c['n_tokens']} tokens) — right for "
                    "models trained that way (Llama 2, Mistral v0.1–v0.2), wrong for ones trained "
                    "with it as a control token (Mistral v0.3 and later); check the model card",
                    MEASURED, {"marker": c["marker"], "ids": c["ids"]}))
                bad += 1
        if in_render and not bad:
            findings.append(Finding(
                "marker-is-special", "ok",
                f"all {len(in_render)} markers in the render are as this model knows them",
                MEASURED, {"markers": [c["marker"] for c in in_render]}))

        # 2. does the end of a reply stop generation?
        if end is None:
            findings.append(Finding("turn-end", "skipped",
                                    "could not render a finished reply to find its end", MEASURED))
        elif not end.strip(C_WHITESPACE):
            findings.append(Finding(
                "turn-end", "info",
                "the template writes nothing after a reply — generation stops on the model's "
                "own end-of-generation token", MEASURED, {"turn_end": end}))
        else:
            # the template's own spacing before the marker is formatting, not
            # something the model must produce
            core = end.lstrip(C_WHITESPACE)
            end_ids = vocab.tokenize(core, add_special=False, parse_special=True)
            first = end_ids[0] if end_ids else None
            stops = first is not None and vocab.is_eog(first)
            ev = {"turn_end": end, "first_token": first,
                  "first_piece": vocab.piece(first).decode("utf-8", "replace")
                  if first is not None else None}
            findings.append(Finding(
                "turn-end", "ok" if stops else "problem",
                (f"a reply ends with '{ev['first_piece']}', which llama.cpp stops on")
                if stops else
                (f"a reply ends with '{ev['first_piece']}' (token {first}), which llama.cpp does "
                 "NOT treat as end-of-generation — replies may run on past their end"),
                MEASURED, ev))

        # 3. BOS, counted the way each runner builds the prompt
        bos_id = vocab.special_ids().get("bos")
        expected = 1 if (adds_bos and bos_id is not None) else 0
        # llama-cpp-python: the render as it is, no special tokens added
        binding = _leading(ids, bos_id) if bos_id is not None else 0
        # llama.cpp's server (common/chat.cpp): strips a leading BOS text when
        # it will add BOS itself, then tokenizes with add-special
        server_text = text[len(sp["bos"]):] if adds_bos and sp["bos"] and text.startswith(sp["bos"]) else text
        with vocab.capture() as said:
            server = _leading(vocab.tokenize(server_text, add_special=True, parse_special=True), bos_id) \
                if bos_id is not None else 0
            # a naive runner: the render plus whatever add-special adds
            naive = _leading(vocab.tokenize(text, add_special=True, parse_special=True), bos_id) \
                if bos_id is not None else 0
        result["bos"] = {
            "id": bos_id, "tokenizer_adds": adds_bos, "expected": expected,
            "llama_cpp_python": binding, "llama_server": server, "naive_runner": naive,
            "llama_cpp_said": " ".join(x.strip() for _l, x in said if x.strip()),
        }
        if bos_id is None:
            pass
        elif binding == expected and server == expected:
            findings.append(Finding(
                "bos-once", "ok",
                "BOS appears once, as this model expects" if expected else
                "no BOS, as this model is configured", MEASURED, result["bos"]))
        else:
            parts = []
            if binding != expected:
                parts.append(f"llama-cpp-python's chat path would send {binding}")
            if server != expected:
                parts.append(f"llama.cpp's server would send {server}")
            findings.append(Finding(
                "bos-once", "warn",
                f"BOS count wrong for this model (expects {expected}): " + "; ".join(parts),
                MEASURED, result["bos"]))
        if bos_id is not None and naive != expected and binding == expected and server == expected:
            findings.append(Finding(
                "bos-naive", "info",
                f"a runner that adds special tokens without checking the render would send "
                f"{naive} BOS (llama.cpp's server and llama-cpp-python both avoid this)",
                MEASURED, result["bos"]))
        result["findings"] = [f.to_dict() for f in findings]
        return result
    finally:
        if own_vocab and vocab is not None:
            vocab.close()


def side_by_side(spec_a: str, spec_b: str, *, model: str | None = None,
                 messages: list | None = None, add_generation_prompt: bool = True) -> dict:
    """Two templates, one conversation: both renders and where they differ."""
    vocab = None
    if model:
        try:
            from ..vocab import Vocab
            vocab = Vocab(model)
        except Exception:
            vocab = None
    try:
        a = analyse(spec_a, model=model, messages=messages,
                    add_generation_prompt=add_generation_prompt, vocab=vocab)
        b = analyse(spec_b, model=model, messages=messages,
                    add_generation_prompt=add_generation_prompt, vocab=vocab)
    finally:
        if vocab is not None:
            vocab.close()
    ra, rb = a.get("render") or "", b.get("render") or ""
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ra, rb, autojunk=False).get_opcodes():
        if tag != "equal":
            ops.append({"op": tag, "a": ra[i1:i2], "b": rb[j1:j2], "a_at": i1, "b_at": j1})
    return {"kind": "template-compare", "a": a, "b": b, "identical": ra == rb,
            "differences": ops,
            "token_difference": (b.get("n_tokens") or 0) - (a.get("n_tokens") or 0)
            if a.get("n_tokens") is not None and b.get("n_tokens") is not None else None}


def library() -> list:
    """The 55 templates: name, label, note."""
    return [{"name": t.name, "label": t.label, "note": t.note} for t in templates.all_templates()]
