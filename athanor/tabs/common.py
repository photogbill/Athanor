"""Shared by the tabs: names for header values, and the chat-template probes
Inspect and Template both need."""

from __future__ import annotations

import re

# llama_ftype, include/llama.h at b11093 (general.file_type)
FTYPES = {
    0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1", 7: "MOSTLY_Q8_0",
    8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M", 13: "MOSTLY_Q3_K_L", 14: "MOSTLY_Q4_K_S", 15: "MOSTLY_Q4_K_M",
    16: "MOSTLY_Q5_K_S", 17: "MOSTLY_Q5_K_M", 18: "MOSTLY_Q6_K", 19: "MOSTLY_IQ2_XXS",
    20: "MOSTLY_IQ2_XS", 21: "MOSTLY_Q2_K_S", 22: "MOSTLY_IQ3_XS", 23: "MOSTLY_IQ3_XXS",
    24: "MOSTLY_IQ1_S", 25: "MOSTLY_IQ4_NL", 26: "MOSTLY_IQ3_S", 27: "MOSTLY_IQ3_M",
    28: "MOSTLY_IQ2_S", 29: "MOSTLY_IQ2_M", 30: "MOSTLY_IQ4_XS", 31: "MOSTLY_IQ1_M",
    32: "MOSTLY_BF16", 36: "MOSTLY_TQ1_0", 37: "MOSTLY_TQ2_0", 38: "MOSTLY_MXFP4_MOE",
    39: "MOSTLY_NVFP4", 40: "MOSTLY_Q1_0", 41: "MOSTLY_Q2_0", 1024: "GUESSED",
}

# tokenizer.ggml.token_type values (llama_token_type)
TOKEN_TYPES = {0: "undefined", 1: "normal", 2: "unknown", 3: "control",
               4: "user_defined", 5: "unused", 6: "byte"}

# Marker-shaped strings in a chat template. Deliberately broad — every
# candidate is then TESTED against the vocabulary, so a false candidate costs
# one tokenization, never a false finding on its own.
_MARKER_RES = [
    re.compile(r"<\|[^<>|\s{}]{1,48}\|>"),                # <|im_start|>, <|eot_id|>
    re.compile(r"<｜[^<>｜]{1,48}｜>"),                      # DeepSeek's full-width bars
    re.compile(r"<</?[A-Z]{2,20}>>"),                     # <<SYS>>, <</SYS>> (Llama 2)
    re.compile(r"\[/?[A-Z][A-Z0-9_]{1,30}\]"),            # [INST], [/INST], [THINK]
    re.compile(r"(?<!<)</?[a-z][a-z0-9_]{0,30}>(?!>)"),   # <start_of_turn>, <think>, </s>
    re.compile(r"(?<!<)<[A-Z][A-Z0-9_]{1,40}>(?!>)"),     # <BOS_TOKEN>-style
    re.compile(r"\[\|[a-z]{2,20}\|\]"),                   # [|system|] (EXAONE)
]
# HTML-ish words a template's prose may contain that are not markers
_NOT_MARKERS = {"<br>", "<p>", "</p>", "<b>", "</b>", "<i>", "</i>", "<em>", "</em>",
                "<code>", "</code>", "<pre>", "</pre>", "<ul>", "</ul>", "<li>", "</li>",
                "<tools>", "</tools>", "<tool_call>", "</tool_call>"}

# Markers that are ALWAYS special tokens in the models whose format they
# belong to: pipe-delimited forms, SentencePiece's <s>/</s>, Gemma's turn
# markers, Cohere's *_TOKEN forms. If a vocabulary splits one of these, the
# model has never seen it as a unit. Every OTHER marker — [INST], <<SYS>>,
# [SYSTEM_PROMPT], <think> — is ordinary text in some models by design
# (Llama 2, Mistral v0.1–v0.2) and a control token in others (Mistral v0.3+).
_ALWAYS_SPECIAL = re.compile(
    r"^(<\|.*\|>|<｜.*｜>|\[\|.*\|\]|</?s>|<(start|end)_of_turn>|<[A-Z_]+_TOKEN>)$")


def marker_candidates(template_text: str) -> list:
    """Marker-shaped literals in a template, in order of first appearance."""
    seen = []
    for rx in _MARKER_RES:
        for m in rx.finditer(template_text):
            s = m.group(0)
            if s not in seen and s not in _NOT_MARKERS:
                seen.append(s)
    order = {s: template_text.find(s) for s in seen}
    return sorted(seen, key=order.get)


def always_special(marker: str) -> bool:
    return bool(_ALWAYS_SPECIAL.match(marker))


def classify_marker(vocab, marker: str) -> dict:
    """Is this marker ONE special token in this vocabulary?"""
    ids = vocab.tokenize(marker, add_special=False, parse_special=True)
    one = len(ids) == 1
    names = vocab.attr_names(ids[0]) if one else []
    special = one and ("control" in names or "user_defined" in names)
    return {"marker": marker, "ids": ids, "n_tokens": len(ids),
            "special": special, "attrs": names,
            "always_special_style": always_special(marker),
            "eog": bool(one and vocab.is_eog(ids[0]))}


# what C's isspace() counts — the trims in llama-chat.cpp use it
C_WHITESPACE = " \t\n\x0b\x0c\r"


REPLY = "REPLY_MARK_9f3"


def turn_end(template_text: str, *, bos_token: str = "", eos_token: str = "") -> str | None:
    """The text a template writes right after an assistant's reply.

    Render [user, assistant(REPLY)] without a generation prompt and take
    what follows REPLY. That is what the model must produce to end its turn,
    whatever the format. None when the template will not render it.
    """
    from ..templates import render
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": REPLY}]
    try:
        out = render(template_text, msgs, add_generation_prompt=False,
                     bos_token=bos_token, eos_token=eos_token)
    except Exception:
        return None
    i = out.rfind(REPLY)
    if i < 0:
        return None
    return out[i + len(REPLY):]
