"""The text of tokens the recording never saw.

A recording keeps the text of every token among its candidates, so it can
be read without the model. The lens names tokens the final distribution
never ranked — what layer 20 "would say" is often a word nowhere near the
answer — and those need looking up. Through llama.cpp's own tokenizer when
the binding is here (exact, ``athanor.vocab.Vocab``, vocab-only load),
else from the GGUF's token list, decoded the way llama.cpp would: GPT-2's
byte-to-character table undone for byte-pair vocabularies, SentencePiece's
``▁`` and ``<0xNN>`` for the rest.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _gpt2_chars_to_bytes() -> dict:
    """GPT-2's ``bytes_to_unicode``, inverted: the character a byte is
    written as in a BPE vocabulary → the byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}


def _decode_token(text: str, model: str, token_type: int | None) -> str:
    """One entry of ``tokenizer.ggml.tokens`` as the text it stands for."""
    if token_type in (3, 4):                   # control, user-defined: as written
        return text
    if model == "gpt2":                        # byte-pair vocabularies (Qwen, Llama 3, Mistral)
        table = _gpt2_chars_to_bytes()
        try:
            return bytes(table[ch] for ch in text).decode("utf-8", "backslashreplace")
        except KeyError:
            return text
    if text.startswith("<0x") and text.endswith(">") and len(text) == 6:
        try:
            return bytes([int(text[3:5], 16)]).decode("utf-8", "backslashreplace")
        except ValueError:
            return text
    return text.replace("▁", " ")


def resolve(model_path, ids, *, prefer_llama: bool = True) -> tuple[dict, str]:
    """({id: text}, how) for the given token ids. ``how`` is ``"llama.cpp"``
    when llama.cpp's tokenizer gave them, ``"the file's token list"`` when
    they were decoded from the GGUF, or a reason nothing could be resolved."""
    ids = sorted({int(t) for t in ids})
    if not ids or not model_path:
        return {}, "nothing to resolve" if not ids else "no model file"
    path = Path(model_path)
    if not path.is_file():
        return {}, f"the model file is not at {path}"
    if prefer_llama:
        try:
            from ..vocab import Vocab
            with Vocab(path) as v:
                out = {}
                for t in ids:
                    if 0 <= t < v.n_tokens:
                        out[t] = v.piece(t).decode("utf-8", "backslashreplace")
                return out, "llama.cpp"
        except Exception:              # noqa: BLE001 — no binding here: read the file
            pass
    try:
        from ..gguf import read as read_gguf
        g = read_gguf(path)
        tokens = g.get("tokenizer.ggml.tokens")
        types = g.get("tokenizer.ggml.token_type")
        model = g.get("tokenizer.ggml.model") or "llama"
    except Exception as exc:           # noqa: BLE001
        return {}, f"the file's token list could not be read ({type(exc).__name__}: {exc})"
    if not tokens:
        return {}, "the file has no token list"
    out = {}
    n = len(tokens)
    for t in ids:
        if 0 <= t < n:
            tt = int(types[t]) if types is not None and t < len(types) else None
            out[t] = _decode_token(str(tokens[t]), str(model), tt)
    return out, "the file's token list"


__all__ = ["resolve"]
