"""Tokenize — the context ruler (plan §3.2).

Question: how much of MY material fits, on which model?

Every model's own tokenizer (vocab-only, no weights) on the same text, at
once: tokens, the pieces, tokens per word and per character, by script;
how many words of this material a context holds; and the analyst's own
strings that split worst.
"""

from __future__ import annotations

import os
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from .. import log
from ..labels import ESTIMATE, MEASURED, Figure
from ..util import file_identity

# Word characters: letters, marks and digits (Unicode categories L*, M*, N*)
# plus the joiners. Python's \w does NOT match combining marks, so it would
# split عَبْدُ at every vowel sign and हिन्दी at every matra — the words
# are built here from categories instead.
_JOINERS = {"\u200c", "\u200d", "_"}
_INNER = {"-", "'", "’"}


def _is_word_char(c: str) -> bool:
    return c in _JOINERS or unicodedata.category(c)[0] in "LMN"


def iter_words(text: str):
    """Words: runs of letters, marks, digits and joiners; a hyphen or an
    apostrophe BETWEEN two word characters stays inside the word."""
    n = len(text)
    i = 0
    while i < n:
        if not _is_word_char(text[i]):
            i += 1
            continue
        j = i + 1
        while j < n:
            c = text[j]
            if _is_word_char(c):
                j += 1
            elif c in _INNER and j + 1 < n and _is_word_char(text[j + 1]):
                j += 2
            else:
                break
        yield text[i:j]
        i = j

#: Synthetic examples of the strings an analyst's material is full of — the
#: ones a tokenizer tends to shred. Invented values, no real people or
#: accounts. Replace or extend with your own (``strings=``).
ANALYST_STRINGS = [
    ("IPv4 address", "10.0.0.5"),
    ("IPv4 address", "192.168.214.254"),
    ("IPv6 address", "2001:db8::ff00:42:8329"),
    ("MAC address", "00:1A:2B:3C:4D:5E"),
    ("MGRS grid", "18SUJ2337106519"),
    ("lat/long", "38.8977° N, 77.0365° W"),
    ("callsign", "KD2ABC"),
    ("frequency", "462.5625 MHz"),
    ("timestamp", "2026-09-28T16:19:00Z"),
    ("phone number", "+93 70 123 4567"),
    ("email", "j.doe@example.org"),
    ("URL", "https://example.org/case/0142?ref=a7"),
    ("Windows path", r"C:\Users\analyst\Desktop\case_014.docx"),
    ("SHA-256 (prefix)", "9f86d081884c7d659a2feaa0c55ad015"),
    ("hex value", "0xDEADBEEF"),
    ("base64", "U29tZSBjYXB0dXJlZCB0ZXh0"),
    ("transliterated name", "Abd al-Rahman ibn Khalid"),
    ("transliterated name", "Farid Ahmad Sultani"),
    ("Arabic-script name", "عبد الرحمن"),
    ("Arabic-script name, voweled", "عَبْدُ الرَّحْمٰن"),
    ("Pashto", "زه په کندهار کې اوسېږم"),
    ("Dari", "من در کابل زندگی می‌کنم"),
    ("Russian", "Встреча перенесена на четверг"),
    ("Chinese", "会议改到星期四"),
]


#: Scripts written without spaces between words (a run of letters is a
#: phrase, not a word). CJK is handled separately: one ideograph, one word.
NO_SPACE_SCRIPTS = {"THAI", "LAO", "KHMER", "MYANMAR", "TIBETAN"}


def script_of(ch: str) -> str:
    """The script of a letter, by its Unicode name ('LATIN', 'ARABIC', …)."""
    if not ch.isalpha():
        return "OTHER"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "OTHER"
    first = name.split(" ")[0]
    if first in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL"):
        return first
    return first


def words_by_script(text: str) -> dict:
    """{script: [words]} — each word assigned by its first letter."""
    out = defaultdict(list)
    for w in iter_words(text):
        letter = next((c for c in w if c.isalpha()), None)
        out[script_of(letter) if letter else "DIGITS"].append(w)
    return dict(out)


def _count_words(text: str) -> int:
    """Words as a reader counts them. CJK has no spaces: each ideograph
    counts as one word, so the figure is comparable across scripts."""
    n = 0
    for w in iter_words(text):
        cjk = sum(1 for c in w if script_of(c) in ("CJK", "HIRAGANA", "KATAKANA"))
        n += cjk if cjk else 1
    return n


def _open_all(paths):
    from ..vocab import Vocab
    opened = []
    try:
        for p in paths:
            opened.append(Vocab(p))
    except Exception:
        for v in opened:
            v.close()
        raise
    return opened


def measure(vocab, text: str, *, show_tokens: int = 200) -> dict:
    """One tokenizer on one text."""
    t0 = time.perf_counter()
    ids = vocab.tokenize(text, add_special=False, parse_special=False)
    secs = time.perf_counter() - t0
    words = _count_words(text)
    chars = len(text)
    nbytes = len(text.encode("utf-8"))
    by_script = {}
    notes = []
    for script, ws in words_by_script(text).items():
        joined = " ".join(ws)
        n = len(vocab.tokenize(joined, add_special=False, parse_special=False))
        chars = sum(len(w) for w in ws)
        if script in NO_SPACE_SCRIPTS:
            # written without spaces between words: a "word" here is a whole
            # phrase, so only characters are a fair unit
            by_script[script] = {"words": None, "chars": chars, "tokens": n,
                                 "tokens_per_word": None,
                                 "tokens_per_char": round(n / chars, 3) if chars else None}
            notes.append(f"{script.title()} is written without spaces between words; its word "
                         "count is not meaningful — use characters")
            continue
        wcount = _count_words(joined)
        by_script[script] = {"words": wcount, "chars": chars, "tokens": n,
                             "tokens_per_word": round(n / wcount, 3) if wcount else None,
                             "tokens_per_char": round(n / chars, 3) if chars else None}
    shown = [{"id": i, "piece": vocab.piece(i, special=False).decode("utf-8", "replace")}
             for i in ids[:show_tokens]]
    return {
        "n_tokens": Figure(len(ids), MEASURED, "tokens", "llama.cpp's tokenizer, no special tokens").to_dict(),
        "words": words,
        "chars": chars,
        "bytes": nbytes,
        "tokens_per_word": round(len(ids) / words, 3) if words else None,
        "chars_per_token": round(chars / len(ids), 3) if ids else None,
        "bytes_per_token": round(nbytes / len(ids), 3) if ids else None,
        "by_script": by_script,
        "notes": notes,
        "tokens_per_second": round(len(ids) / secs) if secs > 0 else None,
        "tokens": shown,
        "tokens_shown": len(shown),
    }


def compare_text(paths: list, text: str, *, context: int | None = None,
                 show_tokens: int = 200) -> dict:
    """Every model's tokenizer on one text, and the context ruler."""
    vocabs = _open_all(paths)
    try:
        rows = []
        for p, v in zip(paths, vocabs):
            log.breadcrumb("tokenize", model=str(p), text_bytes=len(text.encode("utf-8", "replace")))
            m = measure(v, text, show_tokens=show_tokens)
            m["model"] = file_identity(p)
            m["vocab_type"] = v.type
            m["n_vocab"] = v.n_tokens
            if context:
                m["ruler"] = ruler_row(m, context)
            rows.append(m)
    finally:
        for v in vocabs:
            v.close()
    return {"kind": "tokenize", "text_chars": len(text), "context": context, "models": rows}


def ruler_row(m: dict, context: int) -> dict:
    """How many words and characters of THIS material a context holds."""
    tpw = m.get("tokens_per_word")
    cpt = m.get("chars_per_token")
    return {
        "context": context,
        "words": Figure(int(context / tpw) if tpw else None, ESTIMATE, "words",
                        "context ÷ tokens-per-word measured on this sample").to_dict(),
        "chars": Figure(int(context * cpt) if cpt else None, ESTIMATE, "characters",
                        "context × characters-per-token measured on this sample").to_dict(),
    }


TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".log", ".srt", ".vtt",
                   ".html", ".htm", ".xml"}


def load_corpus(path: str | os.PathLike, *, max_bytes: int = 64 << 20) -> dict:
    """A file's text, or a folder's text files joined, and an account of what
    was read: ``{"text", "files", "files_read", "bytes_read", "bytes_total",
    "truncated"}``. Decoded as UTF-8 (a byte-order mark is dropped; bytes
    that are not UTF-8 become U+FFFD)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    files = [p] if p.is_file() else sorted(q for q in p.rglob("*")
                                           if q.is_file() and q.suffix.lower() in TEXT_EXTENSIONS)
    total = sum(q.stat().st_size for q in files)
    parts, got, n_read = [], 0, 0
    for q in files:
        if got >= max_bytes:
            break
        with open(q, "rb") as f:
            data = f.read(max_bytes - got)
        parts.append(data.decode("utf-8-sig", "replace"))
        got += len(data)
        n_read += 1
    return {"text": "\n\n".join(parts), "files": len(files), "files_read": n_read,
            "bytes_read": got, "bytes_total": total, "truncated": got < total}


def read_corpus(path: str | os.PathLike, *, max_bytes: int = 64 << 20) -> str:
    """Just the text of ``load_corpus`` — which says whether ``max_bytes``
    cut it short; this does not."""
    return load_corpus(path, max_bytes=max_bytes)["text"]


def worst_splits(paths: list, strings: list | None = None) -> dict:
    """The analyst's strings, split by each tokenizer, worst first."""
    strings = strings or ANALYST_STRINGS
    items = [s if isinstance(s, (list, tuple)) else ("custom", s) for s in strings]
    vocabs = _open_all(paths)
    try:
        rows = []
        log.breadcrumb("worst-splits", models=[str(p) for p in paths], n_strings=len(items))
        for category, s in items:
            per = []
            for p, v in zip(paths, vocabs):
                ids = v.tokenize(s, add_special=False, parse_special=False)
                per.append({"model": Path(p).name, "n_tokens": len(ids),
                            "pieces": [v.piece(i, special=False).decode("utf-8", "replace")
                                       for i in ids]})
            worst = max(x["n_tokens"] for x in per)
            rows.append({"category": category, "string": s, "chars": len(s),
                         "tokens_per_char_worst": round(worst / max(1, len(s)), 3),
                         "models": per})
    finally:
        for v in vocabs:
            v.close()
    rows.sort(key=lambda r: -r["tokens_per_char_worst"])
    return {"kind": "worst-splits", "label": MEASURED, "strings": rows,
            "note": "Strings a model sees as many pieces are the ones it is likeliest to "
                    "corrupt when it copies them."}


def fertility_table(paths: list, text: str) -> dict:
    """The library as one table: tokens per word, by script, per model."""
    res = compare_text(paths, text, show_tokens=0)
    scripts = sorted({s for m in res["models"] for s in m["by_script"]})
    table = []
    for m in res["models"]:
        table.append({"model": m["model"]["name"], "vocab_type": m["vocab_type"],
                      "n_vocab": m["n_vocab"], "tokens_per_word": m["tokens_per_word"],
                      **{s: (m["by_script"].get(s) or {}).get("tokens_per_word") for s in scripts}})
    return {"kind": "fertility", "label": MEASURED, "scripts": scripts, "rows": table}
