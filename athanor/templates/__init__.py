"""Chat templates: the library of 55, a model's own, a file — and rendering.

Rendering uses the same Jinja environment llama-cpp-python's
``Jinja2ChatFormatter`` builds (ea3b56b, ``llama_chat_format.py``): an
immutable sandbox, ``trim_blocks`` and ``lstrip_blocks``, the
``loopcontrols`` extension, HuggingFace's ``{% generation %}`` tag passed
through, and a ``tojson`` that does not escape non-ASCII — so a render here
is the text that binding would produce.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .library import LLAMA_CPP_TAG, Template, all_templates, detect, get, names

__all__ = ["LLAMA_CPP_TAG", "Template", "all_templates", "detect", "get", "names",
           "environment", "render", "resolve", "model_templates", "SAMPLE_CONVERSATION"]

#: A conversation that exercises a template: a system prompt, a finished
#: exchange, and a new question.
SAMPLE_CONVERSATION = [
    {"role": "system", "content": "You are a careful analyst."},
    {"role": "user", "content": "Who signed the lease?"},
    {"role": "assistant", "content": "Brightline Ltd, on 3 March."},
    {"role": "user", "content": "And who witnessed it?"},
]


def _tojson(x: Any, ensure_ascii: bool = False, indent: int | None = None,
            separators=None, sort_keys: bool = False) -> str:
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def environment():
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class IgnoreGenerationTags(jinja2.ext.Extension):
        tags = {"generation"}

        def parse(self, parser):
            parser.stream.skip(1)
            return parser.parse_statements(("name:endgeneration",), drop_needle=True)

    env = ImmutableSandboxedEnvironment(
        loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True,
        extensions=[IgnoreGenerationTags, jinja2.ext.loopcontrols])
    env.filters["tojson"] = _tojson
    return env


def _raise(message: str):
    raise ValueError(message)


def render(template_text: str, messages: list, *, add_generation_prompt: bool = True,
           bos_token: str = "", eos_token: str = "", **extra: Any) -> str:
    """Render a Jinja chat template as llama-cpp-python would."""
    return environment().from_string(template_text).render(
        messages=messages, add_generation_prompt=add_generation_prompt,
        bos_token=bos_token, eos_token=eos_token, raise_exception=_raise,
        strftime_now=lambda f: datetime.now().strftime(f), **extra)


def model_templates(g) -> dict:
    """A GGUF's own templates: ``{"default": text, "<name>": text, ...}``."""
    out = {}
    t = g.get("tokenizer.chat_template")
    if isinstance(t, str):
        out["default"] = t
    prefix = "tokenizer.chat_template."
    for key in g.keys():
        if key.startswith(prefix) and isinstance(g.get(key), str):
            out[key[len(prefix):]] = g.get(key)
    return out


def resolve(spec: str, *, gguf=None) -> dict:
    """Turn a template choice into ``{"name", "source", "text"}``.

    ``spec`` is one of: a library name (``chatml``, or ``llama.cpp/chatml``);
    ``gguf`` or ``gguf:<name>`` — the model's own (needs ``gguf=``); a path to
    a ``.jinja`` / text file; or Jinja text itself (anything containing
    ``{%`` or ``{{``).
    """
    s = spec.strip()
    lib = s[len("llama.cpp/"):] if s.startswith("llama.cpp/") else s
    t = get(lib)
    if t is not None:
        return {"name": f"llama.cpp/{t.name}", "source": "library", "text": t.text,
                "label": t.label}
    if s == "gguf" or s.startswith("gguf:"):
        if gguf is None:
            raise ValueError("the model's own template needs a model")
        own = model_templates(gguf)
        which = s.split(":", 1)[1] if ":" in s else "default"
        if which not in own:
            have = ", ".join(own) or "none"
            raise ValueError(f"{Path(gguf.path).name} has no chat template {which!r} "
                             f"(it has: {have})")
        return {"name": f"gguf:{which}", "source": "gguf", "text": own[which],
                "label": f"the model's own ({which})"}
    if "{%" in s or "{{" in s:
        return {"name": "text", "source": "text", "text": spec, "label": "given as text"}
    p = Path(os.path.expanduser(s))
    if p.is_file():
        return {"name": f"file:{p.name}", "source": "file",
                "text": p.read_text(encoding="utf-8-sig"), "label": str(p)}
    raise ValueError(f"no template called {spec!r} — not a library name "
                     f"({len(names())} of them: `athanor template --list`), not a file, "
                     "not Jinja text")
