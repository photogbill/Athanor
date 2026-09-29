"""The public Python API — the names other programs may rely on.

Everything else in the package is internal and may change without notice.
Until 1.0 these may change too, but only with a line in CHANGELOG.md; from
1.0, a name here is removed or changed only in a major version, after a
minor version that warns.

Every function returns plain data — dicts, lists, strings, numbers — that
``json.dumps`` can write (``athanor.util.dumps`` handles numpy values), with
every figure labelled MEASURED, DECLARED, ESTIMATE or EXPERIMENTAL. No Qt,
no global state beyond the host (``set_host``). Nothing is written except
Athanor's own log (``athanor.log``; ``ATHANOR_LOG=0`` turns it off) and
what you ask for (``record``).

    from athanor import api
    report = api.inspect("model.gguf")
    for finding in report["findings"]:
        print(finding["status"], finding["message"])
"""

from __future__ import annotations

from . import __version__ as VERSION
from .capabilities import probe as capabilities
from .gguf import GGUFFile, GGUFWriter, read as read_gguf, round_trip_check
from .host import Host, NullHost, get_host, set_host
from .labels import DECLARED, ESTIMATE, EXPERIMENTAL, MEASURED
from .notebook import Notebook
from .tabs.compare import compare, overlap_matrix
from .tabs.inspect import inspect, metadata
from .tabs.template import analyse as template, library as template_library
from .tabs.template import side_by_side as template_compare
from .tabs.tokenize import ANALYST_STRINGS, compare_text as tokenize
from .tabs.tokenize import fertility_table, read_corpus, worst_splits
from .templates import SAMPLE_CONVERSATION, detect as detect_template
from .templates import render as render_template, resolve as resolve_template
from . import log
from .util import dumps, file_identity
from .vocab import LlamaUnavailable, Vocab, VocabLoadError

API_VERSION = "0.1"

__all__ = [
    "API_VERSION", "VERSION",
    # labels
    "MEASURED", "DECLARED", "ESTIMATE", "EXPERIMENTAL",
    # host and environment
    "Host", "NullHost", "set_host", "get_host", "capabilities",
    # files
    "read_gguf", "GGUFFile", "GGUFWriter", "round_trip_check", "file_identity",
    # the tabs
    "inspect", "metadata",
    "tokenize", "worst_splits", "fertility_table", "read_corpus", "ANALYST_STRINGS",
    "compare", "overlap_matrix",
    "template", "template_compare", "template_library",
    "render_template", "resolve_template", "detect_template", "SAMPLE_CONVERSATION",
    # the tokenizer itself
    "Vocab", "LlamaUnavailable", "VocabLoadError",
    # the record, and the log
    "Notebook", "record", "dumps", "log",
]

_QUESTIONS = {
    "inspect": "What is this file, and is its tokenizer healthy?",
    "tokenize": "How much of this material fits, on which model?",
    "worst-splits": "Which of my strings does each tokenizer shred?",
    "fertility": "Tokens per word, by script, across the library",
    "compare": "How do these two files differ — same family?",
    "overlap": "Which models share a vocabulary?",
    "template": "What does the model actually see?",
    "template-compare": "How do these two templates differ?",
}


def record(result: dict, *, notebook: Notebook | None = None, question: str | None = None,
           inputs: list | None = None, settings: dict | None = None) -> str:
    """Write a result to the notebook; returns the run id.

    ``inputs`` defaults to the file identities the result carries.
    """
    kind = result.get("kind", "result")
    if inputs is None:
        inputs = [v for k, v in result.items()
                  if k in ("file", "a", "b", "model", "projector") and isinstance(v, dict)
                  and "path" in v]
        inputs += [m["model"] for m in result.get("models", [])
                   if isinstance(m, dict) and isinstance(m.get("model"), dict)]
    nb = notebook or Notebook()
    run = nb.record(kind, question=question or _QUESTIONS.get(kind, kind),
                    inputs=inputs, settings=settings or {}, result=result)
    from . import log
    log.event("recorded", run=run, kind=kind, notebook=str(nb.path))
    return run
