"""The notebook — every run recorded, append-only (principle 6).

One JSON object per line in ``<data_dir>/notebook/notebook.jsonl``. Two
kinds of line:

* a **run**: ``{"type": "run", "id", "time", "kind", "question", "inputs",
  "settings", "versions", "result"}``
* a **note**: ``{"type": "note", "id", "time", "run", "text"}``

Nothing is ever rewritten; a note about a run is a later line naming it.
The format is specified in ``docs/formats/notebook.md``.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, Iterator

from .host import get_host
from .util import to_jsonable, utc_now, versions

FORMAT_VERSION = 1


def _new_id() -> str:
    return utc_now().replace("-", "").replace(":", "") + "-" + secrets.token_hex(3)


class Notebook:
    def __init__(self, path: str | os.PathLike | None = None):
        if path is None:
            path = get_host().data_dir() / "notebook" / "notebook.jsonl"
        self.path = Path(path)

    def _append(self, obj: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n"
        # one write per line, so a reader never sees half a record
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)

    def record(self, kind: str, *, question: str, inputs: list, settings: dict,
               result: Any) -> str:
        run_id = _new_id()
        self._append({
            "type": "run", "format": FORMAT_VERSION, "id": run_id, "time": utc_now(),
            "kind": kind, "question": question, "inputs": inputs,
            "settings": settings, "versions": versions(), "result": result,
        })
        return run_id

    def note(self, run_id: str, text: str) -> str:
        if self.get(run_id) is None:
            raise KeyError(f"no run {run_id!r} in {self.path}")
        note_id = _new_id()
        self._append({"type": "note", "format": FORMAT_VERSION, "id": note_id,
                      "time": utc_now(), "run": run_id, "text": text})
        return note_id

    def lines(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # a torn last line (power cut mid-write) is reported, not fatal
                    yield {"type": "damaged", "line": n}

    def runs(self, kind: str | None = None) -> list:
        return [r for r in self.lines()
                if r.get("type") == "run" and (kind is None or r.get("kind") == kind)]

    def notes(self, run_id: str) -> list:
        return [r for r in self.lines() if r.get("type") == "note" and r.get("run") == run_id]

    def get(self, run_id: str) -> dict | None:
        for r in self.lines():
            if r.get("type") == "run" and r.get("id") == run_id:
                return r
        return None
