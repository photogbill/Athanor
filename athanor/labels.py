"""How every number Athanor reports was made.

Principle 1 of the plan — measure, don't assume. Every figure carries one
of these labels, on screen, in JSON and in the notebook:

* ``MEASURED``     — observed by running something (llama.cpp tokenized the
  text; a load was timed).
* ``DECLARED``     — what the file says about itself, read from its header
  and not checked (the context length in the metadata, the name).
* ``ESTIMATE``     — arithmetic from other figures, not an observation.
* ``EXPERIMENTAL`` — research-grade: interesting, not evidence.

``DECLARED`` is the fourth label the plan's three grew into on the first
day of building: a model's header claims things (a 1M context, a chat
template) that are neither measurements nor estimates, and a reader needs
to see the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MEASURED = "MEASURED"
DECLARED = "DECLARED"
ESTIMATE = "ESTIMATE"
EXPERIMENTAL = "EXPERIMENTAL"

ALL = (MEASURED, DECLARED, ESTIMATE, EXPERIMENTAL)


@dataclass
class Figure:
    """A number (or short value) with its label and how it was made."""

    value: Any
    label: str
    unit: str | None = None
    how: str | None = None

    def __post_init__(self):
        if self.label not in ALL:
            raise ValueError(f"unknown label {self.label!r}")

    def to_dict(self) -> dict:
        d = {"value": self.value, "label": self.label}
        if self.unit:
            d["unit"] = self.unit
        if self.how:
            d["how"] = self.how
        return d


@dataclass
class Finding:
    """One health-check result: what was checked, what was found."""

    check: str             # short id, e.g. "eos-matches-template"
    status: str            # "ok" | "warn" | "problem" | "info" | "skipped"
    message: str           # a plain sentence
    label: str = MEASURED
    evidence: dict = field(default_factory=dict)

    STATUSES = ("ok", "info", "warn", "problem", "skipped")

    def __post_init__(self):
        if self.status not in self.STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if self.label not in ALL:
            raise ValueError(f"unknown label {self.label!r}")

    def to_dict(self) -> dict:
        return {"check": self.check, "status": self.status, "message": self.message,
                "label": self.label, "evidence": self.evidence}
