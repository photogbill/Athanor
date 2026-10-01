"""Athanor's optional Qt widgets (PySide6): ``pip install athanor[gui]``.

* ``WaterfallPlayer`` — one recording, played back.
* ``WaterfallPanel`` — a folder's recordings beside the player: what a host
  embeds as its Waterfall tab.
* ``WaterfallWindow`` — the same, as a window; ``python -m athanor.gui``.
* ``ExpertPanel`` / ``ExpertMap`` — a mixture-of-experts model's routing
  (the Tap's ``experts``), shown in the player beside the candidates.

The engine never imports this package, and this package never imports a
host.
"""

from .browser import RecordingsList, WaterfallPanel, WaterfallWindow
from .experts import ExpertMap, ExpertPanel
from .waterfall import (CandidateList, ReplyView, WaterfallPlayer, WaterfallView, db_of, lut,
                        visible)

__all__ = ["WaterfallPlayer", "WaterfallPanel", "WaterfallWindow", "RecordingsList",
           "WaterfallView", "ReplyView", "CandidateList", "ExpertMap", "ExpertPanel", "db_of",
           "lut", "visible"]
