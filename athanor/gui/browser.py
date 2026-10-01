"""A folder of recordings, and a window that plays them."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QPushButton, QSplitter, QVBoxLayout, QWidget)

from ..waterfall import RecordingError, list_recordings
from .waterfall import WaterfallPlayer


class _ScanSignals(QObject):
    done = Signal(int, object)          # (generation, rows)


class _Scan(QRunnable):
    """Lists a folder off the GUI thread: every meta file is parsed, and a
    folder of a hundred recordings takes seconds the first time."""

    def __init__(self, folder: Path, generation: int, signals: _ScanSignals):
        super().__init__()
        self.folder, self.generation, self.signals = folder, generation, signals

    def run(self):
        try:
            rows = list_recordings(self.folder)
        except Exception as exc:                          # the list says so
            rows = [{"path": str(self.folder), "error": f"{type(exc).__name__}: {exc}",
                     "preview": "", "model": None, "n_steps": None}]
        try:
            self.signals.done.emit(self.generation, rows)
        except RuntimeError:                             # the widget is gone
            pass


class RecordingsList(QWidget):
    """The recordings in a folder, newest first. ``opened`` carries the path
    of the one chosen; ``listed`` fires each time the list is filled.

    The folder is read on a worker thread when the list is first shown and
    each time it is shown again (``refresh``); ``refresh(wait=True)`` reads
    it here and now."""

    opened = Signal(str)
    listed = Signal(int)

    def __init__(self, folder=None, parent=None):
        super().__init__(parent)
        self.folder = Path(folder) if folder else None
        self.heading = QLabel("")
        self.heading.setWordWrap(True)
        self.items = QListWidget()
        self.refresh_btn = QPushButton("Refresh")
        self.choose_btn = QPushButton("Folder…")
        row = QHBoxLayout()
        row.addWidget(self.refresh_btn)
        row.addWidget(self.choose_btn)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.heading)
        lay.addWidget(self.items, 1)
        lay.addLayout(row)
        self._signals = _ScanSignals(self)
        self._signals.done.connect(self._fill)
        self._generation = 0
        self._select_when_listed: str | None = None
        self.rows: list = []
        self.refresh_btn.clicked.connect(lambda: self.refresh())
        self.choose_btn.clicked.connect(self._choose)
        self.items.itemActivated.connect(self._activated)
        self.items.itemClicked.connect(self._activated)
        self.heading.setText("No folder chosen" if self.folder is None else "")

    def set_folder(self, folder):
        self.folder = Path(folder) if folder else None
        self.refresh()

    def showEvent(self, ev):
        """Shown (again): a host switched back to this tab, and new
        recordings may have been made meanwhile."""
        super().showEvent(ev)
        self.refresh()

    def refresh(self, wait: bool = False):
        """Re-read the folder — on a worker thread, or here with ``wait``."""
        self._generation += 1
        if self.folder is None:
            self._fill(self._generation, [])
            return self.rows
        if wait:
            self._fill(self._generation, list_recordings(self.folder))
            return self.rows
        if not self.rows:
            self.heading.setText(f"reading <b>{self.folder.name}</b>…")
        QThreadPool.globalInstance().start(_Scan(self.folder, self._generation, self._signals))
        return self.rows

    def _fill(self, generation: int, rows: list):
        if generation != self._generation:
            return                                        # a newer scan is on its way
        cur = self.items.currentItem()
        keep = self._select_when_listed or (cur.data(Qt.UserRole) if cur is not None else None)
        self._select_when_listed = None
        self.rows = rows
        self.items.blockSignals(True)
        self.items.clear()
        n = len(rows)
        self.heading.setText(f"{n} recording{'' if n == 1 else 's'} in <b>{self.folder.name}</b>"
                             if self.folder else "No folder chosen")
        self.heading.setToolTip(str(self.folder or ""))
        for r in rows:
            when = (r.get("started") or r.get("created") or "?")[:19].replace("T", " ")
            first = f"{when}   {r.get('model') or '?'}   {r.get('n_steps') or 0} tokens"
            second = ("⚠ " + r["error"]) if r.get("error") else (
                r.get("preview") or "").replace("\n", " ")
            item = QListWidgetItem(f"{first}\n{second[:120]}")
            item.setData(Qt.UserRole, r["path"])
            item.setToolTip(r["path"] + (f"\n{r['error']}" if r.get("error") else ""))
            self.items.addItem(item)
        if keep:
            self.select(keep)
        self.items.blockSignals(False)
        self.listed.emit(n)

    def select(self, path, when_listed: bool = False) -> bool:
        """Select a recording; with ``when_listed``, also once the next
        listing arrives (a recording just made is not listed yet)."""
        for i in range(self.items.count()):
            if Path(self.items.item(i).data(Qt.UserRole)) == Path(path):
                self.items.setCurrentRow(i)
                return True
        if when_listed:
            self._select_when_listed = str(path)
        return False

    def _activated(self, item):
        self.opened.emit(item.data(Qt.UserRole))

    def _choose(self):
        d = QFileDialog.getExistingDirectory(self, "Recordings folder",
                                             str(self.folder or Path.home()))
        if d:
            self.set_folder(d)


class WaterfallPanel(QWidget):
    """The list beside the player: what a host embeds as its Waterfall tab."""

    def __init__(self, folder=None, parent=None):
        super().__init__(parent)
        self.list = RecordingsList(folder)
        self.player = WaterfallPlayer()
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.list)
        split.addWidget(self.player)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 4)
        split.setSizes([280, 1200])
        self.list.setMinimumWidth(200)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        self.list.opened.connect(self.open)

    def open(self, path) -> bool:
        """Show a recording, and select it in the list (re-reading the folder
        when it is new). A recording that cannot be read is said to be so —
        the player is cleared, never left showing the previous reply."""
        try:
            self.player.load(path)
            ok = True
        except (RecordingError, OSError, ValueError, KeyError, TypeError) as exc:
            self.player.load(None)
            self.player.title.setText(f"⚠ {Path(str(path)).name} could not be opened: "
                                      f"{type(exc).__name__}: {exc}")
            ok = False
        if not self.list.select(path):
            self.list.select(path, when_listed=True)
            self.list.refresh()
        return ok


class WaterfallWindow(QMainWindow):
    def __init__(self, folder=None, path=None):
        super().__init__()
        self.setWindowTitle("Athanor — Waterfall")
        self.panel = WaterfallPanel(folder)
        self.setCentralWidget(self.panel)
        self.resize(1280, 800)
        if path:
            self.panel.open(path)
