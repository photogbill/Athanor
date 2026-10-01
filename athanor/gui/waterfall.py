"""The Waterfall player — a recording played back like an RF waterfall.

Top: the reply — the current token lit, what is still to come dimmed (or
hidden, to watch it written), thinking in italics, the words the model was
unsure of underlined. Middle: the
waterfall — time runs down, candidates run across (by rank, or by token id),
brightness is probability in dB; a tick marks the token actually chosen,
amber where the sampler took something other than the favourite. Zoomed in
("read"), every cell shows its candidate's text, so each row reads as the
words the model was choosing between. Beside it, the entropy trace. Right:
the candidates at the cursor. Under it all, the transport: play at the
recorded speed or faster, step, scrub, and jump from one moment of doubt to
the next.

Everything here reads a ``Recording`` (athanor.waterfall); nothing needs the
model, and nothing needs a host.
"""

from __future__ import annotations

import bisect
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QPainter, QPalette, QPen,
                           QTextCharFormat, QTextCursor)
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QPlainTextEdit, QPushButton, QSizePolicy, QSlider, QSplitter,
                               QTextEdit, QToolTip, QVBoxLayout, QWidget)

from ..waterfall import FLAG_EOG, Recording, read

#: The colour ramp of ATK's RF waterfall — deep blue (the floor) to pale
#: yellow (strong) — so a model's waterfall reads the way a band does. The
#: low end keeps moving all the way down: most of the picture is floor, and
#: that is where texture is read.
STOPS = [
    (0.00, (8, 14, 40)),
    (0.18, (18, 46, 110)),
    (0.38, (30, 96, 170)),
    (0.55, (60, 150, 190)),
    (0.72, (140, 195, 150)),
    (0.88, (230, 220, 90)),
    (1.00, (255, 250, 190)),
]
DB_PER_NAT = 10.0 / math.log(10.0)
SPEEDS = [("recorded speed", 1.0), ("×2", 2.0), ("×4", 4.0), ("×10", 10.0), ("×0.5", 0.5),
          ("20 tokens/s", -20.0), ("5 tokens/s", -5.0)]
RANKS = [8, 12, 16, 32, 64, 128, 256]


def lut(n: int = 256) -> np.ndarray:
    """(n, 3) uint8: the ramp, interpolated."""
    pos = np.array([p for p, _c in STOPS])
    cols = np.array([c for _p, c in STOPS], dtype=np.float64)
    x = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(x, pos, cols[:, i]) for i in range(3)], axis=1).round().astype(
        np.uint8)


_LUT = lut()


def db_of(logprob) -> np.ndarray:
    """Probability in decibels: 0 dB is certain, −10 dB is 10 %, −20 dB 1 %."""
    return np.asarray(logprob, np.float64) * DB_PER_NAT


def visible(piece: str) -> str:
    """A token's text with the invisible made visible: spaces at its edges
    as ␣, newlines as ⏎, tabs as ⇥, other control characters as their
    Unicode pictures, and nothing at all as ∅."""
    if piece == "":
        return "∅"
    out = []
    for ch in piece:
        o = ord(ch)
        if ch == "\n":
            out.append("⏎")
        elif ch == "\t":
            out.append("⇥")
        elif o < 0x20:
            out.append(chr(0x2400 + o))
        elif o == 0x7F:
            out.append("␡")
        else:
            out.append(ch)
    s = "".join(out)
    if not s.strip(" "):
        return "␣" * len(s)
    lead = len(s) - len(s.lstrip(" "))
    trail = len(s) - len(s.rstrip(" "))
    return "␣" * lead + s[lead:len(s) - trail] + "␣" * trail


def _ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th … 11th, 12th, 13th … 21st."""
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def display_text(text: str) -> str:
    """The reply for the text view, one character for one: control
    characters other than newline and tab become their Unicode pictures, so
    nothing the model wrote is invisible and every offset still lines up."""
    return "".join(chr(0x2400 + ord(c)) if ord(c) < 0x20 and c not in "\n\t" else c
                   for c in text)


def utf16_offsets(text: str) -> np.ndarray:
    """Qt counts positions in UTF-16 units; Python in code points. out[i] is
    Qt's position of Python index i (len(text)+1 entries)."""
    widths = np.fromiter((2 if ord(c) > 0xFFFF else 1 for c in text), np.int64, len(text))
    return np.concatenate([[0], np.cumsum(widths)])


# =========================================================== the reply text
class ReplyView(QPlainTextEdit):
    """The reply. Past: normal. Current token: lit. Future: dimmed — or, with
    ``reveal`` on, hidden, so the reply is written out as it plays. Hidden
    text still takes its place, so nothing reflows."""

    tokenClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self._rec: Recording | None = None
        self._q = np.zeros(1, np.int64)
        self._thinking: list = []
        self.reveal = False
        self.mark_doubts = True     # a wavy amber line under every moment of doubt
        self._shown = 0
        self.viewport().setMouseTracking(True)

    def load(self, rec: Recording | None):
        self._rec = rec
        text = display_text(rec.text) if rec else ""
        # setPlainText writes with the CURRENT format — which, after a step
        # was shown, is the underlined or italic one under the cursor
        self.setTextCursor(QTextCursor(self.document()))
        self.setCurrentCharFormat(QTextCharFormat())
        self.setPlainText(text)
        self._q = utf16_offsets(text)
        self._thinking = []
        if rec is None:
            return
        fmt = QTextCharFormat()
        fmt.setFontItalic(True)
        fmt.setForeground(QColor(154, 127, 209))
        # ONE edit block for every format below: a cursor per span, each laid
        # out on its own, cost seconds on a long reply (and minutes on one
        # long unbroken line) — measured by the 2026-09-29 review
        c = QTextCursor(self.document())
        c.beginEditBlock()
        try:
            self._format_spans(c, rec, fmt)
        finally:
            c.endEditBlock()

    def _format_spans(self, c: QTextCursor, rec: Recording, fmt: QTextCharFormat):
        for a in rec.meta.get("annotations") or []:
            if a.get("athrec:label") != "thinking":
                continue
            s0 = int(a.get("athrec:step_start", 0))
            s1 = s0 + int(a.get("athrec:step_count", 0)) - 1
            if rec.n_steps == 0 or s1 < s0:
                continue
            start = rec.span(max(0, min(s0, rec.n_steps - 1)))[0]
            end = rec.span(max(0, min(s1, rec.n_steps - 1)))[1]
            c.setPosition(int(self._q[start]))
            c.setPosition(int(self._q[end]), QTextCursor.KeepAnchor)
            c.mergeCharFormat(fmt)
            self._thinking.append((s0, s1))
        if self.mark_doubts:
            wave = QTextCharFormat()
            wave.setUnderlineStyle(QTextCharFormat.WaveUnderline)
            wave.setUnderlineColor(QColor(255, 160, 50))
            wave.setBackground(QColor(255, 160, 50, 46))
            for step in rec.doubts():
                a, b = rec.span(int(step))
                if b > a:
                    c.setPosition(int(self._q[a]))
                    c.setPosition(int(self._q[b]), QTextCursor.KeepAnchor)
                    c.mergeCharFormat(wave)

    def qt_pos(self, i: int) -> int:
        return int(self._q[max(0, min(i, len(self._q) - 1))])

    def show_step(self, step: int):
        rec = self._rec
        if rec is None or rec.n_steps == 0:
            self.setExtraSelections([])
            return
        a, b = rec.span(step)
        self._shown = int(step)
        pal = self.palette()
        cur = QTextEdit.ExtraSelection()
        cur.cursor = QTextCursor(self.document())
        cur.cursor.setPosition(self.qt_pos(a))
        cur.cursor.setPosition(self.qt_pos(max(b, a)), QTextCursor.KeepAnchor)
        f = QTextCharFormat()
        f.setBackground(pal.color(QPalette.Highlight))
        f.setForeground(pal.color(QPalette.HighlightedText))
        cur.format = f
        fut = QTextEdit.ExtraSelection()
        fut.cursor = QTextCursor(self.document())
        fut.cursor.setPosition(self.qt_pos(max(b, a)))
        fut.cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        g = QTextCharFormat()
        dim = pal.color(QPalette.Text)
        dim.setAlphaF(0.0 if self.reveal else 0.28)
        g.setForeground(dim)
        if self.reveal:                   # nothing of what is to come shows, marks included
            g.setBackground(pal.color(QPalette.Base))
            g.setUnderlineStyle(QTextCharFormat.NoUnderline)
        fut.format = g
        self.setExtraSelections([fut, cur])
        c = QTextCursor(self.document())
        c.setPosition(self.qt_pos(a))
        self.setTextCursor(c)
        self.ensureCursorVisible()

    def mouseMoveEvent(self, ev):
        """Hover a word: how sure the model was when it wrote it."""
        super().mouseMoveEvent(ev)
        rec = self._rec
        if rec is None or rec.n_steps == 0:
            return
        qpos = self.cursorForPosition(ev.position().toPoint()).position()
        i = int(np.searchsorted(self._q, qpos, side="right") - 1)
        step = rec.step_at_char(i)
        if step is None or (self.reveal and step > self._shown):
            QToolTip.hideText()
            return
        c = rec.chosen(step)
        p = f"{c['p']:.2%}" if c["p"] is not None else "?"
        how = ("its favourite" if c["rank"] == 0 else
               f"its {_ordinal(c['rank'] + 1)} choice" if c["rank"] > 0 else "rank unknown")
        best = rec.candidates(step, 1)
        also = (f"\nits favourite was {visible(best[0]['piece'])} ({best[0]['p']:.2%})"
                if c["rank"] > 0 and best else "")
        QToolTip.showText(ev.globalPosition().toPoint(),
                          f"step {step + 1} · {visible(c['piece'])} · {p} — {how}{also}", self)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self._rec is None or ev.button() != Qt.LeftButton or self.textCursor().hasSelection():
            return
        qpos = self.cursorForPosition(ev.position().toPoint()).position()
        i = int(np.searchsorted(self._q, qpos, side="right") - 1)
        step = self._rec.step_at_char(i)
        if step is not None:
            self.tokenClicked.emit(step)


# ======================================================== the waterfall view
class WaterfallView(QWidget):
    """Time down, candidates across, dB as colour; the entropy trace at the
    right. One row per token."""

    stepClicked = Signal(int)
    TRACE_W = 70

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._rec: Recording | None = None
        self._img: QImage | None = None
        self._img_bytes = None
        self.axis = "rank"
        self.ranks = 64
        self.floor_db = -40.0
        self.row_px = 4
        self.id_bins = 512
        self.words = True       # write each candidate's text in its cell when there is room
        self.step = 0
        self.top = 0.0          # first visible row
        self._doubt = np.zeros(0, bool)

    # -------------------------------------------------------------- data
    def load(self, rec: Recording | None):
        self._rec = rec
        self.step = 0
        self.top = 0.0
        self._doubt = np.zeros(rec.n_steps if rec else 0, bool)
        if rec is not None and rec.n_steps:
            self._doubt[rec.doubts()] = True
        self.rebuild()

    def rebuild(self):
        """The whole recording as one image: rows are steps, columns ranks
        (or id bins). Rebuilt when the axis, the width or the floor change."""
        rec = self._rec
        if rec is None or rec.n_steps == 0:
            self._img = None
            self.update()
            return
        s = rec.steps
        n = len(s)
        if self.axis == "rank":
            w = min(self.ranks, rec.k)
            dbv = db_of(s["logprobs"][:, :w])
        else:
            w = self.id_bins
            dbv = np.full((n, w), -np.inf)
            nv = max(rec.n_vocab, int(s["ids"].max()) + 1, 1)
            bins = (s["ids"].astype(np.int64) * w) // nv
            vals = db_of(s["logprobs"])
            rows = np.repeat(np.arange(n), s["ids"].shape[1])
            np.maximum.at(dbv, (rows, bins.reshape(-1)), vals.reshape(-1))
        x = np.clip((dbv - self.floor_db) / (0.0 - self.floor_db), 0.0, 1.0)
        x = np.nan_to_num(x, nan=0.0)
        rgb = _LUT[(x * 255).astype(np.int32)]
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self._img_bytes = rgb.tobytes()
        self._img = QImage(self._img_bytes, w, n, 3 * w, QImage.Format_RGB888)
        self.update()

    # ---------------------------------------------------------- geometry
    def _compute(self, words: bool) -> dict:
        r = self.rect()
        fm = QFontMetrics(self.font())
        top = fm.height() + 6                               # the header strip
        h = max(1, r.height() - top)
        gutter_w = fm.horizontalAdvance("0000 ◆") + 10 if words else 7
        taken_w = min(190, max(104, r.width() // 8)) if words else 16
        x = r.left()
        gutter = QRect(x, top, gutter_w, h)
        x += gutter_w + 1
        taken = QRect(x, top, taken_w, h)
        x += taken_w + 4
        grid = QRect(x, top, max(10, r.right() - self.TRACE_W - 4 - x), h)
        trace = QRect(grid.right() + 5, top, self.TRACE_W, h)
        return {"top": top, "gutter": gutter, "taken": taken, "grid": grid, "trace": trace,
                "words": words}

    def _layout(self) -> dict:
        """The areas, left to right under a header strip: the gutter (step
        numbers, moments of doubt), TAKEN (the token the model actually took —
        on every row, whatever its rank), the candidates, the entropy trace.
        Word-sized when the rows are tall enough and the cells wide enough to
        write each candidate in, compact otherwise."""
        fm = QFontMetrics(self.font())
        if (self.words and self.axis == "rank" and self._img is not None
                and self.row_px >= fm.height() + 2):
            lay = self._compute(True)
            if lay["grid"].width() / max(1, self._img.width()) >= 28:
                return lay
        return self._compute(False)

    def _areas(self) -> tuple[QRect, QRect]:
        """(candidates, trace) — kept for callers of the first version."""
        lay = self._layout()
        return lay["grid"], lay["trace"]

    def word_mode(self) -> bool:
        """Is there room to write each candidate's text in its cell?"""
        return bool(self._layout()["words"])

    def reading_zoom(self) -> tuple[int, int]:
        """(row height, ranks) that make every cell readable at this width."""
        fm = QFontMetrics(self.font())
        grid = self._compute(True)["grid"]
        fit = max(4, grid.width() // max(60, fm.horizontalAdvance("M") * 7))
        ranks = max([r for r in RANKS if r <= fit] or [RANKS[0]])
        return fm.height() + 6, ranks

    def visible_rows(self) -> int:
        return max(1, (self.height() - self._layout()["top"]) // self.row_px)

    def set_step(self, step: int, follow: bool = True):
        self.step = int(step)
        if follow:
            vis = self.visible_rows()
            if not (self.top <= self.step < self.top + vis - 1):
                self.top = max(0.0, self.step - vis * 2 / 3)
        self.update()

    def hit(self, x: int, y: int) -> dict | None:
        """What is under a point: the step, and the candidate (a rank, or
        "taken" in the column of tokens taken), if any."""
        rec = self._rec
        if rec is None or rec.n_steps == 0:
            return None
        lay = self._layout()
        if y < lay["top"]:
            return None
        step = int(self.top + (y - lay["top"]) // self.row_px)
        if not 0 <= step < rec.n_steps:
            return None
        out = {"step": step, "candidate": None}
        grid = lay["grid"]
        if lay["taken"].contains(QPoint(x, y)):
            out["candidate"] = "taken"
        elif grid.contains(QPoint(x, y)) and self._img is not None:
            col = int((x - grid.left()) * self._img.width() / max(1, grid.width()))
            row = rec.steps[step]
            if self.axis == "rank":
                if 0 <= col < rec.k:
                    out["candidate"] = col
            else:
                nv = max(rec.n_vocab, 1)
                bins = (row["ids"].astype(np.int64) * self.id_bins) // nv
                hits = np.flatnonzero(bins == col)
                if len(hits):
                    out["candidate"] = int(hits[0])      # the best in that bin
        return out

    # ------------------------------------------------------------- paint
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(*STOPS[0][1]))
        rec = self._rec
        if rec is None or self._img is None:
            p.setPen(self.palette().color(QPalette.PlaceholderText))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "No recording." if rec is None else "This recording has no tokens.")
            return
        lay = self._layout()
        top, grid, trace = lay["top"], lay["grid"], lay["trace"]
        gutter, taken, words = lay["gutter"], lay["taken"], lay["words"]
        n = rec.n_steps
        vis = self.visible_rows() + 1
        first = int(self.top)
        last = min(n, first + vis)
        rows = last - first
        s = rec.steps
        colw = grid.width() / max(1, self._img.width())

        def row_y(i):
            return top + (i - first) * self.row_px

        p.save()
        p.setClipRect(QRect(0, top, self.width(), self.height() - top))
        if rows > 0:
            p.setRenderHint(QPainter.SmoothPixmapTransform, False)
            p.drawImage(QRect(grid.left(), top, grid.width(), rows * self.row_px), self._img,
                        QRect(0, first, self._img.width(), rows))
        if words:
            self._paint_words(p, grid, colw, first, last)
        self._paint_taken(p, taken, words, first, last)
        self._paint_gutter(p, gutter, words, first, last)
        # the chosen token in the candidates: a frame round its cell (a tick
        # in the compact view) — white where it was the model's favourite,
        # amber where the sampler took another, red at the right edge when it
        # ranked beyond the columns shown
        tick_w = max(2, min(6, int(colw // 3)))
        for i in range(first, last):
            y = row_y(i)
            h = max(1, self.row_px - 1)
            r = int(s["rank"][i])
            if self.axis == "rank":
                # found by id: under an exact tie the taken token sits after
                # the others with its logit, while its rank counts only the
                # tokens strictly more probable (docs/formats/recording.md)
                hits = np.flatnonzero(s["ids"][i][: self._img.width()] == s["chosen"][i])
                col_i = int(hits[0]) if len(hits) else -1
            else:
                col_i = (int(s["chosen"][i]) * self.id_bins) // max(rec.n_vocab, 1)
            if 0 <= col_i < self._img.width() and r >= 0:
                colour = QColor(255, 255, 255) if r == 0 else QColor(255, 150, 40)
                if words and r == 0 and self._bright(i, col_i):
                    colour = QColor(10, 12, 18)           # white does not show on pale cells
                if words:
                    x0 = grid.left() + int(col_i * colw)
                    x1 = grid.left() + int((col_i + 1) * colw)
                    p.setPen(QPen(colour, 2))
                    p.setBrush(Qt.NoBrush)
                    p.drawRect(QRect(x0 + 1, y + 1, x1 - x0 - 2, h - 1))
                else:
                    x = grid.left() + int((col_i + 0.5) * colw) - tick_w // 2
                    p.fillRect(x, y, tick_w, h, colour)
            else:
                p.fillRect(grid.right() - 3, y, 3, h, QColor(239, 80, 70))
        # thinking: a violet bar at the far left
        for a in rec.meta.get("annotations") or []:
            if a.get("athrec:label") == "thinking":
                s0 = int(a.get("athrec:step_start", 0))
                s1 = s0 + int(a.get("athrec:step_count", 0))
                y0 = row_y(max(s0, first))
                y1 = row_y(min(s1, last))
                if y1 > y0:
                    p.fillRect(0, y0, 3, y1 - y0, QColor(154, 127, 209))
        # the entropy trace (bits) and the chosen token's surprise (bits)
        p.fillRect(trace, QColor(12, 15, 20))
        ent = s["entropy"][first:last] / math.log(2)
        sur = -s["logprob"][first:last] / math.log(2)
        top_bits = max(4.0, float(np.nanmax(s["entropy"]) / math.log(2)) if n else 4.0)
        for j, e in enumerate(ent):
            y = top + j * self.row_px
            w = int(trace.width() * min(1.0, float(e) / top_bits)) if math.isfinite(e) else 0
            p.fillRect(trace.left(), y, w, max(1, self.row_px - 1), QColor(40, 110, 120))
            su = float(sur[j])
            if math.isfinite(su):
                x = trace.left() + int(trace.width() * min(1.0, su / top_bits))
                p.fillRect(x - 1, y, 2, max(1, self.row_px - 1), QColor(230, 179, 74))
        # what is still to come, dimmed — as in the text above
        if self.step + 1 < last:
            y0 = max(top, row_y(self.step + 1))
            p.fillRect(QRect(0, y0, self.width(), self.height() - y0), QColor(0, 0, 0, 120))
        # the cursor: a lit band across the whole row, and a frame
        if first <= self.step < last:
            y = row_y(self.step)
            p.fillRect(QRect(0, y, self.width(), self.row_px), QColor(255, 255, 255, 34))
            p.setPen(QPen(QColor(255, 255, 255, 210), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRect(0, y - 1, self.width() - 1, self.row_px + 1))
        p.restore()
        self._paint_header(p, lay, words)
        self._paint_legend(p, grid, rec)
        p.end()

    def _paint_header(self, p: QPainter, lay: dict, words: bool):
        """Column titles, so a first-time viewer knows what each part is."""
        fm = QFontMetrics(p.font())
        top = lay["top"]
        p.fillRect(QRect(0, 0, self.width(), top), QColor(12, 15, 20))
        p.setPen(QColor(150, 162, 184))
        mid = Qt.AlignVCenter | Qt.AlignLeft

        def title(rect, text, align=mid):
            r = QRect(rect.left() + 2, 0, rect.width() - 4, top)
            p.drawText(r, align, fm.elidedText(text, Qt.ElideRight, r.width()))

        if words:
            title(lay["gutter"], "step", Qt.AlignVCenter | Qt.AlignRight)
        p.setPen(QColor(236, 214, 150))
        title(lay["taken"], "taken" if words else "")
        p.setPen(QColor(150, 162, 184))
        grid = lay["grid"]
        if self.axis == "rank" and words and self._img is not None:
            colw = grid.width() / max(1, self._img.width())
            for c in range(self._img.width()):
                x0 = grid.left() + int(c * colw)
                w = grid.left() + int((c + 1) * colw) - x0
                p.drawText(QRect(x0 + 3, 0, w - 6, top), mid, _ordinal(c + 1))
        elif self.axis == "rank":
            title(grid, "the candidates, the model's favourite first  →")
        else:
            title(grid, "the candidates by token id  →")
        title(lay["trace"], "entropy")

    def _paint_legend(self, p: QPainter, grid: QRect, rec):
        fm = QFontMetrics(p.font())
        label = (f"{self._img.width()} ranks" if self.axis == "rank"
                 else f"token id 0 … {rec.n_vocab - 1}")
        base = self.height() - 6
        bar = QRect(grid.left() + 6, base - fm.ascent() + 2, 120, fm.ascent() - 4)
        legend = f"{self.floor_db:.0f} … 0 dB    {label}"
        p.fillRect(QRect(grid.left(), base - fm.ascent() - 3,
                         bar.width() + fm.horizontalAdvance(legend) + 20, fm.height() + 6),
                   QColor(8, 14, 40, 225))
        for j in range(bar.width()):
            c = _LUT[int(255 * j / max(1, bar.width() - 1))]
            p.fillRect(bar.left() + j, bar.top(), 1, bar.height(),
                       QColor(int(c[0]), int(c[1]), int(c[2])))
        p.setPen(QColor(220, 228, 240))
        p.drawRect(bar.adjusted(0, 0, -1, -1))
        p.drawText(bar.right() + 6, base, legend)

    def _paint_taken(self, p: QPainter, area: QRect, words: bool, first: int, last: int):
        """The column of tokens TAKEN — every row shows the one the model
        actually wrote, lit in its own probability's colour, framed white when
        it was the model's favourite and amber when the sampler took another
        (its rank beside it), whatever column it sits in among the candidates."""
        rec = self._rec
        s = rec.steps
        fm = QFontMetrics(p.font())
        bold = QFont(p.font())
        bold.setBold(True)
        bfm = QFontMetrics(bold)
        light, dark = QColor(236, 240, 248), QColor(10, 12, 18)
        amber, red = QColor(255, 150, 40), QColor(239, 80, 70)
        shown = self._img.width() if self.axis == "rank" else rec.k
        for i in range(first, last):
            y = area.top() + (i - first) * self.row_px
            h = max(1, self.row_px - 1)
            lp, r = float(s["logprob"][i]), int(s["rank"][i])
            rgb = _LUT[int(self._level(lp) * 255)]
            fill = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            p.fillRect(QRect(area.left(), y, area.width(), h), fill)
            edge = QColor(255, 255, 255) if r == 0 else (amber if 0 < r < shown else red)
            if not words:
                p.fillRect(area.left(), y, 4, h, edge)
                continue
            bright = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 140
            if r == 0 and bright:
                edge = dark
            p.setPen(QPen(edge, 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRect(area.left() + 1, y + 1, area.width() - 2, h - 1))
            tag = "" if r == 0 else (f"#{r + 1}" if r > 0 else "?")
            tag_w = fm.horizontalAdvance(tag) + 6 if tag else 0
            p.setPen(dark if bright else light)
            p.setFont(bold)
            text = bfm.elidedText(visible(rec.piece(int(s["chosen"][i]))), Qt.ElideRight,
                                  area.width() - 10 - tag_w)
            p.drawText(QRect(area.left() + 5, y, area.width() - 10 - tag_w, self.row_px),
                       Qt.AlignLeft | Qt.AlignVCenter, text)
            p.setFont(self.font())
            if tag:
                p.setPen(edge if not bright else dark)
                p.drawText(QRect(area.right() - tag_w - 3, y, tag_w, self.row_px),
                           Qt.AlignRight | Qt.AlignVCenter, tag)

    def _paint_gutter(self, p: QPainter, area: QRect, words: bool, first: int, last: int):
        """Step numbers (when there is room) and a ◆ at every moment of doubt."""
        doubt = self._doubt
        p.fillRect(area, QColor(12, 15, 20))
        muted, amber = QColor(110, 122, 146), QColor(255, 170, 60)
        for i in range(first, last):
            y = area.top() + (i - first) * self.row_px
            d = bool(len(doubt) > i and doubt[i])
            if words:
                p.setPen(amber if d else muted)
                p.drawText(QRect(area.left(), y, area.width() - 4, self.row_px),
                           Qt.AlignRight | Qt.AlignVCenter, f"{i + 1}{' ◆' if d else '  '}")
            elif d:
                p.fillRect(area.left() + 2, y, area.width() - 3, max(1, self.row_px - 1), amber)

    def _level(self, logprob: float) -> float:
        lv = float(logprob) * DB_PER_NAT
        if not math.isfinite(lv):
            return 0.0
        return min(1.0, max(0.0, (lv - self.floor_db) / (0.0 - self.floor_db)))

    def _bright(self, step: int, col: int) -> bool:
        rgb = _LUT[int(self._level(self._rec.steps["logprobs"][step][col]) * 255)]
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 140

    def _paint_words(self, p: QPainter, grid: QRect, colw: float, first: int, last: int):
        """Each candidate's text in its cell — dark on the bright cells, light
        on the dark ones — so a row reads as the words the model weighed."""
        rec = self._rec
        fm = QFontMetrics(p.font())
        s = rec.steps
        light, dark = QColor(226, 232, 244), QColor(12, 16, 26)
        for i in range(first, last):
            y = grid.top() + (i - first) * self.row_px
            ids, lps = s["ids"][i], s["logprobs"][i]
            for c in range(self._img.width()):
                x0 = grid.left() + int(c * colw)
                w = grid.left() + int((c + 1) * colw) - x0
                rgb = _LUT[int(self._level(lps[c]) * 255)]
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                p.setPen(dark if lum > 140 else light)
                text = fm.elidedText(visible(rec.piece(int(ids[c]))), Qt.ElideRight, w - 6)
                p.drawText(QRect(x0 + 3, y, w - 6, self.row_px), Qt.AlignLeft | Qt.AlignVCenter,
                           text)

    # ------------------------------------------------------------- input
    def wheelEvent(self, ev):
        rec = self._rec
        if rec is None:
            return
        if ev.modifiers() & Qt.ControlModifier:
            self.row_px = max(1, min(32, self.row_px + (1 if ev.angleDelta().y() > 0 else -1)))
        else:
            rows = -ev.angleDelta().y() / 120 * 6
            self.top = max(0.0, min(float(max(0, rec.n_steps - 5)), self.top + rows))
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            h = self.hit(int(ev.position().x()), int(ev.position().y()))
            if h:
                self.stepClicked.emit(h["step"])

    def mouseMoveEvent(self, ev):
        pos = ev.position().toPoint()
        h = self.hit(pos.x(), pos.y())
        if h is None or h["candidate"] is None or self._rec is None:
            QToolTip.hideText()
            return
        row = self._rec.steps[h["step"]]
        if h["candidate"] == "taken":
            c = self._rec.chosen(h["step"])
            p = f"{c['p']:.2%}" if c["p"] is not None else "?"
            how = ("the model's favourite" if c["rank"] == 0 else
                   f"the model's {_ordinal(c['rank'] + 1)} choice" if c["rank"] > 0 else
                   "rank unknown")
            QToolTip.showText(ev.globalPosition().toPoint(),
                              f"step {h['step'] + 1} · taken: {visible(c['piece'])}\n"
                              f"{p} — {how}", self)
            return
        c = h["candidate"]
        tid = int(row["ids"][c])
        lp = float(row["logprobs"][c])
        chosen = " ◀ taken" if tid == int(row["chosen"]) else ""
        QToolTip.showText(ev.globalPosition().toPoint(),
                          f"step {h['step'] + 1} · {_ordinal(c + 1)} choice · "
                          f"{visible(self._rec.piece(tid))!s}\n"
                          f"{math.exp(lp):.2%}   ({lp * DB_PER_NAT:.1f} dB){chosen}", self)

    def sizeHint(self):
        return QSize(520, 420)


# ===================================================== candidates at cursor
class CandidateList(QWidget):
    """The candidates at the cursor, best first: a bar for probability, the
    token's text with its invisible parts shown, and the one chosen marked."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list = []
        self._head = ""
        self.n = 20
        self.setMinimumWidth(220)

    def show_candidates(self, head: str, rows: list):
        self._head, self._rows = head, rows
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        fm = QFontMetrics(self.font())
        return QSize(300, (self.n + 2) * (fm.height() + 4))

    def paintEvent(self, _ev):
        p = QPainter(self)
        pal = self.palette()
        fm = QFontMetrics(self.font())
        lh = fm.height() + 4
        p.setPen(pal.color(QPalette.Text))
        f = QFont(self.font())
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRect(4, 2, self.width() - 8, lh), Qt.AlignLeft | Qt.AlignVCenter,
                   QFontMetrics(f).elidedText(self._head, Qt.ElideRight, self.width() - 8))
        p.setFont(self.font())
        barw = self.width() - 8
        for i, c in enumerate(self._rows):
            y = (i + 1) * lh + 4
            w = int(barw * c["p"])
            col = QColor(255, 150, 40, 150) if c["chosen"] else QColor(76, 141, 255, 110)
            p.fillRect(4, y + 2, max(1, w), lh - 4, col)
            p.setPen(pal.color(QPalette.Text))
            mark = "▶ " if c["chosen"] else "   "
            pct = f"{100 * c['p']:6.2f}%"
            p.drawText(QRect(6, y, barw - 4, lh), Qt.AlignLeft | Qt.AlignVCenter,
                       f"{mark}{pct}   {visible(c['piece'])}")
        p.end()


# ================================================================ the player
class WaterfallPlayer(QWidget):
    """Load a recording (``load(path)``) and play it."""

    stepChanged = Signal(int)

    def __init__(self, recording=None, parent=None):
        super().__init__(parent)
        self._rec: Recording | None = None
        self._step = 0
        self._speed = 1.0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick)

        self.title = QLabel("No recording")
        self.title.setWordWrap(True)
        self.text = ReplyView()
        self.view = WaterfallView()
        self.cands = CandidateList()
        from .experts import ExpertPanel
        self.experts = ExpertPanel()
        self.experts.hide()                 # shown when a recording carries the Tap's experts
        self.info = QLabel("")
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.btn_start = QPushButton("⏮")
        self.btn_back = QPushButton("◀")
        self.btn_play = QPushButton("▶")
        self.btn_fwd = QPushButton("▶|")
        self.btn_end = QPushButton("⏭")
        for b, tip in ((self.btn_start, "first token (Home)"), (self.btn_back, "back one (←)"),
                       (self.btn_play, "play / pause (Space)"), (self.btn_fwd, "forward one (→)"),
                       (self.btn_end, "last token (End)")):
            b.setToolTip(tip)
            b.setFixedWidth(40)
        self.slider = QSlider(Qt.Horizontal)
        self.speed = QComboBox()
        for name, v in SPEEDS:
            self.speed.addItem(name, v)
        self.axis = QComboBox()
        self.axis.addItem("candidates by rank", "rank")
        self.axis.addItem("candidates by token id", "id")
        self.ranks = QComboBox()
        for r in RANKS:
            self.ranks.addItem(f"{r} ranks", r)
        self.ranks.setCurrentIndex(RANKS.index(64))
        self.floor = QDoubleSpinBox()
        self.floor.setRange(-120.0, -5.0)
        self.floor.setSingleStep(5.0)
        self.floor.setValue(-40.0)
        self.floor.setSuffix(" dB floor")
        self.read_btn = QPushButton("Aa  read")
        self.read_btn.setCheckable(True)
        self.read_btn.setToolTip(
            "Zoom in until every cell shows its candidate's text: each row then reads as "
            "the words the model was choosing between, best on the left, the one it took "
            "framed. Press again for the overview. (Ctrl+wheel zooms by hand.)")
        self.reveal = QCheckBox("hide what's to come")
        self.reveal.setChecked(False)
        self.reveal.setToolTip(
            "Off (the default): the whole reply is shown, what is still to come dimmed, "
            "so you can see where a word was heading while you look at why it was "
            "picked.\nOn: the reply is written out as the cursor moves — watch it the "
            "way it was written, without knowing the ending.")
        self.mark = QCheckBox("underline doubts")
        self.mark.setChecked(True)
        self.mark.setToolTip("A wavy amber line under every word the model was unsure of "
                             "(it did not take its favourite, or its favourite had less "
                             "than half the probability). Hover a word to see how sure.")
        self.btn_doubt_back = QPushButton("◆◀")
        self.btn_doubt_fwd = QPushButton("▶◆")
        for b, tip in ((self.btn_doubt_back, "the previous moment of doubt ( [ )"),
                       (self.btn_doubt_fwd, "the next moment of doubt ( ] )")):
            b.setToolTip(tip + " — where the model did not take its favourite, or its "
                         "favourite had less than half the probability")
            b.setFixedWidth(48)
        self._doubts = np.zeros(0, np.int64)
        self._overview = (self.view.row_px, 64)

        transport = QHBoxLayout()
        for w in (self.btn_start, self.btn_doubt_back, self.btn_back, self.btn_play,
                  self.btn_fwd, self.btn_doubt_fwd, self.btn_end):
            transport.addWidget(w)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.speed)
        options = QHBoxLayout()
        options.addWidget(self.axis)
        options.addWidget(self.ranks)
        options.addWidget(self.floor)
        options.addWidget(self.read_btn)
        options.addWidget(self.reveal)
        options.addWidget(self.mark)
        options.addStretch(1)
        options.addWidget(self.info)

        side = QSplitter(Qt.Vertical)
        side.addWidget(self.cands)
        side.addWidget(self.experts)
        side.setStretchFactor(0, 1)
        side.setStretchFactor(1, 2)
        middle = QSplitter(Qt.Horizontal)
        middle.addWidget(self.view)
        middle.addWidget(side)
        middle.setStretchFactor(0, 3)
        middle.setStretchFactor(1, 1)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.text)
        split.addWidget(middle)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)

        lay = QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addWidget(split, 1)
        lay.addLayout(options)
        lay.addLayout(transport)

        self.btn_start.clicked.connect(lambda: self.set_step(0))
        self.btn_back.clicked.connect(lambda: self.set_step(self._step - 1))
        self.btn_fwd.clicked.connect(lambda: self.set_step(self._step + 1))
        self.btn_end.clicked.connect(lambda: self.set_step(self.n_steps - 1))
        self.btn_play.clicked.connect(self.toggle)
        self.btn_doubt_back.clicked.connect(lambda: self.next_doubt(-1))
        self.btn_doubt_fwd.clicked.connect(lambda: self.next_doubt(+1))
        self.read_btn.toggled.connect(self.set_reading)
        self.reveal.toggled.connect(self._reveal_changed)
        self.mark.toggled.connect(self._mark_changed)
        self.slider.valueChanged.connect(lambda v: self.set_step(v, from_slider=True))
        self.speed.currentIndexChanged.connect(lambda _i: self.set_speed(self.speed.currentData()))
        self.axis.currentIndexChanged.connect(lambda _i: self._view_changed())
        self.ranks.currentIndexChanged.connect(lambda _i: self._view_changed())
        self.floor.valueChanged.connect(lambda _v: self._view_changed())
        self.view.stepClicked.connect(self.set_step)
        self.text.tokenClicked.connect(self.set_step)
        self.experts.map.stepClicked.connect(self.set_step)
        self.experts.map.piece_of = self._piece_at
        self.setFocusPolicy(Qt.StrongFocus)
        if recording is not None:
            self.load(recording)

    # ------------------------------------------------------------ public
    @property
    def recording(self) -> Recording | None:
        return self._rec

    @property
    def n_steps(self) -> int:
        return self._rec.n_steps if self._rec else 0

    @property
    def step(self) -> int:
        return self._step

    def load(self, recording):
        """A Recording, or a path to one."""
        self.pause()
        rec = recording if isinstance(recording, Recording) or recording is None else read(
            Path(recording))
        self._rec = rec
        self._step = 0
        self._doubts = rec.doubts() if rec is not None else np.zeros(0, np.int64)
        self.text.load(rec)
        self.view.load(rec)
        tap_note = self._load_experts(rec)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self.n_steps - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        if rec is None:
            self.title.setText("No recording")
        else:
            st = rec.settings
            knobs = ", ".join(f"{k} {v}" for k, v in st.items()
                              if v is not None and k in ("temperature", "top_k", "top_p", "min_p",
                                                         "repeat_penalty", "seed"))
            timing = rec.meta.get("timing") or {}
            tps = timing.get("tokens_per_second")
            err = f"   ⚠ {rec.error}" if rec.error else ""
            self.title.setText(
                f"<b>{rec.title}</b> — {rec.n_steps} tokens"
                + (f", {tps:.1f} tokens/s" if tps else "")
                + (f", finish: {rec.finish_reason}" if rec.finish_reason else "")
                + (f"<br><span style='color:gray'>{knobs}</span>" if knobs else "") + err
                + tap_note)
        self._show()

    def _piece_at(self, step: int) -> str | None:
        rec = self._rec
        if rec is None or not 0 <= step < rec.n_steps:
            return None
        return visible(rec.piece(int(rec.steps["chosen"][step])))

    def _load_experts(self, rec) -> str:
        """Show the expert map when the recording carries the Tap's experts;
        a line for the title saying what the Tap recorded, or why not."""
        routing, note = None, ""
        info = rec.tap_info if rec is not None else None
        if info:
            kinds = sorted({s.get("kind") for s in info.get("layout") or [] if s.get("kind")})
            if kinds:
                note = "⚗ the Tap: " + ", ".join(kinds)
            if "ffn_moe_topk" in kinds:       # only then is the Tap's file read here
                try:
                    tap = rec.tap
                    routing = tap.experts() if tap is not None else None
                except Exception as exc:      # a damaged Tap file costs the map, not the player
                    note = f"⚠ the Tap's file could not be read: {exc}"
            for key in ("unavailable", "error"):
                if info.get(key):
                    note += f"{' · ' if note else ''}⚠ the Tap: {info[key]}"
            if info.get("stopped_at") is not None:
                note += f"{' · ' if note else ''}the Tap stopped at step {info['stopped_at'] + 1}"
        groups = {}
        if routing is not None:
            thinking = np.zeros(rec.n_steps, bool)
            for a in rec.meta.get("annotations") or []:
                if a.get("athrec:label") == "thinking":
                    s0 = int(a.get("athrec:step_start", 0))
                    thinking[s0:s0 + int(a.get("athrec:step_count", 0))] = True
            if thinking.any():
                groups = {"thinking": np.flatnonzero(thinking),
                          "answer": np.flatnonzero(~thinking)}
        self.experts.load(routing, groups)
        self.experts.setVisible(routing is not None)
        return f"<br><span style='color:gray'>{note}</span>" if note else ""

    def set_step(self, step: int, from_slider: bool = False):
        if self._rec is None or self.n_steps == 0:
            return
        step = max(0, min(int(step), self.n_steps - 1))
        changed = step != self._step
        self._step = step
        if not from_slider:
            self.slider.blockSignals(True)
            self.slider.setValue(step)
            self.slider.blockSignals(False)
        self._show()
        if changed:
            self.stepChanged.emit(step)

    def next_doubt(self, direction: int = 1) -> int | None:
        """Move to the next (or previous) moment of doubt; returns its step,
        or None when there is none that way."""
        d = self._doubts
        if not len(d):
            return None
        if direction > 0:
            i = int(np.searchsorted(d, self._step, side="right"))
            target = int(d[i]) if i < len(d) else None
        else:
            i = int(np.searchsorted(d, self._step, side="left")) - 1
            target = int(d[i]) if i >= 0 else None
        if target is not None:
            self.pause()
            self.set_step(target)
        return target

    def set_reading(self, on: bool):
        """The reading zoom (every cell's text shown) or the overview."""
        if self.read_btn.isChecked() != bool(on):
            self.read_btn.setChecked(bool(on))       # re-enters through toggled
            return
        if on:
            self._overview = (self.view.row_px, int(self.ranks.currentData()))
            row_px, ranks = self.view.reading_zoom()
        else:
            row_px, ranks = self._overview
        self.view.row_px = row_px
        i = self.ranks.findData(ranks)
        if i >= 0 and i != self.ranks.currentIndex():
            self.ranks.setCurrentIndex(i)            # rebuilds through _view_changed
        else:
            self._view_changed()
        self.view.set_step(self._step)

    def _mark_changed(self, on: bool):
        self.text.mark_doubts = bool(on)
        self.text.load(self._rec)
        self.text.show_step(self._step)

    def _reveal_changed(self, on: bool):
        self.text.reveal = bool(on)
        self.text.show_step(self._step)

    def set_speed(self, v: float):
        self._speed = float(v)
        i = next((j for j, (_n, s) in enumerate(SPEEDS) if s == self._speed), None)
        if i is not None and self.speed.currentIndex() != i:
            self.speed.setCurrentIndex(i)

    def play(self):
        if self._rec is None or self.n_steps == 0:
            return
        if self._step >= self.n_steps - 1:
            self.set_step(0)
        self.btn_play.setText("⏸")
        self._schedule()

    def pause(self):
        self._timer.stop()
        self.btn_play.setText("▶")

    def is_playing(self) -> bool:
        return self._timer.isActive()

    def toggle(self):
        self.pause() if self.is_playing() else self.play()

    def delay_ms(self, step: int) -> int:
        """How long to wait before showing ``step + 1``."""
        if self._speed < 0:                                   # a fixed rate
            return int(1000 / -self._speed)
        t = self._rec.steps["t"]
        if step + 1 >= len(t):
            return 0
        gap = float(t[step + 1] - t[step]) / max(self._speed, 1e-6)
        return int(round(max(5.0, min(gap * 1000, 5000.0))))

    # ----------------------------------------------------------- private
    def _schedule(self):
        if self._step >= self.n_steps - 1:
            self.pause()
            return
        self._timer.start(self.delay_ms(self._step))

    def _tick(self):
        self.set_step(self._step + 1)
        if self.btn_play.text() == "⏸":
            self._schedule()

    def _view_changed(self):
        self.view.axis = self.axis.currentData()
        self.view.ranks = int(self.ranks.currentData())
        self.view.floor_db = float(self.floor.value())
        self.ranks.setEnabled(self.view.axis == "rank")
        self.view.rebuild()

    def _show(self):
        rec = self._rec
        if rec is None or self.n_steps == 0:
            self.cands.show_candidates("", [])
            self.info.setText("")
            self.view.update()
            self.text.show_step(0)
            return
        i = self._step
        c = rec.chosen(i)
        self.text.show_step(i)
        self.view.set_step(i)
        if self.experts.isVisibleTo(self):
            self.experts.set_step(i)
        rank = ("its favourite" if c["rank"] == 0 else
                f"its {_ordinal(c['rank'] + 1)} choice" if c["rank"] > 0 else "rank unknown")
        p = f"{100 * c['p']:.2f}%" if c["p"] is not None else "?"
        end = " · end" if c["flags"] & FLAG_EOG else ""
        self.cands.show_candidates(f"{visible(c['piece'])}   {p}, {rank}",
                                   rec.candidates(i, self.cands.n))
        doubt = " · ◆ doubt" if len(self._doubts) and i in self._doubts else ""
        self.info.setText(f"step {i + 1}/{self.n_steps} · {c['t']:.2f} s · "
                          f"{c['entropy_bits']:.2f} bits{end}{doubt} · "
                          f"{len(self._doubts)} doubts")
        self.info.setToolTip(
            "the step (token) at the cursor · seconds since the recording began · the "
            "entropy of the whole distribution at that step, in bits (how spread the "
            "model's choice was) · how many moments of doubt the reply has")

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key_Space:
            self.toggle()
        elif k == Qt.Key_Left:
            self.set_step(self._step - 1)
        elif k == Qt.Key_Right:
            self.set_step(self._step + 1)
        elif k == Qt.Key_Home:
            self.set_step(0)
        elif k == Qt.Key_End:
            self.set_step(self.n_steps - 1)
        elif k == Qt.Key_BracketRight:
            self.next_doubt(+1)
        elif k == Qt.Key_BracketLeft:
            self.next_doubt(-1)
        else:
            super().keyPressEvent(ev)
