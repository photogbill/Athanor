"""The expert map (M28) — a mixture-of-experts model's routing, beside the Waterfall.

For every token, every layer's router scores every expert and sends the
token to the few that score highest. The map shows that as a second, small
waterfall turned on its side: layers down, experts across, brightness for
the router's score (in dB, like the Waterfall), the experts actually used
framed in amber with how much each counted.

Four ways to look:
* **at the cursor** — every layer, for the token under the Waterfall's cursor;
* **over the reply** (or over its thinking, or its answer) — how often each
  expert was used, as a fraction of the tokens; the cursor's experts stay
  framed;
* **one layer over time** — the second waterfall: time down, that layer's
  experts across, the ones used framed on every row, the cursor's row
  outlined; it scrolls with the cursor, and a click moves it;
* hover a cell for its numbers (and, over time, the token).

What it is not: a map of personas. Which experts a router uses tracks the
geometry of the hidden state, which is often closer to syntax than to
topic (Mixtral's own analysis, 2024). The map shows what was used; what it
means is the question it lets you ask.

Reads ``Recording.tap.experts()`` only — nothing needs the model.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QSizePolicy, QSpinBox,
                               QToolTip, QVBoxLayout, QWidget)

from ..waterfall.reading import LOGIT_ROUTERS, ExpertRouting
from .waterfall import _LUT, STOPS

AMBER = QColor(255, 150, 40)
FLOOR_DB = -30.0


def router_levels(routing: ExpertRouting, step: int | None = None,
                  layer: int | None = None) -> np.ndarray | None:
    """The router's scores for drawing, 0..1: as probabilities (a softmax
    first when the router gives logits), in dB from FLOOR_DB to 0.
    ``step``: [layers, experts] at that step; ``layer`` (an index into
    ``routing.layers``): [steps, experts] for that layer over the reply."""
    if routing.probs is None:
        return None
    p = (routing.probs[step] if layer is None else routing.probs[:, layer]).astype(np.float64)
    if routing.arch in LOGIT_ROUTERS or (np.isfinite(p).any() and np.nanmin(p) < 0):
        m = np.nanmax(p, axis=1, keepdims=True)
        e = np.exp(p - m)
        p = e / np.nansum(e, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.clip(p, 1e-12, None))
    return np.nan_to_num(np.clip((db - FLOOR_DB) / -FLOOR_DB, 0.0, 1.0))


class ExpertMap(QWidget):
    """The grid: layers down, experts across — or, over time, steps down and
    one layer's experts across."""

    MODES = ("cursor", "reply", "thinking", "answer", "time")
    stepClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.routing: ExpertRouting | None = None
        self.step = 0
        self.mode = "cursor"
        self._groups: dict = {}          # mode -> step indices
        self._usage: dict = {}           # mode -> [layers, experts]
        self._img: QImage | None = None
        self._img_bytes = None
        self._grid = QRect()
        self.layer = 0                   # an index into routing.layers, for "time"
        self._time: dict = {}            # layer index -> [steps, experts] levels
        self._window = (0, 0, 1.0)       # time mode: first step, rows, row height
        self.piece_of = None             # step -> the token's text (for tooltips)

    # ------------------------------------------------------------ data
    def load(self, routing: ExpertRouting | None, groups: dict | None = None):
        """``groups``: mode -> the steps it covers ('thinking', 'answer')."""
        self.routing = routing
        self.step = 0
        self._groups = dict(groups or {})
        self._usage = {}
        self._time = {}
        self.layer = 0
        self._rebuild()

    def set_step(self, step: int):
        if self.routing is None:
            return
        self.step = max(0, min(int(step), self.routing.n_steps - 1))
        if self.mode in ("cursor", "time"):
            self._rebuild()
        else:
            self.update()

    def set_layer(self, index: int):
        if self.routing is None:
            return
        self.layer = max(0, min(int(index), len(self.routing.layers) - 1))
        self._rebuild()

    def set_mode(self, mode: str):
        self.mode = mode if mode in self.MODES else "cursor"
        self._rebuild()

    def usage(self, mode: str | None = None) -> np.ndarray | None:
        mode = mode or self.mode
        if self.routing is None or mode in ("cursor", "time"):
            return None
        if mode not in self._usage:
            steps = self._groups.get(mode) if mode != "reply" else None
            self._usage[mode] = self.routing.usage(steps)
        return self._usage[mode]

    def levels(self) -> np.ndarray | None:
        """What is drawn, [layers, experts] in 0..1."""
        r = self.routing
        if r is None or r.n_steps == 0:
            return None
        if self.mode == "cursor":
            lv = router_levels(r, self.step)
            if lv is None:                       # no router scores: the chosen, lit by share
                lv = r.grid(self.step)["chosen"]
            return lv
        if self.mode == "time":
            return self.time_levels()
        u = self.usage()
        top = float(u.max()) if u is not None and u.size else 0.0
        return u / top if top > 0 else u

    def time_levels(self, index: int | None = None) -> np.ndarray | None:
        """[steps, experts] for one layer over the reply: the router's scores,
        or (without them) each used expert's share."""
        r = self.routing
        li = self.layer if index is None else index
        if r is None:
            return None
        if li not in self._time:
            lv = router_levels(r, layer=li)
            if lv is None:
                lv = self._chosen_over_time(li)
            self._time[li] = lv
        return self._time[li]

    def _chosen_over_time(self, li: int) -> np.ndarray:
        r = self.routing
        out = np.zeros((r.n_steps, r.n_expert), np.float32)
        share = r.share()
        ids = r.ids[:, li, :]
        for j in range(ids.shape[1]):
            ok = (ids[:, j] >= 0) & (ids[:, j] < r.n_expert)
            rows = np.flatnonzero(ok)
            out[rows, ids[rows, j]] = share[rows, li, j] if share is not None else 1.0
        return out

    def _time_window(self, grid_h: int) -> tuple:
        """(first step, rows shown, row height) — centred on the cursor."""
        n = self.routing.n_steps
        row_h = max(3.0, min(14.0, grid_h / max(1, n)))
        rows = max(1, min(n, int(grid_h // row_h)))
        first = max(0, min(self.step - rows // 2, n - rows))
        return first, rows, row_h

    def _rebuild(self):
        lv = self.levels()
        if lv is not None and self.mode == "time":
            first, rows, _h = self._time_window(self._grid_height())
            lv = lv[first:first + rows]
        if lv is None:
            self._img = None
            self.update()
            return
        lv = np.nan_to_num(np.asarray(lv, np.float64), nan=0.5, posinf=1.0, neginf=0.0)
        rgb = np.ascontiguousarray(_LUT[(np.clip(lv, 0, 1) * 255).astype(np.int32)], np.uint8)
        h, w = lv.shape
        self._img_bytes = rgb.tobytes()
        self._img = QImage(self._img_bytes, w, h, 3 * w, QImage.Format_RGB888)
        self.update()

    # --------------------------------------------------------- geometry
    def _grid_height(self) -> int:
        """The grid's height before any cap: the widget less the header and
        the expert numbers under the grid."""
        fm = QFontMetrics(self.font())
        return max(1, self._layout().height() - (fm.height() + 2))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.mode == "time":
            self._rebuild()                  # how many rows fit changed

    def _layout(self) -> QRect:
        fm = QFontMetrics(self.font())
        left = fm.horizontalAdvance("L000") + 6
        top = fm.height() + 4
        return self.rect().adjusted(left, top, -4, -4)

    def cell_at(self, x: int, y: int):
        """(layer index, expert) — or, over time, (step, expert) as
        ('step', step, expert) — under a point; None outside the grid."""
        r = self.routing
        g = self._grid
        if r is None or not g.contains(x, y):
            return None
        X = r.n_expert
        e = min(int((x - g.left()) * X / max(1, g.width())), X - 1)
        if self.mode == "time":
            first, rows, row_h = self._window
            step = first + int((y - g.top()) / max(row_h, 1e-6))
            return ("step", min(step, r.n_steps - 1), e)
        L = len(r.layers)
        li = int((y - g.top()) * L / max(1, g.height()))
        return (min(li, L - 1), e)

    # ------------------------------------------------------------ paint
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().color(QPalette.Window))
        r = self.routing
        if r is None or self._img is None:
            p.setPen(self.palette().color(QPalette.PlaceholderText))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap,
                       "No expert routing in this recording.\n(Record a mixture-of-experts "
                       "model with the Tap: experts.)")
            return
        if self.mode == "time":
            self._paint_time(p)
        else:
            self._paint_layers(p)
        p.end()

    def _paint_columns(self, p: QPainter, g: QRect, cell_w: float):
        """Expert numbers under the grid, as many as fit."""
        fm = QFontMetrics(self.font())
        X = self.routing.n_expert
        every = max(1, math.ceil((fm.horizontalAdvance(str(X - 1)) + 6) / max(cell_w, 1e-6)))
        for e in range(0, X, every):
            x = g.left() + e * cell_w
            p.drawText(QRect(int(x), g.bottom() + 2, int(max(cell_w * every, cell_w)),
                             fm.height()), Qt.AlignLeft | Qt.AlignTop, str(e))

    def _paint_layers(self, p: QPainter):
        r = self.routing
        fm = QFontMetrics(self.font())
        g = self._layout().adjusted(0, 0, 0, -(fm.height() + 2))
        L, X = len(r.layers), r.n_expert
        cell_h = g.height() / L
        cell_w = g.width() / X
        if cell_h > 22:                                   # a few layers: don't stretch them
            g.setHeight(int(22 * L))
            cell_h = 22.0
        self._grid = g
        p.fillRect(g, QColor(*STOPS[0][1]))
        p.drawImage(g, self._img)
        p.setPen(self.palette().color(QPalette.Text))
        every = max(1, math.ceil(fm.height() / max(cell_h, 1e-6)))
        for li in range(0, L, every):
            y = g.top() + li * cell_h
            p.drawText(QRect(0, int(y), g.left() - 4, max(int(cell_h * every), fm.height())),
                       Qt.AlignRight | Qt.AlignTop, str(r.layers[li]))
        head = {"cursor": f"step {self.step + 1}: the router, every layer",
                "reply": "how often each expert was used, over the reply",
                "thinking": "how often each expert was used, over the thinking",
                "answer": "how often each expert was used, over the answer"}[self.mode]
        p.drawText(QRect(g.left(), 0, g.width(), fm.height() + 2),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(head + "  ·  layers ↓ experts →", Qt.ElideRight, g.width()))
        self._paint_columns(p, g, cell_w)
        # the experts used at the cursor: amber frames, and how much each counted
        grid = r.grid(self.step)["chosen"]
        pen = QPen(AMBER if self.mode == "cursor" else QColor(255, 150, 40, 170))
        pen.setWidthF(2.0 if min(cell_w, cell_h) >= 6 else 1.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        small = QFont(self.font())
        small.setPointSizeF(max(6.0, self.font().pointSizeF() * 0.75))
        for li, e in zip(*np.nonzero(grid)):
            x = g.left() + e * cell_w
            y = g.top() + li * cell_h
            p.drawRect(QRect(int(x), int(y), max(2, int(cell_w)), max(2, int(cell_h))))
            if self.mode == "cursor" and cell_w >= 28 and cell_h >= 12 and \
                    np.isfinite(grid[li, e]):
                p.setFont(small)
                p.setPen(QColor(20, 20, 20))
                p.drawText(QRect(int(x), int(y), int(cell_w), int(cell_h)), Qt.AlignCenter,
                           f"{100 * grid[li, e]:.0f}%")
                p.setPen(pen)
                p.setFont(self.font())

    def _paint_time(self, p: QPainter):
        r = self.routing
        fm = QFontMetrics(self.font())
        g = self._layout().adjusted(0, 0, 0, -(fm.height() + 2))
        X = r.n_expert
        first, rows, row_h = self._time_window(self._grid_height())
        self._window = (first, rows, row_h)
        g.setHeight(int(rows * row_h))
        cell_w = g.width() / X
        self._grid = g
        p.fillRect(g, QColor(*STOPS[0][1]))
        p.drawImage(g, self._img)
        p.setPen(self.palette().color(QPalette.Text))
        p.drawText(QRect(g.left(), 0, g.width(), fm.height() + 2),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(f"layer {r.layers[self.layer]} over time  ·  tokens ↓ "
                                 "experts →", Qt.ElideRight, g.width()))
        every = max(1, math.ceil(fm.height() / row_h))
        for i in range(0, rows, every):
            y = g.top() + i * row_h
            p.drawText(QRect(0, int(y), g.left() - 4, fm.height()), Qt.AlignRight | Qt.AlignTop,
                       str(first + i + 1))
        self._paint_columns(p, g, cell_w)
        ids = r.ids[first:first + rows, self.layer, :]
        pen = QPen(AMBER)
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        for i in range(len(ids)):
            y = g.top() + i * row_h
            for e in ids[i]:
                if 0 <= e < X:
                    p.drawRect(QRect(int(g.left() + e * cell_w), int(y),
                                     max(2, int(cell_w) - 1), max(2, int(row_h) - 1)))
        cur = self.step - first
        if 0 <= cur < rows:
            p.setPen(QPen(QColor(255, 255, 255), 1.5))
            y = g.top() + cur * row_h
            p.drawRect(QRect(g.left() - 1, int(y) - 1, g.width() + 1, int(row_h) + 1))

    def mousePressEvent(self, ev):
        pos = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        c = self.cell_at(pos.x(), pos.y())
        if c is not None and c[0] == "step":
            self.stepClicked.emit(int(c[1]))

    def mouseMoveEvent(self, ev):
        pos = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        c = self.cell_at(pos.x(), pos.y())
        r = self.routing
        if c is None or r is None:
            QToolTip.hideText()
            return
        if c[0] == "step":
            _tag, step, e = c
            li = self.layer
            text = self.piece_of(step) if self.piece_of else None
            lines = [f"step {step + 1}" + (f" · {text!r}" if text is not None else "")
                     + f" · layer {r.layers[li]} · expert {e}"]
            lv = router_levels(r, step)
            if lv is not None:
                db = FLOOR_DB + float(lv[li, e]) * -FLOOR_DB
                lines.append(f"router: {10 ** (db / 10):.2%} ({db:.1f} dB)")
            ch = r.grid(step)["chosen"][li, e]
            lines.append("used (how much it counted is unknown)" if not np.isfinite(ch) else
                         f"used, counting {ch:.0%}" if ch > 0 else "not used")
            lines.append("click: move the cursor here")
            QToolTip.showText(ev.globalPosition().toPoint() if hasattr(ev, "globalPosition")
                              else ev.globalPos(), "\n".join(lines), self)
            return
        li, e = c
        lines = [f"layer {r.layers[li]} · expert {e}"]
        chosen = r.grid(self.step)["chosen"][li, e]
        if r.probs is not None:
            lv = router_levels(r, self.step)
            if lv is not None:
                db = FLOOR_DB + float(lv[li, e]) * -FLOOR_DB
                lines.append(f"router at step {self.step + 1}: {10 ** (db / 10):.2%} "
                             f"({db:.1f} dB)" + (" — the floor" if lv[li, e] <= 0 else ""))
        if not np.isfinite(chosen):
            lines.append(f"used at step {self.step + 1} (how much it counted is unknown)")
        else:
            lines.append(f"used at step {self.step + 1}, counting {chosen:.0%}" if chosen > 0
                         else f"not used at step {self.step + 1}")
        u = self.usage("reply")
        if u is not None and np.isfinite(u[li, e]):
            lines.append(f"used for {u[li, e]:.0%} of the reply's tokens")
        for mode in ("thinking", "answer"):
            if mode in self._groups:
                lines.append(f"… {self.usage(mode)[li, e]:.0%} of its {mode}")
        QToolTip.showText(ev.globalPosition().toPoint() if hasattr(ev, "globalPosition")
                          else ev.globalPos(), "\n".join(lines), self)

    def sizeHint(self):
        return QSize(320, 280)


class ExpertPanel(QWidget):
    """The map with its mode chooser and a line of what the cursor shows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.map = ExpertMap()
        self.mode = QComboBox()
        self.mode.addItem("at the cursor", "cursor")
        self.mode.addItem("over the reply", "reply")
        self.mode.addItem("one layer, over time", "time")
        self.layer = QSpinBox()
        self.layer.setPrefix("layer ")
        self.layer.setToolTip("The layer shown over time")
        self.layer.hide()
        self.mode.setToolTip(
            "At the cursor: the router's score for every expert at the token under the "
            "cursor (brightness, in dB), the ones used framed with how much each counted.\n"
            "Over the reply (or its thinking, or its answer): how often each expert was "
            "used, as a share of the tokens — the cursor's experts stay framed.\n"
            "One layer, over time: a second waterfall — tokens down, that layer's experts "
            "across, the ones used framed; click a row to move the cursor.")
        self.title = QLabel("<b>Experts</b>")
        self.line = QLabel("")
        self.line.setWordWrap(True)
        self.line.setToolTip("Experts are not personas: this is which feed-forward blocks the "
                             "router used, not what they mean.")
        controls = QHBoxLayout()
        controls.addWidget(self.mode)
        controls.addWidget(self.layer)
        controls.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.title)
        lay.addLayout(controls)
        lay.addWidget(self.map, 1)
        lay.addWidget(self.line)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.layer.valueChanged.connect(self._layer_changed)

    def load(self, routing: ExpertRouting | None, groups: dict | None = None):
        self.mode.blockSignals(True)
        while self.mode.count() > 3:
            self.mode.removeItem(3)
        for name in ("thinking", "answer"):
            if groups and name in groups and len(groups[name]):
                self.mode.addItem(f"over the {name}", name)
        self.mode.setCurrentIndex(0)
        self.mode.blockSignals(False)
        self.layer.blockSignals(True)
        layers = routing.layers if routing is not None else [0]
        self.layer.setRange(int(layers[0]), int(layers[-1]))
        self.layer.setValue(int(layers[0]))
        self.layer.blockSignals(False)
        self.layer.hide()
        self.map.mode = "cursor"
        self.map.load(routing, groups)
        if routing is not None:
            self.title.setText(f"<b>Experts</b> — {len(routing.layers)} × {routing.n_expert}, "
                               f"{routing.n_used} per token")
            self.title.setToolTip(f"{len(routing.layers)} layers, {routing.n_expert} experts in "
                                  f"each, {routing.n_used} used for every token")
        self._describe()

    def set_step(self, step: int):
        self.map.set_step(step)
        self._describe()

    def _mode_changed(self, _i):
        self.layer.setVisible(self.mode.currentData() == "time")
        self.map.set_mode(self.mode.currentData())
        self._describe()

    def _layer_changed(self, value: int):
        r = self.map.routing
        if r is None:
            return
        idx = min(range(len(r.layers)), key=lambda i: abs(r.layers[i] - value))
        self.map.set_layer(idx)

    def _describe(self):
        r = self.map.routing
        if r is None or r.n_steps == 0:
            self.line.setText("")
            return
        i = self.map.step
        if i == 0:
            self.line.setText(f"token 1 of {r.n_steps}")
            return
        same = sum(1 for li in range(len(r.layers))
                   if set(r.ids[i, li].tolist()) == set(r.ids[i - 1, li].tolist()))
        self.line.setText(f"token {i + 1}: the same experts as the token before in {same} of "
                          f"{len(r.layers)} layers")
