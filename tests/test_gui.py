"""The Waterfall player (PySide6, offscreen). Skipped where PySide6 is not
installed — the engine never needs it."""

import math
import os
import unittest

import numpy as np

from athanor.waterfall import Recorder, read
from fixtures import TempDir

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except Exception:                                      # pragma: no cover
    HAVE_QT = False

needs_qt = unittest.skipUnless(HAVE_QT, "PySide6 is not installed here")


def make_recording(folder, pieces=None, thinking=False):
    pieces = pieces or [b"<think>", b" hm", b"</think>", b" The", b" \xf0\x9f\x93\xa1",
                        b" is", b"\xe4", b"\xbd\xa0", b"\n", b"</s>"]
    n = len(pieces)
    vocab = {i: p for i, p in enumerate(pieces)}
    rng = np.random.default_rng(3)
    r = Recorder(n_vocab=n + 30, k=8, piece=lambda t: vocab.get(t, f"w{t}".encode()),
                 is_eog=lambda t: t == n - 1)
    for i in range(n):
        x = rng.standard_normal(n + 30).astype(np.float32)
        x[i] += 4.0 if i % 3 else 0.0                    # every third is not the favourite
        r.step(x, i, t=r.t0 + 0.1 * (i + 1))
    r.meta["model"] = {"name": "test.gguf"}
    return read(r.save(folder, finish_reason="stop"))


def make_tapped_recording(folder, n=12, layers=4, experts=8, probs=True):
    """A recording with the Tap's expert streams, made by hand."""
    from athanor.tap.track import TapTrack
    rng = np.random.default_rng(5)
    r = Recorder(n_vocab=40, k=8, piece=lambda t: f" w{t}".encode())
    r.tap_track = TapTrack(["experts"], facts={"arch": "llama", "n_expert": experts,
                                               "n_expert_used": 2})
    for i in range(n):
        r.step(rng.standard_normal(40).astype(np.float32), i % 40, t=r.t0 + 0.1 * (i + 1))
        rows = {}
        for li in range(layers):
            p = rng.dirichlet(np.ones(experts)).astype(np.float32)
            top = np.argsort(-p)[:2].astype(np.int32)
            rows[f"ffn_moe_topk-{li}"] = top
            rows[f"ffn_moe_weights-{li}"] = p[top]
            if probs:
                rows[f"ffn_moe_probs-{li}"] = p
        r.tap_track.step(i, rows)
    r.meta["model"] = {"name": "moe.gguf"}
    return read(r.save(folder, finish_reason="length"))


@needs_qt
class Experts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._t = TempDir()
        self.dir = self._t.__enter__()

    def tearDown(self):
        self._t.__exit__(None, None, None)

    def player(self, rec):
        from athanor.gui import WaterfallPlayer
        w = WaterfallPlayer(rec.meta_path)
        w.resize(1200, 700)
        w.show()
        self.app.processEvents()
        return w

    def test_the_map_appears_only_when_the_tap_recorded_experts(self):
        w = self.player(make_tapped_recording(self.dir))
        self.assertTrue(w.experts.isVisible())
        self.assertIn("the Tap", w.title.text())
        w.load(make_recording(self.dir))
        self.app.processEvents()
        self.assertFalse(w.experts.isVisible())

    def test_the_cursor_frames_the_experts_used(self):
        rec = make_tapped_recording(self.dir)
        w = self.player(rec)
        w.set_step(5)
        self.app.processEvents()
        m = w.experts.map
        self.assertEqual(m.step, 5)
        ex = rec.tap.experts()
        grid = ex.grid(5)["chosen"]
        self.assertEqual(int(np.count_nonzero(grid)), 4 * 2)
        lv = m.levels()
        self.assertEqual(lv.shape, (4, 8))
        for li in range(4):          # the used experts are the brightest: the router's top two
            used = set(np.flatnonzero(grid[li]).tolist())
            self.assertEqual(used, set(np.argsort(-lv[li])[:2].tolist()))
        self.assertIsNotNone(m.cell_at(m._grid.center().x(), m._grid.center().y()))
        img = m.grab().toImage()
        self.assertFalse(img.isNull())

    def test_over_the_reply_and_over_time(self):
        rec = make_tapped_recording(self.dir, n=30)
        w = self.player(rec)
        p = w.experts
        p.mode.setCurrentIndex(p.mode.findData("reply"))
        self.app.processEvents()
        u = p.map.usage()
        np.testing.assert_allclose(u.sum(axis=1), 2.0, rtol=1e-5, err_msg="2 used per token")
        p.mode.setCurrentIndex(p.mode.findData("time"))
        p.layer.setValue(2)
        self.app.processEvents()
        self.assertTrue(p.layer.isVisible())
        self.assertEqual(p.map.layer, 2)
        self.assertEqual(p.map.levels().shape, (30, 8))
        first, rows, row_h = p.map._window
        g = p.map._grid
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        QTest.mouseClick(p.map, Qt.LeftButton, Qt.NoModifier,
                         QPoint(g.left() + 3, int(g.top() + 4 * row_h + row_h / 2)))
        self.assertEqual(w.step, first + 4, "a click on a row moves the cursor there")

    def test_without_router_scores_the_used_experts_still_show(self):
        rec = make_tapped_recording(self.dir, probs=False)
        w = self.player(rec)
        lv = w.experts.map.levels()
        self.assertEqual(int(np.count_nonzero(lv)), 4 * 2)

    def test_a_step_with_unknown_shares_draws(self):
        """No router scores, and a step whose weights are NaN (review,
        2026-09-29): every step, mode and size still draws."""
        from athanor.tap.track import TapTrack
        r = Recorder(n_vocab=40, k=8, piece=lambda t: f" w{t}".encode())
        r.tap_track = TapTrack(["x"], facts={"n_expert": 8})
        rng = np.random.default_rng(1)
        for i in range(6):
            r.step(rng.standard_normal(40).astype(np.float32), i)
            w = np.array([np.nan, np.nan] if i == 3 else [0.6, 0.3], np.float32)
            r.tap_track.step(i, {f"ffn_moe_topk-{li}": np.array([1, 4], np.int32)
                                 for li in range(3)} | {f"ffn_moe_weights-{li}": w
                                                        for li in range(3)})
        w = self.player(read(r.save(self.dir)))
        for mode in ("cursor", "reply", "time"):
            w.experts.mode.setCurrentIndex(w.experts.mode.findData(mode))
            for i in range(6):
                w.set_step(i)
                self.app.processEvents()
            w.resize(900, 500)
            self.app.processEvents()
            self.assertFalse(w.experts.map.grab().toImage().isNull())

    def test_thinking_and_answer_are_offered_when_the_reply_thought(self):
        from athanor.gui.experts import ExpertPanel
        rec = make_tapped_recording(self.dir)
        p = ExpertPanel()
        p.load(rec.tap.experts(), {"thinking": np.arange(4), "answer": np.arange(4, 12)})
        self.assertGreaterEqual(p.mode.findData("thinking"), 0)
        p.mode.setCurrentIndex(p.mode.findData("thinking"))
        np.testing.assert_allclose(p.map.usage().sum(axis=1), 2.0, rtol=1e-5)

    def test_a_damaged_tap_file_costs_the_map_not_the_player(self):
        from athanor.waterfall import fileformat as ff
        rec = make_tapped_recording(self.dir)
        tp = ff.tap_path(rec.meta_path)
        tp.write_bytes(tp.read_bytes()[:-1])
        w = self.player(read(rec.meta_path))
        self.assertEqual(w.n_steps, rec.n_steps)
        self.assertFalse(w.experts.isVisible())
        self.assertIn("could not be read", w.title.text())


@needs_qt
class Player(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._t = TempDir()
        self.dir = self._t.__enter__()
        self.rec = make_recording(self.dir)

    def tearDown(self):
        self._t.__exit__(None, None, None)

    def test_it_loads_and_steps(self):
        from athanor.gui import WaterfallPlayer
        w = WaterfallPlayer(self.rec.meta_path)
        w.resize(900, 600)
        w.show()
        self.app.processEvents()
        self.assertEqual(w.n_steps, self.rec.n_steps)
        seen = []
        w.stepChanged.connect(seen.append)
        w.set_step(3)
        self.assertEqual((w.step, seen), (3, [3]))
        w.set_step(999)
        self.assertEqual(w.step, w.n_steps - 1, "clamped")
        w.set_step(-5)
        self.assertEqual(w.step, 0)
        self.assertEqual(w.slider.value(), 0)
        img = w.grab()
        self.assertGreater(img.width(), 100)

    def test_the_current_token_is_lit_in_the_text(self):
        from athanor.gui import WaterfallPlayer
        w = WaterfallPlayer(self.rec)
        w.set_step(4)                                  # the emoji: two UTF-16 units
        sels = w.text.extraSelections()
        cur = sels[-1].cursor
        a, b = self.rec.span(4)
        self.assertEqual(self.rec.text[a:b], " 📡")
        self.assertEqual(cur.selectedText(), " 📡")
        w.set_step(5)
        sel = w.text.extraSelections()[-1]
        self.assertEqual(sel.cursor.selectedText(), " is")

    def test_a_split_character(self):
        from athanor.gui import WaterfallPlayer
        w = WaterfallPlayer(self.rec)
        self.assertEqual(self.rec.span(6)[0], self.rec.span(6)[1], "half a character: no text")
        w.set_step(7)
        sel = w.text.extraSelections()[-1]
        self.assertEqual(sel.cursor.selectedText(), "你")

    def test_playback_follows_the_recorded_clock(self):
        from athanor.gui import WaterfallPlayer
        w = WaterfallPlayer(self.rec)
        self.assertEqual(w.delay_ms(0), 100)
        w.set_speed(2.0)
        self.assertEqual(w.delay_ms(0), 50)
        w.set_speed(-20.0)
        self.assertEqual(w.delay_ms(0), 50)
        w.set_speed(1.0)
        w.play()
        self.assertTrue(w.is_playing())
        for _ in range(w.n_steps + 2):
            w._tick() if w.is_playing() else None
        self.assertEqual(w.step, w.n_steps - 1)
        self.assertFalse(w.is_playing(), "stops at the end")
        w.play()
        self.assertEqual(w.step, 0, "play at the end starts again")
        w.pause()

    def test_the_waterfall_image_and_what_is_under_the_mouse(self):
        from athanor.gui import WaterfallPlayer
        w = WaterfallPlayer(self.rec)
        w.resize(900, 600)
        w.show()
        self.app.processEvents()
        v = w.view
        self.assertEqual((v._img.width(), v._img.height()), (8, self.rec.n_steps))
        wf, _tr = v._areas()
        h = v.hit(wf.left() + 1, wf.top() + v.row_px * 2 + 1)
        self.assertEqual(h["step"], 2)
        self.assertEqual(h["candidate"], 0)
        h = v.hit(wf.right() - 1, wf.top() + 1)
        self.assertEqual(h["candidate"], 7)
        self.assertIsNone(v.hit(wf.left() + 1, 1), "the header strip is not a step")
        taken = v._layout()["taken"]
        self.assertEqual(v.hit(taken.left() + 2, taken.top() + 1)["candidate"], "taken")
        w.axis.setCurrentIndex(1)                     # by token id
        self.assertEqual(v._img.width(), v.id_bins)
        self.assertIsNone(v.hit(1, 10_000))
        w.floor.setValue(-60.0)
        self.assertEqual(v.floor_db, -60.0)

    def test_the_candidates_at_the_cursor(self):
        from athanor.gui import WaterfallPlayer
        w = WaterfallPlayer(self.rec)
        w.set_step(4)
        rows = w.cands._rows
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(r["chosen"] for r in rows), 1)
        self.assertAlmostEqual(rows[0]["p"], math.exp(float(self.rec.steps[4]["logprobs"][0])))
        w.set_step(3)                        # not boosted: chosen far down, not in the list
        self.assertGreater(self.rec.chosen(3)["rank"], 0)

    def test_thinking_is_marked(self):
        ann = self.rec.meta["annotations"]
        self.assertEqual(ann[0]["athrec:label"], "thinking")
        self.assertEqual((ann[0]["athrec:step_start"], ann[0]["athrec:step_count"]), (0, 3))

    def test_an_empty_recording_and_none(self):
        from athanor.gui import WaterfallPlayer
        r = Recorder(n_vocab=4, k=2)
        empty = read(r.save(self.dir))
        w = WaterfallPlayer(empty)
        w.show()
        self.app.processEvents()
        w.play()
        self.assertFalse(w.is_playing())
        w.load(None)
        self.assertEqual(w.n_steps, 0)
        w.grab()

    def test_the_reading_zoom_writes_each_candidate_in_its_cell(self):
        """Bill, 2026-09-28: the potentials on the waterfall — readable."""
        from athanor.gui import WaterfallPlayer
        w = WaterfallPlayer(self.rec)
        w.resize(1300, 700)
        w.show()
        self.app.processEvents()
        before = (w.view.row_px, w.ranks.currentData())
        self.assertFalse(w.view.word_mode(), "the overview is a heat map")
        w.read_btn.click()
        self.app.processEvents()
        self.assertTrue(w.view.word_mode())
        self.assertLessEqual(w.ranks.currentData(), self.rec.k * 2)
        w.grab()                                        # paints the words
        w.read_btn.click()
        self.assertEqual((w.view.row_px, w.ranks.currentData()), before)

    def test_what_was_taken_is_on_every_row(self):
        """Bill, 2026-09-29: "have what is decided each time also present and
        highlighted" — even when it ranked past the columns on screen."""
        from athanor.gui import WaterfallPlayer
        from athanor.gui.waterfall import _ordinal
        w = WaterfallPlayer(self.rec)
        w.resize(1300, 700)
        w.show()
        w.ranks.setCurrentIndex(w.ranks.findData(8))
        w.set_reading(True)
        self.app.processEvents()
        lay = w.view._layout()
        self.assertTrue(lay["words"])
        self.assertGreater(lay["taken"].width(), 90, "wide enough for the word")
        self.assertLess(lay["taken"].right(), lay["grid"].left(), "left of the candidates")
        w.grab()
        self.assertEqual([_ordinal(i) for i in (1, 2, 3, 4, 11, 12, 13, 21, 22, 111)],
                         ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd",
                          "111th"])

    def test_doubts_are_underlined_in_the_reply(self):
        from PySide6.QtGui import QTextCharFormat, QTextCursor
        from athanor.gui import WaterfallPlayer
        w = WaterfallPlayer(self.rec)
        step = next(int(d) for d in self.rec.doubts() if self.rec.span(int(d))[1] >
                    self.rec.span(int(d))[0])
        a, _b = self.rec.span(step)
        c = QTextCursor(w.text.document())
        c.setPosition(w.text.qt_pos(a) + 1)
        self.assertEqual(c.charFormat().underlineStyle(), QTextCharFormat.WaveUnderline)
        w.mark.setChecked(False)
        c = QTextCursor(w.text.document())
        c.setPosition(w.text.qt_pos(a) + 1)
        self.assertNotEqual(c.charFormat().underlineStyle(), QTextCharFormat.WaveUnderline)

    def test_moments_of_doubt(self):
        from athanor.gui import WaterfallPlayer
        doubts = list(self.rec.doubts())
        off = [i for i in range(self.rec.n_steps) if self.rec.chosen(i)["rank"] > 0]
        self.assertTrue(off)
        self.assertTrue(set(off) <= set(doubts))
        self.assertEqual(doubts, sorted(doubts))
        w = WaterfallPlayer(self.rec)
        w.set_step(0)
        seen = []
        while (t := w.next_doubt(+1)) is not None:
            seen.append(t)
        self.assertEqual(seen, [d for d in doubts if d > 0])
        self.assertEqual(w.step, doubts[-1])
        self.assertEqual(w.next_doubt(-1), doubts[-2] if len(doubts) > 1 else None)
        self.assertEqual(self.rec.summary()["moments_of_doubt"], len(doubts))

    def test_what_is_to_come_is_dimmed_unless_asked_to_hide(self):
        """Bill, 2026-09-29: "why should hidden text stay hidden?" — it
        should not: the whole reply shows, the rest dimmed, by default."""
        from athanor.gui import WaterfallPlayer
        w = WaterfallPlayer(self.rec)
        w.set_step(3)
        self.assertFalse(w.reveal.isChecked())
        future = w.text.extraSelections()[0]
        self.assertAlmostEqual(future.format.foreground().color().alphaF(), 0.28, places=2)
        w.reveal.setChecked(True)
        future = w.text.extraSelections()[0]
        self.assertEqual(future.format.foreground().color().alphaF(), 0.0)

    def _wait_listed(self, lst, timeout=5.0):
        import time
        from PySide6.QtCore import QThreadPool
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            QThreadPool.globalInstance().waitForDone(50)
            self.app.processEvents()
            if not lst.heading.text().startswith("reading"):
                return
        self.fail("the folder was never listed")

    def test_the_panel_lists_and_opens(self):
        from athanor.gui import WaterfallPanel
        make_recording(self.dir)
        p = WaterfallPanel(self.dir)
        self.assertEqual(p.list.items.count(), 0, "nothing is read until it is shown")
        p.show()
        self._wait_listed(p.list)
        self.assertEqual(p.list.items.count(), 2)
        p.open(str(self.rec.meta_path))
        self.assertEqual(p.player.n_steps, self.rec.n_steps)
        self.assertEqual(p.list.items.currentItem().data(0x0100), str(self.rec.meta_path))

    def test_a_recording_made_after_the_list_is_selected_when_it_arrives(self):
        from athanor.gui import WaterfallPanel
        p = WaterfallPanel(self.dir)
        p.show()
        self._wait_listed(p.list)
        newer = make_recording(self.dir)
        p.open(str(newer.meta_path))
        self._wait_listed(p.list)
        self.assertEqual(p.list.items.count(), 2)
        self.assertEqual(p.list.items.currentItem().data(0x0100), str(newer.meta_path))

    def test_a_damaged_recording_says_so_and_is_not_shown_as_another(self):
        from athanor.gui import WaterfallPanel
        from athanor.waterfall import list_recordings
        other = make_recording(self.dir)
        with open(other.data_path, "r+b") as f:          # cut the data short
            f.truncate(100)
        rows = {r["path"]: r for r in list_recordings(self.dir)}
        self.assertIn("damaged", rows[str(other.meta_path)]["error"])
        self.assertIsNone(rows[str(self.rec.meta_path)]["error"])
        p = WaterfallPanel(self.dir)
        self.assertTrue(p.open(str(self.rec.meta_path)))
        self.assertFalse(p.open(str(other.meta_path)))
        self.assertEqual(p.player.n_steps, 0, "the previous recording is not left on screen")
        self.assertIn("could not be opened", p.player.title.text())


class Helpers(unittest.TestCase):

    @needs_qt
    def test_visible(self):
        from athanor.gui import visible
        self.assertEqual(visible(" the"), "␣the")
        self.assertEqual(visible("  "), "␣␣")
        self.assertEqual(visible("a\nb\t"), "a⏎b⇥")
        self.assertEqual(visible(""), "∅")
        self.assertEqual(visible("\x00x "), "␀x␣")

    @needs_qt
    def test_offsets_and_colours(self):
        from athanor.gui.waterfall import db_of, display_text, lut, utf16_offsets
        self.assertEqual(list(utf16_offsets("a📡b")), [0, 1, 3, 4])
        self.assertEqual(display_text("a\rb\n"), "a␍b\n")
        np.testing.assert_allclose(db_of([0.0, math.log(0.1), math.log(0.01)]), [0, -10, -20])
        t = lut()
        self.assertEqual(t.shape, (256, 3))
        self.assertEqual(tuple(t[0]), (8, 14, 40))
        self.assertEqual(tuple(t[-1]), (255, 250, 190))


if __name__ == "__main__":
    unittest.main()
