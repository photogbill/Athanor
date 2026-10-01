"""The Tap (spike S1): copying a model's tensors out while llama.cpp computes.

What is checked against llama.cpp itself, on models written on the spot
(``athanor.testing.tiny_model``, dense and mixture-of-experts):

* tapping changes nothing — logits bit-identical to an untapped context,
  idle and while copying;
* the Tap reads the truth — its copy of ``result_output`` IS the logits;
* the logit lens stands — the last layer's output, normed and multiplied
  by the output matrix here, reproduces llama.cpp's logits;
* the expert map is the router's — chosen experts are the top-scoring
  ones, their weights the router's values;
* a recording with the Tap is the same reply, one Tap record per token,
  and a Tap failure costs the Tap, never the reply.
"""

import contextlib
import io
import json
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from athanor import cli
from athanor.tap.core import patterns, split_name, token_axis, _glob
from athanor.tap.track import (TAP_FLAG_NONE, TAP_FLAG_PARTIAL, TAP_FLAG_STOPPED, TapTrack)
from athanor.waterfall import Recorder, RecordingError, read
from athanor.waterfall import fileformat as ff
from athanor.waterfall.reading import ExpertRouting
from fixtures import HAVE_LLAMA, TempDir, needs_llama


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------- no llama
class Names(unittest.TestCase):

    def test_split_name(self):
        self.assertEqual(split_name("ffn_moe_topk-12"), ("ffn_moe_topk", 12))
        self.assertEqual(split_name("result_output"), ("result_output", None))
        self.assertEqual(split_name("cache_k_l3 (view)"), ("cache_k_l3 (view)", None))

    def test_token_axis(self):
        self.assertEqual(token_axis("l_out-3"), 1)
        self.assertEqual(token_axis("ffn_moe_topk-3"), 1)
        self.assertEqual(token_axis("ffn_moe_probs-0"), 1)
        self.assertEqual(token_axis("ffn_moe_weights-3"), 2, "[1, n_used, tokens]")
        self.assertEqual(token_axis("ffn_moe_weighted-3"), 2)
        self.assertEqual(token_axis("result_output"), 1)

    def test_presets(self):
        self.assertEqual(patterns("experts"),
                         ("ffn_moe_topk-*", "ffn_moe_weights-*", "ffn_moe_probs-*"))
        self.assertEqual(patterns(["residual", "result_norm"]), ("l_out-*", "result_norm"))
        self.assertEqual(patterns("l_out-3"), ("l_out-3",))

    def test_a_glob_never_crosses_a_space(self):
        import re
        rx = re.compile(_glob("l_out-*"))
        self.assertTrue(rx.match("l_out-12"))
        self.assertFalse(rx.match("l_out-12 (view)"), "llama.cpp's derived names are not taken")
        self.assertFalse(re.compile(_glob("ffn_moe_weights-*")).match("ffn_moe_weights_norm-2"))


class Track(unittest.TestCase):

    def rows(self, i=0):
        return {"ffn_moe_topk-0": np.array([1, 2], np.int32),
                "ffn_moe_probs-0": np.full(4, 0.25, np.float32) + i,
                "l_out-1": np.arange(3, dtype=np.float32)}

    def test_layout_is_fixed_by_the_first_capture_and_early_steps_backfilled(self):
        t = TapTrack(["x"])
        t.step(0, None)
        t.step(1, self.rows(1))
        rec, block = t.finish(3)
        self.assertEqual(len(rec), 3)
        self.assertEqual(list(rec["flags"]), [TAP_FLAG_NONE, 0, TAP_FLAG_NONE])
        self.assertEqual([s["name"] for s in block["layout"]],
                         ["ffn_moe_probs-0", "ffn_moe_topk-0", "l_out-1"], "by layer, then kind")
        self.assertEqual(list(rec["ffn_moe_topk-0"][0]), [-1, -1])
        self.assertTrue(np.isnan(rec["ffn_moe_probs-0"][0]).all())
        self.assertEqual(block["record_bytes"], ff.layout_dtype(block["layout"]).itemsize)

    def test_a_missing_or_misshapen_stream_is_flagged_not_fatal(self):
        t = TapTrack(["x"])
        t.step(0, self.rows())
        r = self.rows()
        del r["l_out-1"]
        t.step(1, r)
        r = self.rows()
        r["ffn_moe_topk-0"] = np.array([1, 2, 3], np.int32)
        t.step(2, r)
        rec, block = t.finish(3)
        self.assertEqual(list(rec["flags"]), [0, TAP_FLAG_PARTIAL, TAP_FLAG_PARTIAL])
        self.assertTrue(any("3 values" in n for n in block["notes"]))

    def test_steps_stay_aligned_when_one_is_missed(self):
        t = TapTrack(["x"])
        t.step(0, self.rows(0))
        t.step(2, self.rows(2))          # step 1 never came
        rec, _ = t.finish(3)
        self.assertEqual(list(rec["flags"]), [0, TAP_FLAG_NONE, 0])
        self.assertAlmostEqual(float(rec["ffn_moe_probs-0"][2][0]), 2.25)

    def test_the_limit_stops_the_tap_and_says_where(self):
        t = TapTrack(["x"], limit_bytes=100)
        for i in range(5):
            t.step(i, self.rows())
        rec, block = t.finish(5)
        self.assertEqual(block["stopped_at"], 2)
        self.assertEqual(list(rec["flags"][2:]), [TAP_FLAG_STOPPED] * 3)
        self.assertTrue(any("limit" in n for n in block["notes"]))

    def test_a_failure_inside_the_track_is_kept_not_raised(self):
        t = TapTrack(["x"])
        t.step(0, self.rows())
        with mock.patch.object(TapTrack, "_append", side_effect=RuntimeError("boom")):
            t.step(1, self.rows())
        t.step(2, self.rows())
        rec, block = t.finish(3)
        self.assertIn("boom", block["error"])
        self.assertEqual(len(rec), 3)
        self.assertEqual(int(rec["flags"][2]), TAP_FLAG_STOPPED)


class File(unittest.TestCase):

    def recording(self, d, *, tap=True):
        r = Recorder(n_vocab=6, k=3, piece=lambda t: f"<{t}>".encode())
        if tap:
            r.tap_track = TapTrack(["experts"], facts={"arch": "llama", "n_expert": 4})
        for i in range(4):
            r.step(np.arange(6, dtype=np.float32) * (i + 1), 5)
            if tap:
                r.tap_track.step(i, {"ffn_moe_topk-0": np.array([i % 4, 3], np.int32),
                                     "ffn_moe_weights-0": np.array([0.6, 0.2], np.float32),
                                     "ffn_moe_probs-0": np.array([.1, .2, .1, .6], np.float32)})
        return r.save(d)

    def test_round_trip(self):
        with TempDir() as d:
            rec = read(self.recording(d), verify=True)
            self.assertTrue(ff.tap_path(rec.meta_path).is_file())
            self.assertEqual(len(rec.tap), rec.n_steps)
            self.assertEqual(rec.tap.kinds(), ["ffn_moe_probs", "ffn_moe_topk", "ffn_moe_weights"])
            ex = rec.tap.experts()
            self.assertEqual((ex.n_steps, ex.n_expert, ex.n_used), (4, 4, 2))
            np.testing.assert_allclose(ex.share()[0, 0], [0.75, 0.25], rtol=1e-6)
            self.assertEqual(rec.summary()["tap"]["recorded"], rec.tap.kinds())
            self.assertEqual(ff.stems(ff.tap_path(rec.meta_path))[0], rec.meta_path)

    def test_a_recording_without_the_tap_has_no_tap_file(self):
        with TempDir() as d:
            rec = read(self.recording(d, tap=False))
            self.assertIsNone(rec.tap)
            self.assertIsNone(rec.tap_info)
            self.assertFalse(ff.tap_path(rec.meta_path).exists())

    def test_a_damaged_tap_file_is_named_when_it_is_read(self):
        with TempDir() as d:
            p = self.recording(d)
            tp = ff.tap_path(p)
            tp.write_bytes(tp.read_bytes()[:-3])
            rec = read(p)                   # the Waterfall itself still opens
            with self.assertRaises(RecordingError) as cm:
                rec.tap
            self.assertIn("damaged", str(cm.exception))
            with self.assertRaises(RecordingError):
                read(p, verify=True)

    def test_verify_catches_a_changed_byte(self):
        with TempDir() as d:
            p = self.recording(d)
            tp = ff.tap_path(p)
            raw = bytearray(tp.read_bytes())
            raw[5] ^= 0xFF
            tp.write_bytes(bytes(raw))
            read(p).tap                     # the size is right: readable
            with self.assertRaises(RecordingError) as cm:
                read(p, verify=True)
            self.assertIn("SHA-256", str(cm.exception))

    def test_a_missing_tap_file_is_named(self):
        with TempDir() as d:
            p = self.recording(d)
            ff.tap_path(p).unlink()
            with self.assertRaises(RecordingError) as cm:
                read(p).tap
            self.assertIn("missing", str(cm.exception))

    def test_layout_types_are_checked(self):
        with self.assertRaises(RecordingError):
            ff.layout_dtype([{"name": "x", "dtype": "<f8", "count": 2}])

    def test_the_listing_says_what_the_tap_recorded(self):
        from athanor.waterfall import list_recordings
        with TempDir() as d:
            self.recording(d)
            rows = list_recordings(d)
            self.assertEqual(rows[0]["tap"], ["ffn_moe_probs", "ffn_moe_topk", "ffn_moe_weights"])


class Routing(unittest.TestCase):

    def test_share_normalises_probabilities_and_softmaxes_logits(self):
        ids = np.array([[[0, 1]]], np.int32)
        r = ExpertRouting([0], ids, np.array([[[0.3, 0.1]]], np.float32), n_expert=4)
        np.testing.assert_allclose(r.share()[0, 0], [0.75, 0.25], rtol=1e-6)
        r = ExpertRouting([0], ids, np.array([[[1.0, -1.0]]], np.float32), n_expert=4)
        e = math.exp(2)
        np.testing.assert_allclose(r.share()[0, 0], [e / (1 + e), 1 / (1 + e)], rtol=1e-6)
        r = ExpertRouting([0], ids, np.array([[[2.0, 1.0]]], np.float32), n_expert=4,
                          arch="gpt-oss")
        self.assertAlmostEqual(float(r.share()[0, 0, 0]), e ** 0.5 / (1 + e ** 0.5), places=5)

    def test_usage_and_grid(self):
        ids = np.array([[[0, 1], [2, 3]], [[0, 2], [2, -1]]], np.int32)   # 2 steps, 2 layers
        w = np.ones((2, 2, 2), np.float32)
        r = ExpertRouting([0, 1], ids, w, n_expert=4)
        u = r.usage()
        np.testing.assert_allclose(u[0], [1.0, 0.5, 0.5, 0.0])
        np.testing.assert_allclose(u[1], [0.0, 0.0, 1.0, 0.5])
        g = r.grid(1)["chosen"]
        self.assertGreater(g[0, 0], 0)
        self.assertEqual(float(g[1, 3]), 0.0, "a missing id (-1) lights nothing")
        np.testing.assert_allclose(r.usage([0])[1], [0, 0, 1, 1])


# ----------------------------------------------------------- with llama.cpp
@needs_llama
class Live(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from athanor.testing import tiny_model
        cls._tmp = TempDir()
        cls.dir = cls._tmp.__enter__()
        cls.dense = tiny_model(cls.dir / "dense.gguf", n_layers=3)
        cls.moe = tiny_model(cls.dir / "moe.gguf", n_layers=3, n_experts=8, n_experts_used=2)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.__exit__(None, None, None)

    def llm(self, path=None, **kw):
        import llama_cpp
        return llama_cpp.Llama(str(path or self.dense), n_ctx=256, verbose=False, seed=1,
                               n_threads=2, **kw)

    def tappable(self, path=None):
        from athanor import tap
        llm = self.llm(path)
        tap.make_tappable(llm)
        return llm

    def greedy(self, llm, t=None, n=8, prompt="the radio signal"):
        """(tokens, logits per step, the Tap's rows per step)."""
        import llama_cpp
        from athanor import tap
        steps = []
        orig = llm.sample

        def sample(*a, **k):
            tok = orig(*a, **k)
            ptr = llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1)
            steps.append((tok, np.ctypeslib.as_array(ptr, shape=(llm.n_vocab(),)).copy(),
                          t.output(-1) if t is not None else None))
            return tok

        llm.sample = sample
        llm.reset()
        try:
            ctx = tap.watching(llm, t) if t is not None else contextlib.nullcontext()
            with ctx:
                llm.create_completion(prompt, max_tokens=n, temperature=0)
        finally:
            del llm.sample
        return [s[0] for s in steps], [s[1] for s in steps], [s[2] for s in steps]

    # -- the binding ----------------------------------------------------------
    def test_ggml_is_bound_and_its_layout_checked(self):
        from athanor.tap import check_tap
        from athanor.tap.ggml import NAME_OFFSET, ggml
        self.assertIsNone(check_tap())
        g = ggml()
        self.assertTrue(g.where)
        self.assertEqual(NAME_OFFSET, 256)

    def test_capabilities_report_the_tap(self):
        from athanor.capabilities import probe
        self.assertTrue(probe()["features"]["eval_callback"]["available"])

    def test_make_tappable_is_idempotent_and_reversible(self):
        from athanor import tap
        llm = self.llm()
        self.assertFalse(tap.is_tappable(llm))
        d = tap.make_tappable(llm)
        self.assertIs(tap.make_tappable(llm), d)
        self.assertTrue(tap.is_tappable(llm))
        tap.make_plain(llm)
        self.assertFalse(tap.is_tappable(llm))
        with self.assertRaises(tap.TapUnavailable):
            with tap.watching(llm, tap.Tap()):
                pass
        llm.create_completion("x", max_tokens=2)          # still a working model

    def test_the_tap_cannot_be_opened_twice(self):
        from athanor import tap
        llm = self.tappable()
        with tap.watching(llm, tap.Tap()):
            with self.assertRaises(tap.TapUnavailable):
                with tap.watching(llm, tap.Tap()):
                    pass
        self.assertNotIn("decode", vars(llm._ctx), "the context is as it was")

    # -- exactness -----------------------------------------------------------
    def test_tapping_changes_nothing(self):
        from athanor import tap
        base_toks, base_logits, _ = self.greedy(self.llm())
        llm = self.tappable()
        for t in (None, tap.Tap("residual"), tap.Tap(("logits", "result_norm", "l_out-0"))):
            toks, logits, _ = self.greedy(llm, t)
            self.assertEqual(toks, base_toks)
            self.assertEqual(max(float(np.abs(a - b).max()) for a, b in zip(base_logits, logits)),
                             0.0, f"bit-identical ({t.streams if t else 'idle'})")

    def test_the_tap_reads_the_logits_themselves(self):
        from athanor import tap
        _toks, logits, rows = self.greedy(self.tappable(), tap.Tap("logits"))
        for lg, r in zip(logits, rows):
            np.testing.assert_array_equal(r["result_output"], lg)

    def test_the_logit_lens_reproduces_the_logits(self):
        from athanor import gguf, tap
        _t, logits, rows = self.greedy(self.tappable(), tap.Tap(("l_out-2", "result_norm")))
        g = gguf.read(self.dense)
        E = int(g.get("llama.embedding_length"))
        W = np.frombuffer(g.tensor_bytes("token_embd.weight"), np.float32).reshape(-1, E)
        for lg, r in zip(logits, rows):
            x = r["l_out-2"].astype(np.float64)
            normed = x / math.sqrt(float((x * x).mean()) + 1e-5)     # output_norm is ones
            np.testing.assert_allclose(normed, r["result_norm"], atol=1e-5)
            np.testing.assert_allclose(normed @ W.T.astype(np.float64), lg, atol=1e-4)

    def test_rows_all_holds_the_prompt(self):
        from athanor import tap
        llm = self.tappable()
        n = len(llm.tokenize(b"the radio signal"))
        with tap.capture(llm, ("l_out-0", "l_out-2"), rows="all") as t:
            llm.reset()
            llm.create_completion("the radio signal", max_tokens=3, temperature=0)
        first = t.decodes[0]
        self.assertEqual(first["n_tokens"], n)
        self.assertEqual(first["tensors"]["l_out-0"].shape, (n, 64))
        self.assertEqual(first["tensors"]["l_out-2"].shape, (1, 64),
                         "llama.cpp keeps only the output rows at the last layer")

    def test_a_forward_pass_nobody_announced_gives_its_last_row(self):
        """A vision handler decodes by itself (mtmd), past the wrapper: the
        Tap cannot know the batch, so it keeps each tensor's last row."""
        import llama_cpp as lc
        from athanor import tap
        llm = self.tappable()
        ids = llm.tokenize(b"the radio signal")
        with tap.capture(llm, "l_out-0", rows="all") as ref:
            llm.reset()
            llm.eval(ids)
        want = ref.decodes[0]["tensors"]["l_out-0"][-1]
        llm.reset()
        llm._ctx.kv_cache_clear()
        t = tap.Tap("l_out-0")
        b = lc.llama_batch_init(len(ids), 0, 1)
        try:
            b.n_tokens = len(ids)
            for i, x in enumerate(ids):
                b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = x, i, 1, i == len(ids) - 1
                b.seq_id[i][0] = 0
            with tap.watching(llm, t):
                self.assertEqual(lc.llama_decode(llm._ctx.ctx, b), 0)
                got = t.output(-1)["l_out-0"]
        finally:
            lc.llama_batch_free(b)
        np.testing.assert_array_equal(got, want)

    def test_micro_batches_are_one_forward_pass(self):
        """A batch longer than n_ubatch runs its graph once per micro-batch
        (review, 2026-09-29): every layer is kept, rows in batch order."""
        import llama_cpp
        from athanor import tap
        from athanor.waterfall import attach
        prompt = b"the radio signal and the water fall of the model token"

        def load(**kw):
            m = llama_cpp.Llama(str(self.moe), n_ctx=256, verbose=False, seed=1, n_threads=2,
                                **kw)
            tap.make_tappable(m)
            return m

        small = load(n_batch=64, n_ubatch=16)
        ids = small.tokenize(prompt)
        self.assertTrue(32 < len(ids) <= 64, "one decode, several micro-batches")
        with tap.capture(small, ("l_out-0", "ffn_moe_topk-0", "l_out-2"), rows="all") as t:
            small.reset()
            small.eval(ids)
        first = t.decodes[0]["tensors"]
        self.assertEqual(first["l_out-0"].shape, (len(ids), 64))
        self.assertEqual(first["ffn_moe_topk-0"].shape, (len(ids), 2))
        self.assertEqual(first["l_out-2"].shape, (1, 64), "outputs only, from the last micro-batch")
        self.assertEqual(t.report()["notes"], [])
        big = load()                                          # one graph for the whole prompt
        with tap.capture(big, ("l_out-0",), rows="all") as ref:
            big.reset()
            big.eval(ids)
        np.testing.assert_allclose(first["l_out-0"], ref.decodes[0]["tensors"]["l_out-0"],
                                   atol=1e-4)
        small.reset()                  # else the cached prompt is not computed again
        with attach(small, tap="experts") as rec:
            small.create_completion(prompt.decode(), max_tokens=4, temperature=0)
        r = read(rec.save(self.dir / "ubatch"))
        self.assertEqual(r.tap.experts().layers, [0, 1, 2])
        self.assertEqual(int(np.count_nonzero(r.tap.flags)), 0)

    def test_the_token_axis_comes_from_the_shape(self):
        """Qcur is [head_dim, heads, tokens] after its reshape: a prompt as
        long as the model has heads must not be read along the heads."""
        from athanor import tap
        llm = self.tappable()
        n_head = int(llm.metadata["llama.attention.head_count"])
        ids = llm.tokenize(b"the radio signal the radio signal the radio")[:n_head]
        self.assertEqual(len(ids), n_head)
        with tap.capture(llm, ("Qcur-0", "attn_norm-0"), rows="all") as t:
            llm.reset()
            llm.eval(ids)
        d = t.decodes[0]["tensors"]
        self.assertEqual(d["Qcur-0"].shape[0], n_head)
        self.assertEqual(d["attn_norm-0"].shape, (n_head, 64))
        self.assertTrue(any("more than one tensor" in n for n in t.report()["notes"]),
                        "a name llama.cpp gives twice is said, not merged")

    def test_a_context_that_cannot_be_rebuilt_raises_instead_of_crashing(self):
        """Review, 2026-09-29: if neither the tapped nor a plain context can
        be made, the model must raise in Python, never hand llama.cpp a
        freed context. In a subprocess, so a crash cannot take the suite."""
        import subprocess
        import sys
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import llama_cpp\n"
            "from unittest import mock\n"
            "from llama_cpp import _internals\n"
            "from athanor import tap\n"
            "llm = llama_cpp.Llama(%r, n_ctx=128, verbose=False)\n"
            "with mock.patch.object(_internals, 'LlamaContext', side_effect=ValueError('no room')):\n"
            "    try:\n"
            "        tap.make_tappable(llm)\n"
            "    except tap.TapUnavailable as e:\n"
            "        print('refused:', 'reload' in str(e))\n"
            "try:\n"
            "    llm.create_completion('x', max_tokens=2)\n"
            "except tap.TapUnavailable as e:\n"
            "    print('raised:', 'reload' in str(e))\n"
        ) % (str(Path(__file__).resolve().parents[1]), str(self.dense))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("refused: True", r.stdout)
        self.assertIn("raised: True", r.stdout)

    def test_a_failure_inside_the_tap_never_reaches_llama_cpp(self):
        from athanor import tap
        base_toks, base_logits, _ = self.greedy(self.llm())
        llm = self.tappable()
        t = tap.Tap("residual")
        with mock.patch.object(tap.Tap, "take", side_effect=RuntimeError("boom")):
            toks, logits, _ = self.greedy(llm, t)
        self.assertEqual(toks, base_toks)
        self.assertEqual(max(float(np.abs(a - b).max()) for a, b in zip(base_logits, logits)), 0)
        self.assertIn("boom", t.error)

    def test_tensor_names(self):
        from athanor import tap
        llm = self.tappable(self.moe)
        names = tap.tensor_names(llm)
        for n in ("l_out-0", "l_out-2", "ffn_moe_topk-1", "ffn_moe_probs-2", "result_output"):
            self.assertIn(n, names)
        self.assertEqual(llm.n_tokens, 0, "the model's cache is left empty")

    # -- experts ------------------------------------------------------------------
    def test_the_expert_map_is_the_routers(self):
        from athanor import tap
        llm = self.tappable(self.moe)
        with tap.capture(llm, ("experts", "ffn_moe_weights_norm-*"), rows="all") as t:
            llm.reset()
            llm.create_completion("the radio signal", max_tokens=6, temperature=0)
        checked = 0
        for d in t.decodes:
            for layer in range(3):
                ts = d["tensors"]
                probs, top = ts[f"ffn_moe_probs-{layer}"], ts[f"ffn_moe_topk-{layer}"]
                w, wn = ts[f"ffn_moe_weights-{layer}"], ts[f"ffn_moe_weights_norm-{layer}"]
                self.assertEqual(top.shape[1], 2)
                for i in range(len(top)):
                    self.assertEqual(set(top[i].tolist()),
                                     set(np.argsort(-probs[i], kind="stable")[:2].tolist()))
                    np.testing.assert_allclose(w[i], probs[i][top[i]], rtol=1e-6)
                    np.testing.assert_allclose(wn[i], w[i] / w[i].sum(), rtol=1e-5)
                    checked += 1
        self.assertGreater(checked, 20)

    # -- with the Waterfall ------------------------------------------------------
    MSGS = [{"role": "user", "content": "Hello there, radio operator"}]

    def chat(self, llm, **kw):
        kw.setdefault("max_tokens", 16)
        kw.setdefault("temperature", 0.9)
        kw.setdefault("seed", 11)
        return llm.create_chat_completion(self.MSGS, **kw)["choices"][0]

    def test_a_tapped_recording_is_the_same_reply_with_a_record_per_token(self):
        from athanor.waterfall import attach
        plain = self.chat(self.llm(self.moe))
        llm = self.tappable(self.moe)
        with attach(llm, tap="experts") as rec:
            out = self.chat(llm)
        self.assertEqual(out["message"]["content"], plain["message"]["content"])
        r = read(rec.save(self.dir / "tapped"), verify=True)
        self.assertEqual(len(r.tap), r.n_steps)
        self.assertEqual(int(np.count_nonzero(r.tap.flags)), 0)
        ex = r.tap.experts()
        self.assertEqual(ex.ids.shape, (r.n_steps, 3, 2))
        self.assertEqual(ex.n_expert, 8)
        self.assertTrue(((ex.ids >= 0) & (ex.ids < 8)).all())
        np.testing.assert_allclose(ex.share().sum(axis=2), 1.0, rtol=1e-5)
        info = r.tap_info
        self.assertEqual(info["model"]["n_expert"], 8)
        self.assertIsNone(info["error"])
        self.assertGreater(info["forward_passes"], 0)

    def test_each_record_is_the_pass_that_produced_its_token(self):
        """Alignment: the Tap's result_output for a step IS that step's
        distribution, as the Waterfall recorded it."""
        from athanor.waterfall import attach
        llm = self.tappable()
        with attach(llm, tap="logits", k=8) as rec:
            self.chat(llm, max_tokens=10)
        r = read(rec.save(self.dir / "aligned"))
        logits = r.tap.stream("result_output")
        for i in range(r.n_steps):
            lg = logits[i].astype(np.float64)
            lse = lg.max() + math.log(np.exp(lg - lg.max()).sum())
            ids = r.steps["ids"][i]
            np.testing.assert_allclose(lg[ids] - lse, r.steps["logprobs"][i], atol=1e-4)
            self.assertEqual(int(np.argmax(lg)), int(ids[0]))

    def test_an_untapped_model_is_recorded_without_and_says_why(self):
        from athanor.waterfall import attach
        llm = self.llm(self.moe)
        with attach(llm, tap="experts") as rec:
            self.chat(llm, max_tokens=4)
        r = read(rec.save(self.dir / "untapped"))
        self.assertIsNone(r.tap)
        self.assertIn("make_tappable", r.tap_info["unavailable"])
        self.assertGreaterEqual(r.n_steps, 4, "the reply itself is recorded")

    def test_a_reply_in_two_parts_keeps_one_aligned_tap(self):
        from athanor.waterfall import attach
        llm = self.tappable(self.moe)
        with attach(llm, tap="experts") as rec:
            self.chat(llm, max_tokens=5)
        first = rec.n_steps
        with attach(llm, tap="experts", recorder=rec):
            llm.create_completion("and then", max_tokens=4, temperature=0)
        r = read(rec.save(self.dir / "parts"))
        self.assertGreater(r.n_steps, first)
        self.assertEqual(len(r.tap), r.n_steps)
        self.assertEqual(int(np.count_nonzero(r.tap.flags)), 0)
        self.assertEqual(len(rec.tap_track.sessions), 2, "one Tap session per part")

    def test_a_tap_failure_costs_the_tap_not_the_recording(self):
        from athanor import tap
        from athanor.waterfall import attach
        plain = self.chat(self.llm(self.moe))
        llm = self.tappable(self.moe)
        with mock.patch.object(tap.Tap, "output", side_effect=RuntimeError("tap broke")):
            with attach(llm, tap="experts") as rec:
                out = self.chat(llm)
        self.assertEqual(out["message"]["content"], plain["message"]["content"])
        r = read(rec.save(self.dir / "tapfail"))
        self.assertIsNone(r.error)
        self.assertGreaterEqual(r.n_steps, 16)
        self.assertIn("tap broke", r.tap_info["error"])

    def test_an_error_inside_the_tap_marks_the_steps_stopped(self):
        from athanor import tap
        from athanor.tap.track import TAP_FLAG_STOPPED
        from athanor.waterfall import attach
        llm = self.tappable(self.moe)
        calls = {"n": 0}
        real = tap.Tap.take

        def flaky(self_, t):
            calls["n"] += 1
            if calls["n"] > 20:
                raise RuntimeError("copy failed")
            return real(self_, t)

        with mock.patch.object(tap.Tap, "take", flaky):
            with attach(llm, tap="experts") as rec:
                self.chat(llm, max_tokens=8)
        r = read(rec.save(self.dir / "stopped"))
        self.assertIn("copy failed", r.tap_info["error"])
        self.assertIsNotNone(r.tap_info["stopped_at"])
        self.assertEqual(int(r.tap.flags[-1]), TAP_FLAG_STOPPED)

    # -- the probe and the command line --------------------------------------------
    def test_the_probe(self):
        from athanor.tap.probe import probe
        r = probe(self.llm(self.moe), tokens=6)
        for k, v in r["exactness"].items():
            if k.endswith("_max_abs"):
                self.assertEqual(v, 0.0, k)
        self.assertTrue(r["exactness"]["idle_tokens_identical"])
        self.assertEqual(r["result_output_vs_logits_max_abs"], 0.0)
        self.assertTrue(r["lens"]["checked"], r["lens"])
        self.assertTrue(r["lens"]["reproduces"])
        self.assertEqual(r["experts"]["chosen_not_router_top"], 0)
        self.assertGreater(r["experts"]["layer_steps_checked"], 0)
        self.assertEqual(set(r["ms_per_token"]),
                         {"plain", "tapped_idle", "logits", "residual", "experts"})
        self.assertIn("changed nothing", r["verdict"])

    def test_the_command_line(self):
        with TempDir() as d:
            code, out, err = run("tap", "probe", str(self.moe), "--tokens", "4", "--gpu-layers",
                                 "0", "--json", "--no-record", "--data", str(d))
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(out)["kind"], "tap-probe")
            code, out, err = run("tap", "names", str(self.moe), "--gpu-layers", "0",
                                 "--data", str(d))
            self.assertEqual(code, 0, err)
            self.assertIn("ffn_moe_topk", out)
            code, out, err = run("record", str(self.moe), "--prompt", "hello", "--max-tokens",
                                 "4", "--gpu-layers", "0", "--tap", "experts", "--out",
                                 str(d / "rec"), "--no-record", "--data", str(d))
            self.assertEqual(code, 0, err)
            self.assertIn("the Tap recorded", out)
            r = read(next((d / "rec").glob("*.athrec-meta")))
            self.assertEqual(len(r.tap), r.n_steps)


if __name__ == "__main__":
    unittest.main()
