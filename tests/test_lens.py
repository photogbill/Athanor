"""The lens (M1/M30): every recorded layer read as logits, and decision depth.

Checked without llama.cpp, on a model written on the spot and a recording
whose residual stream is made by hand so the answers are known:

* the K-quant decoders match a plain transcription of ggml's C loops;
* the unembedding is the file's own norm and matrix, cached and reloaded,
  and a damaged cache is rebuilt rather than trusted;
* the pass reproduces the recording's own logits at the top (so the label is
  MEASURED), reads depth and first-seen as constructed, flags the steps the
  Tap missed, refuses the wrong model, and never touches the recording;
* the track reads back through ``Recording.lens`` and the command line.

With llama.cpp installed, the same pass runs on a real tapped recording of
the tiny model and must reproduce llama.cpp's logits at the last layer.
"""

import contextlib
import io
import json
import struct
import unittest

import numpy as np

from athanor import cli, lens
from athanor.gguf import GGMLType, GGUFWriter, ValueType
from athanor.gguf import dequant as D
from athanor.gguf import read as read_gguf
from athanor.lens import fileformat as lf
from athanor.lens.run import _run_back, parse_layers
from athanor.lens.unembed import LensUnavailable, build, fingerprint, load
from athanor.tap.track import TapTrack
from athanor.testing import tiny_model
from athanor.waterfall import Recorder, RecordingError, list_recordings, read
from fixtures import TempDir, needs_llama, tiny_gguf


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


# ------------------------------------------------------------- dequant
def _f16(x: float) -> bytes:
    return np.float16(x).tobytes()


def _ref_q6_k(block: bytes) -> list:
    """ggml's dequantize_row_q6_K, one block, written out loop for loop."""
    ql, qh = block[0:128], block[128:192]
    sc = struct.unpack("16b", block[192:208])
    d = float(np.frombuffer(block[208:210], np.float16)[0])
    y = [0.0] * 256
    for half in range(2):
        yo, qlo, qho, so = 128 * half, 64 * half, 32 * half, 8 * half
        for l in range(32):
            is_ = l // 16
            q1 = ((ql[qlo + l] & 0xF) | (((qh[qho + l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32
            y[yo + l] = d * sc[so + is_] * q1
            y[yo + l + 32] = d * sc[so + is_ + 2] * q2
            y[yo + l + 64] = d * sc[so + is_ + 4] * q3
            y[yo + l + 96] = d * sc[so + is_ + 6] * q4
    return y


def _scale_min(q: bytes, j: int) -> tuple:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def _ref_q4_k(block: bytes) -> list:
    d = float(np.frombuffer(block[0:2], np.float16)[0])
    dmin = float(np.frombuffer(block[2:4], np.float16)[0])
    scales, qs = block[4:16], block[16:144]
    y = [0.0] * 256
    is_, q = 0, 0
    for j in range(0, 256, 64):
        sc1, m1 = _scale_min(scales, is_)
        sc2, m2 = _scale_min(scales, is_ + 1)
        for l in range(32):
            y[j + l] = d * sc1 * (qs[q + l] & 0xF) - dmin * m1
            y[j + 32 + l] = d * sc2 * (qs[q + l] >> 4) - dmin * m2
        q += 32
        is_ += 2
    return y


def _ref_q5_k(block: bytes) -> list:
    d = float(np.frombuffer(block[0:2], np.float16)[0])
    dmin = float(np.frombuffer(block[2:4], np.float16)[0])
    scales, qh, ql = block[4:16], block[16:48], block[48:176]
    y = [0.0] * 256
    is_, q, u1, u2 = 0, 0, 1, 2
    for j in range(0, 256, 64):
        sc1, m1 = _scale_min(scales, is_)
        sc2, m2 = _scale_min(scales, is_ + 1)
        for l in range(32):
            y[j + l] = d * sc1 * ((ql[q + l] & 0xF) + (16 if qh[l] & u1 else 0)) - dmin * m1
            y[j + 32 + l] = d * sc2 * ((ql[q + l] >> 4) + (16 if qh[l] & u2 else 0)) - dmin * m2
        q += 32
        is_ += 2
        u1 <<= 2
        u2 <<= 2
    return y


def _ref_q5_0(block: bytes) -> list:
    d = float(np.frombuffer(block[0:2], np.float16)[0])
    qh = struct.unpack("<I", block[2:6])[0]
    qs = block[6:22]
    y = [0.0] * 32
    for j in range(16):
        xh_0 = ((qh >> j) << 4) & 0x10
        xh_1 = (qh >> (j + 12)) & 0x10
        y[j] = ((qs[j] & 0x0F) | xh_0) - 16
        y[j + 16] = ((qs[j] >> 4) | xh_1) - 16
    return [v * d for v in y]


class Dequant(unittest.TestCase):
    """The numpy decoders against a transcription of ggml's loops."""

    def blocks(self, nbytes, n, f16_at, seed=5):
        rng = np.random.default_rng(seed)
        raw = rng.integers(0, 256, size=(n, nbytes), dtype=np.uint8)
        for off in f16_at:
            vals = (rng.standard_normal(n) * 0.02).astype(np.float16)
            raw[:, off:off + 2] = vals.view(np.uint8).reshape(n, 2)
        return raw.tobytes()

    def check(self, gtype, nbytes, per, ref, f16_at):
        raw = self.blocks(nbytes, 9, f16_at)
        mine = D.dequantize_row(raw, gtype, 9 * per)
        want = np.concatenate([ref(raw[i * nbytes:(i + 1) * nbytes]) for i in range(9)])
        np.testing.assert_allclose(mine, want.astype(np.float32), rtol=1e-6, atol=1e-7)

    def test_q6_k(self):
        self.check(GGMLType.Q6_K, 210, 256, _ref_q6_k, (208,))

    def test_q4_k(self):
        self.check(GGMLType.Q4_K, 144, 256, _ref_q4_k, (0, 2))

    def test_q5_k(self):
        self.check(GGMLType.Q5_K, 176, 256, _ref_q5_k, (0, 2))

    def test_q5_0(self):
        self.check(GGMLType.Q5_0, 22, 32, _ref_q5_0, (0,))

    def test_native_types_are_declared(self):
        for t in (GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_0, GGMLType.F16):
            self.assertIsNone(D.can_dequantize(t))

    def test_a_row_range_equals_the_rows_one_by_one(self):
        rng = np.random.default_rng(2)
        emb = rng.standard_normal((40, 64)).astype(np.float32)
        with TempDir() as d:
            for etype in (GGMLType.F32, GGMLType.Q8_0):
                p = tiny_gguf(d / f"e{int(etype)}.gguf", n_vocab=40, n_embd=64, emb=emb,
                              emb_type=etype)
                g = read_gguf(p)
                t = g.tensor("token_embd.weight")
                a = D.dequantize_row_range(g, t, 5, 23)
                b = D.dequantize_rows(g, t, range(5, 23))
                np.testing.assert_array_equal(a, b)
                self.assertEqual(a.shape, (18, 64))
                self.assertEqual(D.dequantize_row_range(g, t, 7, 7).shape, (0, 64))
                with self.assertRaises(IndexError):
                    D.dequantize_row_range(g, t, 30, 41)
            np.testing.assert_allclose(D.dequantize_row_range(read_gguf(d / "e0.gguf"),
                                                              "token_embd.weight", 0, 40), emb)


# ---------------------------------------------------------- unembedding
class TheUnembedding(unittest.TestCase):

    def setUp(self):
        self._tmp = TempDir()
        self.dir = self._tmp.__enter__()
        self.model = tiny_model(self.dir / "tiny.gguf", n_layers=2)
        self.cache = self.dir / "lenses"

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def test_it_is_the_files_own_norm_and_matrix(self):
        u = build(self.model, cache_dir=self.cache)
        g = read_gguf(self.model)
        E = int(g.get("llama.embedding_length"))
        W = np.frombuffer(g.tensor_bytes("token_embd.weight"), np.float32).reshape(-1, E)
        self.assertEqual((u.n_vocab, u.n_embd), W.shape)
        self.assertTrue(u.info["output"]["tied"], "no output.weight: the embeddings are the output")
        self.assertEqual(u.kind, "rms")
        self.assertAlmostEqual(u.eps, 1e-5, places=9)
        np.testing.assert_array_equal(np.asarray(u.W), W)
        x = np.random.default_rng(1).standard_normal((5, E)).astype(np.float32)
        n = x / np.sqrt((x.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-5)
        np.testing.assert_allclose(u.norm(x), n, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(u.logits(x), n @ W.T, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(u.logits(x[0]), (n @ W.T)[0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(u.logits_for(x, [3, 7, 11]), (n @ W.T)[:, [3, 7, 11]],
                                   rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(u.logits(x, rows_per_chunk=7), n @ W.T, rtol=1e-4, atol=1e-4)

    def test_the_cache_is_written_under_the_fingerprint_and_reused(self):
        u = build(self.model, cache_dir=self.cache)
        g = read_gguf(self.model)
        fp = fingerprint(g.header_sha256(), self.model.stat().st_size)
        self.assertEqual(u.fingerprint, fp)
        folder = self.cache / fp
        self.assertTrue((folder / "unembed.f32").is_file())
        self.assertTrue((folder / "unembed.json").is_file())
        info = json.loads((folder / "unembed.json").read_text(encoding="utf-8"))
        self.assertEqual(info["output"]["tensor"], "token_embd.weight")
        self.assertEqual(info["n_vocab"], u.n_vocab)
        self.assertNotIn("label", info, "a decoded copy of the file is not a measurement")
        before = (folder / "unembed.f32").stat().st_mtime_ns
        again = build(self.model, cache_dir=self.cache)
        self.assertEqual((folder / "unembed.f32").stat().st_mtime_ns, before, "loaded, not redone")
        np.testing.assert_array_equal(np.asarray(again.W), np.asarray(u.W))
        self.assertIsNone(load(self.cache / "nothing-here"))

    def test_a_damaged_cache_is_rebuilt_not_trusted(self):
        u = build(self.model, cache_dir=self.cache)
        folder = self.cache / u.fingerprint
        with open(folder / "unembed.f32", "r+b") as f:
            f.truncate(100)
        self.assertIsNone(load(folder, expect=u.fingerprint))
        u2 = build(self.model, cache_dir=self.cache)
        self.assertEqual((folder / "unembed.f32").stat().st_size, u.n_vocab * u.n_embd * 4)
        self.assertEqual(u2.n_vocab, u.n_vocab)

    def test_a_layernorm_model_with_biases_and_a_softcap(self):
        """GPT-2-style final norm (mean-subtracting, with a bias), an output
        bias, and Gemma's logit soft-cap, all taken from the file."""
        E, V = 8, 20
        rng = np.random.default_rng(4)
        W = rng.standard_normal((V, E)).astype(np.float32)
        nw = rng.standard_normal(E).astype(np.float32)
        nb = rng.standard_normal(E).astype(np.float32)
        ob = rng.standard_normal(V).astype(np.float32)
        w = GGUFWriter()
        w.add("general.architecture", "gpt2")
        w.add("gpt2.embedding_length", E, ValueType.UINT32)
        w.add("gpt2.block_count", 1, ValueType.UINT32)
        w.add("gpt2.attention.layer_norm_epsilon", 1e-5, ValueType.FLOAT32)
        w.add("gpt2.final_logit_softcapping", 30.0, ValueType.FLOAT32)
        w.add("tokenizer.ggml.model", "gpt2")
        w.add("tokenizer.ggml.tokens", [f"t{i}" for i in range(V)], ValueType.ARRAY,
              ValueType.STRING)
        w.add_tensor("token_embd.weight", (E, V), GGMLType.F32, rng.standard_normal((V, E))
                     .astype(np.float32))
        w.add_tensor("output.weight", (E, V), GGMLType.F32, W)
        w.add_tensor("output.bias", (V,), GGMLType.F32, ob)
        w.add_tensor("output_norm.weight", (E,), GGMLType.F32, nw)
        w.add_tensor("output_norm.bias", (E,), GGMLType.F32, nb)
        p = w.write_file(self.dir / "ln.gguf")
        u = build(p, cache_dir=self.cache)
        self.assertEqual(u.kind, "layer")
        self.assertFalse(u.info["output"]["tied"])
        self.assertEqual(u.softcap, 30.0)
        x = rng.standard_normal((3, E)).astype(np.float32)
        mu = x.mean(axis=1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
        n = (x - mu) / np.sqrt(var + 1e-5) * nw + nb
        want = np.tanh((n @ W.T + ob) / 30.0) * 30.0
        np.testing.assert_allclose(u.logits(x), want, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(u.logits_for(x, [1, 2]), want[:, [1, 2]], rtol=1e-4, atol=1e-4)

    def test_a_file_without_a_final_norm_is_refused_with_a_reason(self):
        w = GGUFWriter()
        w.add("general.architecture", "llama")
        w.add("tokenizer.ggml.model", "llama")
        w.add("tokenizer.ggml.tokens", ["a", "b"], ValueType.ARRAY, ValueType.STRING)
        w.add_tensor("token_embd.weight", (4, 2), GGMLType.F32, np.zeros((2, 4), np.float32))
        p = w.write_file(self.dir / "nonorm.gguf")
        with self.assertRaises(LensUnavailable) as cm:
            build(p, cache_dir=self.cache)
        self.assertIn("output_norm.weight", str(cm.exception))


# ------------------------------------------------------------- the pass
def make_recording(folder, model, u, *, n=12, k=16, miss=None, chosen_off=3, seed=0,
                   early=4, streams=("l_out-0", "l_out-1")):
    """A recording of ``n`` steps whose residual stream is made by hand from
    the tiny model's own unembedding: steps below ``early`` have layer 0's
    output equal to the top layer's (decided at 0); every ``chosen_off``-th
    step the sampler takes the second-best token; step ``miss`` has no Tap
    rows. Returns (meta path, H)."""
    rng = np.random.default_rng(seed)
    L, E = 2, u.n_embd
    H = rng.standard_normal((n, L, E)).astype(np.float32) * 3
    H[:early, 0] = H[:early, 1]
    final = u.logits(H[:, -1])
    rec = Recorder(n_vocab=u.n_vocab, k=k, piece=lambda t: f"tok{t}".encode())
    rec.meta["model"] = {"path": str(model)}
    track = TapTrack(("l_out-*",), facts={"arch": "llama", "n_layer": 2, "n_embd": E,
                                          "n_vocab": u.n_vocab})
    rec.tap_track = track
    for i in range(n):
        fav = int(final[i].argmax())
        chosen = fav if (i % chosen_off) else int(np.argsort(-final[i])[1])
        rec.step(final[i], chosen, t=float(i))
        if i != miss:
            track.step(i, {s: H[i, int(s.split("-")[1])] for s in streams})
    return rec.save(folder, finish_reason="stop"), H


class ThePass(unittest.TestCase):

    def setUp(self):
        self._tmp = TempDir()
        self.dir = self._tmp.__enter__()
        self.model = tiny_model(self.dir / "tiny.gguf", n_layers=2)
        self.cache = self.dir / "lenses"
        self.u = build(self.model, cache_dir=self.cache)

    def tearDown(self):
        self._tmp.__exit__(None, None, None)

    def test_depth_and_the_check_as_constructed(self):
        path, H = make_recording(self.dir / "rec", self.model, self.u, miss=7)
        before = {p.name: p.read_bytes() for p in (self.dir / "rec").iterdir()}
        r = lens.run(path, cache_dir=self.cache)
        after = {p.name: p.read_bytes() for p in (self.dir / "rec").iterdir()}
        for name, raw in before.items():
            self.assertEqual(after[name], raw, f"{name} was changed by the lens")
        self.assertEqual(r["label"], "MEASURED")
        c = r["check"]
        self.assertTrue(c["reproduces"])
        self.assertEqual(c["favourite_agrees"], 11)       # step 7 has no residual
        self.assertEqual(c["steps_compared"], 11)
        self.assertLess(c["max_abs_logprob_error"], 1e-4)
        rec = read(path)
        ld = rec.lens
        self.assertIsNotNone(ld)
        self.assertEqual(ld.layers, [0, 1])
        self.assertEqual(ld.k, 8)
        self.assertEqual(ld.n_steps, 12)
        depth = ld.depth
        np.testing.assert_array_equal(depth[:4], 0)
        np.testing.assert_array_equal(depth[4:7], 1)
        self.assertEqual(depth[7], -1)
        self.assertEqual(int(ld.flags[7]), lf.FLAG_NO_RESIDUAL)
        self.assertEqual(int(ld.records["fav_rank"][7][0]), -1)
        self.assertTrue(np.isnan(ld.records["entropy"][7]).all())
        np.testing.assert_array_equal(depth[8:], 1)
        self.assertTrue((ld.fav_rank[depth >= 0][:, -1] == 0).all(), "the top layer's answer is the favourite")
        # the sampler took the second-best at steps 0, 3, 6, 9: no layer decides THAT token
        cd = ld.chosen_depth
        for i in range(12):
            if i == 7:
                self.assertEqual(cd[i], -1)
            elif i % 3 == 0:
                self.assertEqual(cd[i], -1, f"step {i}: the chosen token was never the lens's answer")
            else:
                self.assertEqual(cd[i], depth[i])
        fs = ld.first_seen
        np.testing.assert_array_equal(fs[:4], 0)
        self.assertTrue((fs[8:] >= 0).all())
        # what each layer says at the top matches the recording's candidates
        ids, lps = ld.top(2)
        self.assertEqual(int(ids[1][0]), int(rec.steps["ids"][2][0]))
        np.testing.assert_allclose(lps[1], rec.steps["logprobs"][2][:8], atol=1e-4)
        # reading helpers
        agree = ld.agreement_by_layer()
        self.assertEqual(agree[1], 1.0)
        self.assertAlmostEqual(float(agree[0]), 4 / 11, places=5)
        self.assertEqual(ld.depth_histogram().tolist(), [4, 7])
        col = ld.column(2, piece=rec.piece, n=2)
        self.assertEqual(len(col), 2)
        self.assertEqual(col[1]["fav_rank"], 0)
        self.assertEqual(col[1]["candidates"][0]["piece"], rec.piece(int(ids[1][0])))
        s = rec.summary()["lens"]
        self.assertEqual(s["label"], "MEASURED")
        self.assertEqual(s["depth"]["histogram"], [4, 7])
        self.assertEqual(s["depth"]["undecided"], 1)
        rows = list_recordings(self.dir / "rec")
        self.assertTrue(rows[0]["lens"])
        self.assertIn("1 of 12 steps had no residual", " ".join(r["notes"]))

    def test_a_lens_is_not_redone_unless_asked(self):
        path, _ = make_recording(self.dir / "rec", self.model, self.u)
        lens.run(path, cache_dir=self.cache)
        with self.assertRaises(FileExistsError):
            lens.run(path, cache_dir=self.cache)
        r = lens.run(path, cache_dir=self.cache, k=4, replace=True)
        self.assertEqual(r["k"], 4)
        self.assertEqual(read(path).lens.k, 4)

    def test_without_the_top_layer_the_lens_is_experimental(self):
        path, _ = make_recording(self.dir / "rec", self.model, self.u)
        r = lens.run(path, cache_dir=self.cache, layers="0")
        self.assertEqual(r["layers"], [0])
        self.assertEqual(r["label"], "EXPERIMENTAL")
        self.assertFalse(r["check"]["made"])
        self.assertIn("l_out-1", r["check"]["why_not"])
        self.assertTrue(any("could not be checked" in n for n in r["notes"]))
        self.assertEqual(parse_layers("0-2,5"), [0, 1, 2, 5])
        self.assertEqual(parse_layers([3, 1, 1]), [1, 3])
        self.assertIsNone(parse_layers(None))
        with self.assertRaises(ValueError):
            parse_layers("5-2")
        with self.assertRaises(LensUnavailable):
            lens.run(path, cache_dir=self.cache, layers="40-41", replace=True)

    def test_a_recording_without_the_residual_is_refused_with_the_reason(self):
        rec = Recorder(n_vocab=self.u.n_vocab, k=4)
        rec.meta["model"] = {"path": str(self.model)}
        x = np.zeros(self.u.n_vocab, np.float32)
        rec.step(x, 1)
        path = rec.save(self.dir / "plain")
        with self.assertRaises(LensUnavailable) as cm:
            lens.run(path, cache_dir=self.cache)
        self.assertIn("--tap residual", str(cm.exception))
        # a Tap that recorded something else
        rec2 = Recorder(n_vocab=self.u.n_vocab, k=4)
        rec2.meta["model"] = {"path": str(self.model)}
        track = TapTrack(("ffn_moe_topk-*",), facts={"n_layer": 2})
        rec2.tap_track = track
        rec2.step(x, 1)
        track.step(0, {"ffn_moe_topk-0": np.array([1, 2], np.int32)})
        path2 = rec2.save(self.dir / "experts")
        with self.assertRaises(LensUnavailable) as cm:
            lens.run(path2, cache_dir=self.cache)
        self.assertIn("ffn_moe_topk", str(cm.exception))

    def test_the_wrong_model_file_is_refused_and_a_missing_one_named(self):
        path, _ = make_recording(self.dir / "rec", self.model, self.u)
        # the cache is empty and the recording names a file that is gone
        rec = read(path)
        rec.meta["global"]["athrec:model"]["path"] = str(self.dir / "gone.gguf")
        with self.assertRaises(LensUnavailable) as cm:
            lens.run(rec, cache_dir=self.dir / "fresh")
        self.assertIn("--model", str(cm.exception))
        # a file at the recorded path that is not the recorded file is refused
        other = tiny_model(self.dir / "other.gguf", n_layers=3, seed=9)
        rec = read(path)
        rec.meta["global"]["athrec:model"]["path"] = str(other)
        with self.assertRaises(LensUnavailable) as cm:
            lens.run(rec, cache_dir=self.dir / "fresh")
        self.assertIn("fingerprints differ", str(cm.exception))
        # named by hand (another quant of the same model, say): allowed, and noted
        r = lens.run(read(path), cache_dir=self.dir / "fresh2", model_path=other)
        self.assertTrue(any("not the file the recording names" in n for n in r["notes"]))
        # a model whose width does not match is refused outright
        wide = tiny_model(self.dir / "wide.gguf", n_layers=2, n_embd=32)
        with self.assertRaises(LensUnavailable) as cm:
            lens.run(read(path), cache_dir=self.dir / "fresh3", model_path=wide, replace=True)
        self.assertIn("wide", str(cm.exception))

    def test_a_damaged_lens_file_is_named_when_read(self):
        path, _ = make_recording(self.dir / "rec", self.model, self.u)
        lens.run(path, cache_dir=self.cache)
        lmeta, ldata = lf.paths(path)
        with open(ldata, "r+b") as f:
            f.truncate(10)
        with self.assertRaises(RecordingError) as cm:
            read(path).lens
        self.assertIn("damaged or incomplete", str(cm.exception))
        s = read(path).summary()["lens"]
        self.assertIn("error", s)
        # verify catches a changed byte
        lens.run(path, cache_dir=self.cache, replace=True)
        with open(ldata, "r+b") as f:
            f.seek(40)
            b = f.read(1)
            f.seek(40)
            f.write(bytes([b[0] ^ 0xFF]))
        with self.assertRaises(RecordingError):
            read(path, verify=True).lens

    def test_run_back(self):
        self.assertEqual(_run_back(np.array([False, True, True])), 1)
        self.assertEqual(_run_back(np.array([True, True, True])), 0)
        self.assertEqual(_run_back(np.array([True, False, True])), 2)
        self.assertEqual(_run_back(np.array([True, True, False])), -1)
        self.assertEqual(_run_back(np.array([], bool)), -1)

    def test_the_record_layout_is_as_documented(self):
        dt = lf.record_dtype(3, 5)
        self.assertEqual(dt.itemsize, 20 + 3 * (8 * 5 + 20))
        self.assertEqual(dt.names, ("flags", "depth", "first_seen", "fav", "chosen_depth", "ids",
                                    "logprobs", "fav_rank", "fav_logprob", "chosen_rank",
                                    "chosen_logprob", "entropy"))
        with self.assertRaises(ValueError):
            lf.record_dtype(0, 5)

    def test_the_command_line(self):
        path, _ = make_recording(self.dir / "rec", self.model, self.u)
        code, out, err = run("lens", str(path), "--data", str(self.dir / "data"),
                             "--show", "5", "--step", "2")
        # --data points the cache at the scratch folder; the model is decoded again there
        self.assertEqual(code, 0, err)
        self.assertIn("MEASURED", out)
        self.assertIn("decision depth", out)
        self.assertIn("step 2:", out)
        self.assertIn("what the layer would say", out)
        self.assertTrue((self.dir / "data" / "lenses").is_dir())
        code, out, err = run("lens", str(path), "--json", "--data", str(self.dir / "data"))
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertTrue(r["already"], "an existing lens is reported, not redone")
        self.assertEqual(r["kind"], "lens")
        self.assertEqual(r["lens_kind"], "logit")
        code, out, err = run("lens", str(path), "--json", "--replace", "--k", "3",
                             "--data", str(self.dir / "data"))
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["k"], 3)
        code, _out, err = run("lens", str(self.dir / "nothing.athrec-meta"), "--json")
        self.assertEqual(code, 3)
        code, out, err = run("recording", str(path), "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["lens"]["k"], 3)
        code, out, err = run("recording", str(path))
        self.assertIn("the lens (logit", out)


# ---------------------------------------------------------------- live
@needs_llama
class Live(unittest.TestCase):
    """The pass on a recording llama.cpp itself made, with the Tap."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = TempDir()
        cls.dir = cls._tmp.__enter__()
        cls.model = tiny_model(cls.dir / "dense.gguf", n_layers=3)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.__exit__(None, None, None)

    def test_the_lens_reproduces_llama_cpps_logits_at_the_top(self):
        import llama_cpp
        from athanor import tap
        from athanor.waterfall import attach
        llm = llama_cpp.Llama(str(self.model), n_ctx=256, verbose=False, seed=1, n_threads=2)
        tap.make_tappable(llm)
        with attach(llm, tap="residual", k=16) as rec:
            llm.create_chat_completion([{"role": "user", "content": "the radio signal"}],
                                       max_tokens=12, temperature=0)
        path = rec.save(self.dir / "live")
        r = lens.run(path, cache_dir=self.dir / "lenses")
        self.assertEqual(r["label"], "MEASURED", r["check"])
        self.assertEqual(r["layers"], [0, 1, 2])
        c = r["check"]
        self.assertEqual(c["favourite_agrees"], c["steps_compared"])
        self.assertLess(c["max_abs_logprob_error"], 1e-3)
        ld = read(path).lens
        self.assertTrue((ld.depth >= 0).all())
        self.assertTrue((ld.fav_rank[:, -1] == 0).all())
        self.assertEqual(int(np.count_nonzero(ld.flags)), 0)
        # with greedy sampling the chosen token is the favourite: the depths agree
        np.testing.assert_array_equal(ld.chosen_depth, ld.depth)
        llm.close()


if __name__ == "__main__":
    unittest.main()


class Pieces(unittest.TestCase):
    """The text of tokens the layers name that the recording never ranked."""

    def test_gpt2_bytes_and_sentencepiece_marks_are_decoded(self):
        from athanor.lens.pieces import _decode_token
        self.assertEqual(_decode_token("Ġyou", "gpt2", 1), " you")
        self.assertEqual(_decode_token("Ċ", "gpt2", 1), "\n")
        self.assertEqual(_decode_token("å¾Īéļ¾", "gpt2", 1), "很难")
        self.assertEqual(_decode_token("<|im_end|>", "gpt2", 3), "<|im_end|>")
        self.assertEqual(_decode_token("▁radio", "llama", 1), " radio")
        self.assertEqual(_decode_token("<0x41>", "llama", 6), "A")
        self.assertEqual(_decode_token("<unk>", "llama", 2), "<unk>")

    def test_the_pass_stores_the_pieces_it_needs_and_the_reader_uses_them(self):
        from athanor.lens.pieces import resolve
        with TempDir() as d:
            model = tiny_model(d / "tiny.gguf", n_layers=2)
            u = build(model, cache_dir=d / "lenses")
            path, _ = make_recording(d / "rec", model, u)
            lens.run(path, cache_dir=d / "lenses")
            rec = read(path)
            ld = rec.lens
            self.assertIn(ld.meta["pieces_from"], ("llama.cpp", "the file's token list"))
            self.assertEqual(ld.unresolved(rec.pieces), set(), "every token the layers name has its text")
            self.assertTrue(ld.unresolved(), "the lens keeps only what the recording lacks")
            named = {int(t) for t in ld.records["ids"].reshape(-1).tolist()} - set(rec.pieces)
            self.assertTrue(named, "the layers name tokens outside the recording's candidates")
            for t in list(named)[:5]:
                self.assertNotEqual(ld.piece(t, rec.piece), f"<{t}>")
            self.assertEqual(ld.piece(int(rec.steps["chosen"][0]), rec.piece),
                             rec.piece(int(rec.steps["chosen"][0])), "the recording's text first")
            # the byte vocabulary's own spellings, straight from the file
            found, how = resolve(model, [5, 5 + 0x41], prefer_llama=False)
            self.assertEqual(how, "the file's token list")
            self.assertEqual(found[5], "\x00")
            self.assertEqual(found[5 + 0x41], "A")
            self.assertEqual(resolve(d / "gone.gguf", [1])[0], {})
            self.assertEqual(resolve(model, [])[0], {})
            # ids past the vocabulary are left out, not invented
            self.assertEqual(resolve(model, [10 ** 6], prefer_llama=False)[0], {})
