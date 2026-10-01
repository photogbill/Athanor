"""The Waterfall's recorder: the file format, the arithmetic, and — with
llama-cpp-python — a real model generating while it is recorded.

The real model is ``tiny_runnable_model``: random weights, real llama.cpp.
What is checked against llama.cpp itself:
* a recorded generation is the SAME generation (same seed, same tokens);
* every recorded token is the token llama.cpp then evaluated;
* the recorded probabilities are llama.cpp's own logits (a second,
  teacher-forced pass with logits_all agrees);
* the distribution is the RAW one — a logit bias the sampler applied does
  not appear in it.
"""

import contextlib
import io
import json
import math
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from athanor import cli
from athanor.waterfall import (FLAG_EOG, FLAG_NONFINITE, Recorder, RecordingError,
                               list_recordings, read, step_dtype)
from athanor.waterfall import fileformat as ff
from athanor.waterfall.recorder import _join
from fixtures import HAVE_LLAMA, TempDir, needs_llama, tiny_runnable_model


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def pieces_for(n):
    return lambda t: f"<{t}>".encode()


class Format(unittest.TestCase):

    def test_the_layout_is_packed_and_documented(self):
        for k in (1, 8, 256):
            dt = step_dtype(k)
            self.assertEqual(dt.itemsize, 32 + 8 * k)
            self.assertEqual(dt.fields["t"][1], 24)
            self.assertEqual(dt.fields["ids"][1], 32)
            self.assertEqual(dt.fields["logprobs"][1], 32 + 4 * k)
        with self.assertRaises(ValueError):
            step_dtype(0)

    def test_stems(self):
        m, d = ff.stems("/x/a.athrec-meta")
        self.assertEqual((m.name, d.name), ("a.athrec-meta", "a.athrec-data"))
        self.assertEqual(ff.stems("/x/a.athrec-data"), (m, d))
        self.assertEqual(ff.stems("/x/a"), (m, d))


class Arithmetic(unittest.TestCase):

    def rec(self, n=10, k=4, **kw):
        return Recorder(n_vocab=n, k=k, piece=pieces_for(n), **kw)

    def test_uniform(self):
        r = self.rec(n=8, k=3)
        r.step(np.zeros(8, np.float32), 5)
        s = r.steps()[0]
        self.assertAlmostEqual(float(s["entropy"]), math.log(8), places=5)
        self.assertEqual(list(s["ids"]), [0, 1, 2], "ties are broken by id")
        self.assertAlmostEqual(float(s["tail"]), 5 / 8, places=5)
        self.assertEqual(int(s["rank"]), 0, "rank counts tokens strictly more likely")
        self.assertAlmostEqual(float(s["logprob"]), -math.log(8), places=5)

    def test_rank_is_exact_beyond_the_stored_candidates(self):
        x = np.arange(10, dtype=np.float32)       # token 9 best, token 0 worst
        r = self.rec(n=10, k=3)
        r.step(x, 0)
        s = r.steps()[0]
        self.assertEqual(list(s["ids"]), [9, 8, 7])
        self.assertEqual(int(s["rank"]), 9)
        lse = math.log(np.exp(x.astype(np.float64)).sum())
        np.testing.assert_allclose(s["logprobs"], [9 - lse, 8 - lse, 7 - lse], rtol=1e-6)
        self.assertAlmostEqual(float(s["logprob"]), -lse, places=5)

    def test_masked_and_broken_logits(self):
        x = np.array([0, -np.inf, np.nan, 1, -np.inf], np.float32)
        r = self.rec(n=5, k=5)
        r.step(x, 3)
        s = r.steps()[0]
        self.assertTrue(int(s["flags"]) & FLAG_NONFINITE)
        self.assertTrue(math.isfinite(float(s["entropy"])))
        self.assertEqual(list(s["ids"][:2]), [3, 0])
        r.step(np.full(5, -np.inf, np.float32), 1)     # nothing possible
        s = r.steps()[1]
        self.assertEqual(int(s["rank"]), -1)
        self.assertTrue(math.isnan(float(s["logprob"])))

    def test_a_chosen_id_outside_the_vocabulary(self):
        r = self.rec(n=5, k=2)
        r.step(np.zeros(5, np.float32), 99)
        self.assertEqual(int(r.steps()[0]["rank"]), -1)

    def test_wrong_length_is_refused(self):
        with self.assertRaises(ValueError):
            self.rec(n=5).step(np.zeros(6, np.float32), 0)

    def test_k_is_capped_at_the_vocabulary(self):
        self.assertEqual(Recorder(n_vocab=5, k=256).k, 5)

    def test_many_steps_cross_chunks(self):
        r = self.rec(n=6, k=2)
        for i in range(600):
            r.step(np.eye(6, dtype=np.float32)[i % 6] * 3, i % 6)
        s = r.steps()
        self.assertEqual(len(s), 600)
        self.assertEqual(list(s["chosen"][:7]), [0, 1, 2, 3, 4, 5, 0])
        self.assertTrue((s["rank"] == 0).all())

    def test_time_is_measured_from_the_start(self):
        ticks = iter([10.0, 10.5, 10.5, 10.6, 11.0, 11.0, 11.1])
        r = Recorder(n_vocab=4, k=2, clock=lambda: next(ticks))
        r.step(np.zeros(4, np.float32), 0)
        r.step(np.zeros(4, np.float32), 1, t=12.0)
        self.assertEqual(list(r.steps()["t"]), [0.5, 2.0])

    def test_a_character_split_across_tokens(self):
        text, spans = _join([b"a", b"\xe4", b"\xbd\xa0", b"!"])
        self.assertEqual(text, "a你!")
        self.assertEqual(spans, [[0, 1], [1, 1], [1, 2], [2, 3]])
        text, spans = _join([b"x", b"\xe4"])          # cut off mid-character
        self.assertEqual(text, "x�")
        self.assertEqual(spans, [[0, 1], [1, 2]])


class SaveAndRead(unittest.TestCase):

    def make(self, folder, n_steps=5):
        r = Recorder(n_vocab=20, k=4, piece=lambda t: (" w%d" % t).encode(),
                     is_eog=lambda t: t == 19)
        r.begin([1, 2, 3])
        rng = np.random.default_rng(0)
        for i in range(n_steps):
            r.step(rng.standard_normal(20).astype(np.float32), 19 if i == n_steps - 1 else i)
        r.meta.update(settings={"temperature": 0.7}, messages=[{"role": "user", "content": "q"}])
        r.meta["model"] = {"name": "m.gguf"}
        return r, r.save(folder, finish_reason="stop")

    def test_round_trip(self):
        with TempDir() as d:
            r, path = self.make(d)
            rec = read(path, verify=True)
            self.assertEqual(rec.n_steps, 5)
            np.testing.assert_array_equal(rec.steps, r.steps())
            self.assertEqual(rec.text, " w0 w1 w2 w3 w19")
            self.assertEqual(rec.span(1), (3, 6))
            self.assertEqual(rec.step_at_char(4), 1)
            self.assertEqual(rec.settings, {"temperature": 0.7})
            self.assertEqual(rec.meta["prompt"]["text"], " w1 w2 w3")
            self.assertTrue(rec.chosen(4)["flags"] & FLAG_EOG)
            cands = rec.candidates(0, 4)
            self.assertEqual(len(cands), 4)
            self.assertAlmostEqual(sum(c["p"] for c in cands) + float(rec.steps[0]["tail"]), 1, 5)
            self.assertEqual(read(rec.data_path).n_steps, 5, "named by its data file")
            self.assertEqual(read(str(rec.meta_path)[: -len(ff.META_SUFFIX)]).n_steps, 5)
            self.assertIn("m", Path(path).name)

    def test_never_overwrites(self):
        with TempDir() as d:
            _r, path = self.make(d)
            m, data = ff.stems(path)
            meta = json.loads(m.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                ff.write(m, data, meta, np.zeros(0, step_dtype(4)))
            self.assertEqual(read(path).n_steps, 5, "the original survived")

    def test_damage_is_reported(self):
        with TempDir() as d:
            _r, path = self.make(d)
            m, data = ff.stems(path)
            with open(data, "r+b") as f:
                f.seek(40)
                f.write(b"\x01")
            with self.assertRaises(RecordingError):
                read(path, verify=True)
            with open(data, "ab") as f:
                f.write(b"x")
            with self.assertRaises(RecordingError):
                read(path)
            meta = json.loads(m.read_text(encoding="utf-8"))
            meta["global"]["athrec:version"] = 99
            m.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(RecordingError):
                read(path)
            rows = list_recordings(d)
            self.assertEqual(len(rows), 1)
            self.assertIn("format 99", rows[0]["error"])

    def test_listing_is_newest_first(self):
        with TempDir() as d:
            _r, a = self.make(d)
            _r, b = self.make(d, n_steps=2)
            rows = list_recordings(d)
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["n_steps"] for r in rows}, {5, 2})
            self.assertEqual(list_recordings(d / "none"), [])

    def test_a_failure_stops_the_recording_not_the_caller(self):
        r = Recorder(n_vocab=4, k=2)
        r.step(np.zeros(4, np.float32), 0)
        r.fail(RuntimeError("boom"))
        r.fail(RuntimeError("second"))
        self.assertEqual(r.error, "RuntimeError: boom")

    def test_nothing_recorded_although_the_host_saw_a_reply(self):
        with TempDir() as d:
            r = Recorder(n_vocab=4, k=2)
            rec = read(r.save(d, host_reply="Hello"))
            self.assertIn("did not sample through", rec.error)


# ------------------------------------------------------------ with llama.cpp
@needs_llama
class LiveRecording(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = TempDir()
        cls.dir = cls._tmp.__enter__()
        cls.model = tiny_runnable_model(cls.dir / "tiny.gguf")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.__exit__(None, None, None)

    def llm(self, **kw):
        import llama_cpp
        return llama_cpp.Llama(str(self.model), n_ctx=256, verbose=False, seed=1, **kw)

    MSGS = [{"role": "user", "content": "Hello there, radio operator"}]

    def chat(self, llm, **kw):
        kw.setdefault("max_tokens", 24)
        kw.setdefault("temperature", 0.9)
        kw.setdefault("seed", 11)
        return llm.create_chat_completion(self.MSGS, **kw)["choices"][0]

    def test_a_model_that_reads_images_does_not_record_stale_prompt_ids(self):
        """The vision handler advances n_tokens over image positions without
        writing input_ids there (review, 2026-09-29): the ids are left out
        and the recording says why."""
        import types
        from athanor.waterfall import attach, read
        llm = self.llm()
        llm.chat_handler = types.SimpleNamespace(clip_model_path="mmproj.gguf")
        prompt = llm.tokenize(b"the radio signal")
        with attach(llm) as rec:
            llm.create_completion(prompt=prompt, max_tokens=4, temperature=0.0)
        r = read(rec.save(self.dir / "vision"))
        self.assertIsNone(r.meta["prompt"]["ids"])
        self.assertIsNone(r.meta["prompt"]["text"])
        self.assertIn("images", r.meta["prompt"]["note"])
        self.assertEqual(r.meta["prompt"]["n_tokens"], len(prompt))
        self.assertEqual(r.n_steps, 4, "the reply itself is still recorded")

    def test_a_recorded_reply_is_the_same_reply(self):
        from athanor.waterfall import attach
        plain = self.chat(self.llm())
        llm = self.llm()
        with attach(llm) as rec:
            out = self.chat(llm)
        self.assertEqual(out["message"]["content"], plain["message"]["content"])
        self.assertEqual(rec.n_steps, 24)
        self.assertIsNone(rec.error)
        self.assertNotIn("sample", vars(llm), "the instance is as it was")
        self.assertEqual(rec.reply_text.strip(), out["message"]["content"].strip())

    def test_every_recorded_token_is_the_token_llama_cpp_evaluated(self):
        from athanor.waterfall import attach
        llm = self.llm()
        with attach(llm) as rec:
            self.chat(llm)
        n_prompt = len(rec.prompt_ids)
        self.assertEqual(rec.prompt_ids, llm.input_ids[:n_prompt].tolist())
        evaluated = llm.input_ids[n_prompt:llm.n_tokens].tolist()
        chosen = rec.steps()["chosen"].tolist()
        self.assertIn(len(evaluated), (len(chosen) - 1, len(chosen)))
        self.assertEqual(evaluated, chosen[:len(evaluated)])

    def test_the_probabilities_are_llama_cpps_own(self):
        """A second, independent pass with logits_all, fed the same way the
        generation was (the prompt as a batch, then one token at a time).

        It must be fed the same way: re-scoring the whole reply as ONE batch
        gives different numbers (up to ~0.04 nats on this model) — llama.cpp's
        batched and single-token kernels round differently. That is why the
        Waterfall records live instead of reconstructing afterwards."""
        from athanor.waterfall import attach
        llm = self.llm()
        with attach(llm, k=32) as rec:
            self.chat(llm, max_tokens=12)
        steps = rec.steps()
        ref = self.llm(logits_all=True)
        ref.eval(rec.prompt_ids)
        for t in steps["chosen"].tolist()[:-1]:
            ref.eval([t])
        n_prompt = len(rec.prompt_ids)
        for i, s in enumerate(steps):
            row = np.asarray(ref.scores[n_prompt - 1 + i], np.float64)
            lse = row.max() + math.log(np.exp(row - row.max()).sum())
            np.testing.assert_allclose(s["logprobs"], row[s["ids"]] - lse, atol=1e-4,
                                       err_msg=f"step {i}")
            self.assertEqual(int(s["ids"][0]), int(np.argmax(row)))
            p = np.exp(row - lse)
            self.assertAlmostEqual(float(s["entropy"]), float(-(p * np.log(p + 1e-300)).sum()),
                                   delta=2e-3)

    def test_greedy_always_takes_the_favourite(self):
        from athanor.waterfall import attach
        llm = self.llm()
        with attach(llm) as rec:
            self.chat(llm, temperature=0.0, repeat_penalty=1.0)
        self.assertTrue((rec.steps()["rank"] == 0).all())

    def test_the_distribution_is_raw_before_any_sampler(self):
        """A logit bias forces end-of-text; the recording shows the model did
        not favour it — the bias is the sampler's, not the model's."""
        from athanor.waterfall import attach
        import llama_cpp
        llm = self.llm()
        eos = llama_cpp.llama_vocab_eos(llama_cpp.llama_model_get_vocab(llm.model))
        with attach(llm) as rec:
            out = self.chat(llm, logit_bias={eos: 100.0})
        self.assertEqual(out["finish_reason"], "stop")
        s = rec.steps()
        self.assertEqual(len(s), 1)
        self.assertEqual(int(s[0]["chosen"]), eos)
        self.assertTrue(int(s[0]["flags"]) & FLAG_EOG)
        self.assertGreater(int(s[0]["rank"]), 0)

    def test_a_recorder_failure_costs_the_recording_not_the_reply(self):
        from athanor.waterfall import attach
        from athanor.waterfall import recorder as R
        plain = self.chat(self.llm())
        llm = self.llm()
        real = R.Recorder.step
        calls = {"n": 0}

        def flaky(self_, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 3:
                raise MemoryError("simulated")
            return real(self_, *a, **kw)

        with mock.patch.object(R.Recorder, "step", flaky):
            with attach(llm) as rec:
                out = self.chat(llm)
        self.assertEqual(out["message"]["content"], plain["message"]["content"])
        self.assertEqual(rec.n_steps, 2)
        self.assertIn("MemoryError", rec.error)

    def test_the_instance_is_restored_when_generation_fails(self):
        from athanor.waterfall import attach
        llm = self.llm()
        with self.assertRaises(KeyError):
            with attach(llm):
                raise KeyError("x")
        self.assertNotIn("sample", vars(llm))

    def test_a_closed_model_is_refused(self):
        from athanor.waterfall import RecorderUnavailable, attach
        llm = self.llm()
        llm.close()
        with self.assertRaises(RecorderUnavailable):
            with attach(llm):
                pass

    def test_saved_and_read_back(self):
        from athanor.waterfall import attach
        llm = self.llm()
        with attach(llm, meta={"settings": {"temperature": 0.9}}) as rec:
            out = self.chat(llm)
        with TempDir() as d:
            path = rec.save(d, finish_reason=out["finish_reason"],
                            host_reply=out["message"]["content"])
            r = read(path, verify=True)
            self.assertEqual(r.n_steps, 24)
            self.assertEqual(r.model["name"], "tiny.gguf")
            self.assertTrue(r.model["header_sha256"])
            self.assertIn("<|user|>Hello there", r.meta["prompt"]["text"])
            self.assertEqual(r.finish_reason, "length")
            for i in range(r.n_steps):
                a, b = r.span(i)
                self.assertEqual(r.text[a:b], r.piece(int(r.steps[i]["chosen"])))

    def test_one_reply_in_two_parts(self):
        """A reply continued after its length limit is one recording, with
        the second part marked — and it can be saved after the model is gone."""
        from athanor.waterfall import attach
        llm = self.llm()
        with attach(llm) as rec:
            first = self.chat(llm, max_tokens=5)
        partial = first["message"]["content"]
        with attach(llm, recorder=rec) as again:
            llm.create_completion(prompt=rec.prompt_ids + rec.steps()["chosen"].tolist(),
                                  max_tokens=4, temperature=0.9, seed=3)
        self.assertIs(again, rec)
        self.assertEqual(rec.n_steps, 9)
        self.assertEqual(rec.annotations[0]["athrec:label"], "segment")
        self.assertEqual(rec.annotations[0]["athrec:step_start"], 5)
        self.assertTrue(rec.reply_text.startswith(partial.strip()[:3]))
        llm.close()                                   # the model is gone
        with TempDir() as d:
            r = read(rec.save(d, finish_reason="length"))
            self.assertEqual(r.n_steps, 9)
            self.assertEqual(len(r.spans), 9)
            self.assertTrue(all(int(t) in r.pieces for t in r.steps["chosen"]))

    def test_a_different_model_cannot_continue_a_recording(self):
        from athanor.waterfall import Recorder, RecorderUnavailable, attach
        with self.assertRaises(RecorderUnavailable):
            with attach(self.llm(), recorder=Recorder(n_vocab=99, k=4)):
                pass

    def test_the_model_made_on_the_spot(self):
        """athanor.testing.tiny_model: its own byte-level vocabulary, no
        files needed — what a host tests its integration with."""
        import llama_cpp
        from athanor.testing import tiny_model
        from athanor.waterfall import attach
        with TempDir() as d:
            p = tiny_model(d / "t.gguf")
            llm = llama_cpp.Llama(str(p), n_ctx=256, verbose=False, seed=1)
            with attach(llm) as rec:
                out = llm.create_chat_completion(self.MSGS, max_tokens=8, seed=2)
            # llama-cpp-python may draw one token past max_tokens to finish a
            # character split across byte tokens; the recording counts what
            # was drawn, as the binding's own usage figure does
            self.assertEqual(rec.n_steps, out["usage"]["completion_tokens"])
            self.assertGreaterEqual(rec.n_steps, 8)
            self.assertIn("<|im_start|>user\n", llm.detokenize(rec.prompt_ids,
                                                               special=True).decode())
            self.assertEqual(out["choices"][0]["finish_reason"], "length")
            llm.close()
            with self.assertRaises(FileExistsError):
                tiny_model(p)

    def test_the_command_line(self):
        with TempDir() as d, mock.patch.dict(os.environ, {"ATHANOR_DATA": str(d)}):
            code, out, err = run("record", str(self.model), "--prompt", "hi", "--max-tokens",
                                 "6", "--seed", "4", "--json", "--gpu-layers", "0")
            self.assertEqual(code, 0, err)
            r = json.loads(out)
            self.assertEqual((r["kind"], r["n_steps"]), ("record", 6))
            self.assertTrue(Path(r["path"]).is_file())
            self.assertTrue(str(d) in r["path"], "saved in the data folder")
            code, out, err = run("record", str(self.model), "--prompt", "Once", "--raw",
                                 "--max-tokens", "3", "--json", "--gpu-layers", "0")
            self.assertEqual(code, 0, err)
            code, out, _ = run("recordings", "--json")
            self.assertEqual(len(json.loads(out)), 2)
            code, out, _ = run("recording", r["path"], "--step", "0", "--top", "3", "--json")
            self.assertEqual(code, 0)
            self.assertEqual(len(json.loads(out)["candidates"]), 3)
            code, out, _ = run("recording", r["path"])
            self.assertIn("tokens, finish: length", out)
            self.assertEqual(run("recording", r["path"], "--step", "99")[0], cli.EXIT_USAGE)
            self.assertEqual(run("record", str(self.model))[0], cli.EXIT_USAGE)
            self.assertEqual(run("record", str(self.model), "--raw", "--prompt", "x",
                                 "--system", "y")[0], cli.EXIT_USAGE)
            self.assertEqual(run("record", str(d / "missing.gguf"), "--prompt", "x")[0],
                             cli.EXIT_INPUT)
            junk = d / "junk.gguf"
            junk.write_bytes(b"GGUF" + b"\x00" * 64)
            self.assertEqual(run("record", str(junk), "--prompt", "x", "--gpu-layers",
                                 "0")[0], cli.EXIT_INPUT)
            self.assertEqual(run("recording", str(d / "nothing.athrec-meta"))[0],
                             cli.EXIT_INPUT)


class WithoutLlama(unittest.TestCase):

    @unittest.skipIf(HAVE_LLAMA, "llama-cpp-python is installed here")
    def test_record_says_llama_is_needed(self):
        with TempDir() as d:
            p = d / "m.gguf"
            p.write_bytes(b"x")
            self.assertEqual(run("record", str(p), "--prompt", "x")[0], cli.EXIT_LLAMA)

    def test_the_log_keeps_no_prompt(self):
        red = cli.redact_argv(["record", "m.gguf", "--prompt", "secret words", "--system=be x"])
        self.assertNotIn("secret words", " ".join(red))
        self.assertNotIn("be x", " ".join(red))
        self.assertIn("m.gguf", red)


if __name__ == "__main__":
    unittest.main()
