"""The tokenizer through llama.cpp itself — needs llama-cpp-python.

Against the bundled vocab-only files (always), llama.cpp's full vocabulary
set (ATHANOR_TEST_VOCAB_DIR) and a real model (ATHANOR_TEST_MODEL).
"""

import unittest

from athanor.capabilities import probe
from fixtures import REAL_MODEL, VOCABS, extra_vocab, needs_llama


@needs_llama
class Vocabulary(unittest.TestCase):

    def test_loads_and_counts(self):
        from athanor.vocab import Vocab
        with Vocab(VOCABS["llama-spm"]) as v:
            self.assertEqual(v.type, "SPM")
            self.assertEqual(v.n_tokens, 32000)
            self.assertTrue(v.add_bos)
            ids = v.special_ids()
            self.assertEqual((ids["bos"], ids["eos"]), (1, 2))
            self.assertLess(v.load_seconds, 30)

    def test_tokenize_and_back(self):
        from athanor.vocab import Vocab
        with Vocab(VOCABS["llama-spm"]) as v:
            text = "Hello world, 10.0.0.5 café"
            ids = v.tokenize(text)
            self.assertEqual(v.detokenize(ids).strip(), text)
            self.assertEqual(v.tokenize(text, add_special=True)[0], 1, "BOS added")
            self.assertEqual(b"".join(v.piece(i) for i in ids).decode().strip(), text)

    def test_special_text_is_parsed_only_when_asked(self):
        from athanor.vocab import Vocab
        with Vocab(VOCABS["phi-3"]) as v:
            one = v.tokenize("<|end|>", parse_special=True)
            many = v.tokenize("<|end|>", parse_special=False)
            self.assertEqual(len(one), 1)
            self.assertGreater(len(many), 1)
            self.assertIn("control", v.attr_names(one[0]))

    def test_long_input_grows_the_buffer(self):
        from athanor.vocab import Vocab
        with Vocab(VOCABS["llama-spm"]) as v:
            self.assertGreater(len(v.tokenize("x " * 20000)), 20000)

    def test_the_load_log_is_captured_at_the_right_level(self):
        """b11093's levels: 3 is WARN. The binding's own logger maps the old
        order; Athanor must not."""
        from athanor.vocab import Vocab
        with Vocab(VOCABS["gpt-neox"]) as v:
            warns = " ".join(t for _l, t in v.log_lines())
            self.assertIn("missing pre-tokenizer", warns)
            self.assertNotIn("loaded meta data", warns, "INFO lines are not warnings")

    def test_capture_keeps_llama_quiet_and_restores(self):
        from athanor.vocab import Vocab
        with Vocab(VOCABS["phi-3"]) as v:
            with v.capture() as said:
                v.tokenize("<s>hi", add_special=True, parse_special=True)
            self.assertIn("2 BOS", " ".join(t for _l, t in said))

    def test_a_file_that_is_not_there(self):
        from athanor.vocab import Vocab
        with self.assertRaises(FileNotFoundError):
            Vocab("/no/such/model.gguf")

    def test_more_vocabularies_when_available(self):
        p = extra_vocab("llama-bpe")
        if not p:
            self.skipTest("ATHANOR_TEST_VOCAB_DIR not set")
        from athanor.vocab import Vocab
        with Vocab(p) as v:
            self.assertEqual(v.type, "BPE")
            self.assertEqual(v.special_ids()["eot"], 128009)

    def test_a_real_model_vocab_only(self):
        if not REAL_MODEL:
            self.skipTest("ATHANOR_TEST_MODEL not set")
        from athanor.vocab import Vocab
        with Vocab(REAL_MODEL) as v:
            self.assertGreater(v.n_tokens, 1000)
            self.assertTrue(v.tokenize("hello"))
            print(f"\n  vocab-only load of {REAL_MODEL}: {v.load_seconds:.2f} s (spike S4)")


@needs_llama
class ReviewFindings(unittest.TestCase):

    def test_an_id_outside_the_vocabulary_is_an_error_not_a_dead_process(self):
        """llama.cpp answers one with an uncaught C++ exception (the host
        process dies). Athanor checks first."""
        from athanor.vocab import Vocab
        with Vocab(VOCABS["llama-spm"]) as v:
            for bad in (-1, v.n_tokens, 10**9):
                with self.assertRaises(IndexError):
                    v.piece(bad)
                with self.assertRaises(IndexError):
                    v.text(bad)
                with self.assertRaises(IndexError):
                    v.attr(bad)
                with self.assertRaises(IndexError):
                    v.detokenize([1, bad])
            with self.assertRaises(TypeError):
                v.piece("7")
        with self.assertRaises(RuntimeError):
            v.piece(1)   # closed

    def test_the_hosts_own_logger_is_put_back(self):
        import ctypes
        import llama_cpp as lc
        from athanor.vocab import Vocab
        seen = []

        @lc.llama_log_callback
        def host_cb(level, text, ud):
            seen.append(text)

        def current():
            cb = lc.llama_log_callback()
            ud = ctypes.c_void_p()
            lc.llama_log_get(ctypes.byref(cb), ctypes.byref(ud))
            return ctypes.cast(cb, ctypes.c_void_p).value

        lc.llama_log_set(host_cb, ctypes.c_void_p(0))
        try:
            with Vocab(VOCABS["phi-3"]) as v:
                with v.capture():
                    v.tokenize("<s>x", add_special=True)
            self.assertEqual(current(), ctypes.cast(host_cb, ctypes.c_void_p).value)
            self.assertEqual(seen, [], "nothing leaked to the host while Athanor captured")
        finally:
            from athanor.vocab import _restore_logger
            _restore_logger(lc)

    def test_the_retry_path_when_the_first_buffer_is_too_small(self):
        from unittest import mock
        from athanor import vocab as V
        with V.Vocab(VOCABS["llama-spm"]) as v:
            want = v.tokenize("Hello world, this is a longer sentence.")
            with mock.patch.object(V, "_initial_capacity", lambda n: 1):
                self.assertEqual(v.tokenize("Hello world, this is a longer sentence."), want)

    def test_meta_is_not_cut_short(self):
        from athanor import gguf
        from athanor.vocab import Vocab
        want = gguf.read(VOCABS["phi-3"]).get("tokenizer.chat_template")
        with Vocab(VOCABS["phi-3"]) as v:
            self.assertEqual(v.meta("tokenizer.chat_template"), want)


@needs_llama
class Capabilities(unittest.TestCase):

    def test_the_pinned_binding_has_what_phase_1_needs(self):
        caps = probe()
        self.assertTrue(caps["binding"]["installed"])
        self.assertTrue(caps["features"]["vocab"]["available"], caps["features"]["vocab"])
        self.assertIn("eval_callback", caps["features"])


if __name__ == "__main__":
    unittest.main()
