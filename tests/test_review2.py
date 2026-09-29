"""Second-round review findings (verification pass on 0.1), each pinned."""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from athanor import gguf
from athanor.cli import redact_argv
from athanor.gguf import GGMLType, GGUFWriter, ValueType
from athanor.util import clean_str
from fixtures import TempDir, VOCABS, needs_llama, tiny_gguf

ROOT = Path(__file__).resolve().parents[1]


@needs_llama
class ClosedVocab(unittest.TestCase):

    def test_every_native_call_is_refused_after_close(self):
        from athanor.vocab import Vocab
        v = Vocab(VOCABS["llama-spm"])
        v.close()
        self.assertTrue(v.closed)
        for call in (lambda: v.meta("general.architecture"), v.eog_ids, v.special_ids,
                     lambda: v.type, lambda: v.add_bos, lambda: v.add_eos,
                     lambda: v.tokenize("x"), lambda: v.piece(1), lambda: v.detokenize([1])):
            with self.assertRaises(RuntimeError):
                call()
        v.close()   # twice is harmless


@needs_llama
class TemplateFormats(unittest.TestCase):

    def statuses(self, r):
        return {(f["check"], f["status"]) for f in r["findings"]}

    def test_a_library_template_is_judged_by_its_own_name(self):
        """detect() calls the library's mistral-v3 text 'llama2'; the check
        must use 'mistral-v3', whose models have [INST] as a control token."""
        r = self._analyse("mistral-v3")
        self.assertEqual(r["format"], "mistral-v3")
        self.assertIn(("marker-is-special", "problem"), self.statuses(r))

    def test_bos_as_the_runners_really_send_it(self):
        """A model whose tokenizer adds no BOS still gets one from a template
        that writes bos_token — both runners pass the BOS text."""
        from athanor.tabs import template as TT
        r = TT.analyse("llama3", model=str(VOCABS["gpt-neox"]))
        b = r["bos"]
        self.assertEqual(b["expected"], 0)
        self.assertEqual(b["llama_cpp_python"], 1)
        self.assertIn(("bos-once", "warn"), self.statuses(r))

    def _analyse(self, spec):
        from athanor.tabs import template as TT
        return TT.analyse(spec, model=str(VOCABS["llama-spm"]))


class WriterAndText(unittest.TestCase):

    def test_the_alignment_cannot_be_set_behind_the_keys_back(self):
        w = GGUFWriter()
        with self.assertRaises(AttributeError):
            w.alignment = 64
        w.add("general.alignment", 64, ValueType.UINT32)
        self.assertEqual(w.alignment, 64)

    def test_without_hard_links_the_target_is_still_never_overwritten(self):
        with TempDir() as d:
            with mock.patch("os.link", side_effect=OSError(1, "Operation not permitted")):
                p = tiny_gguf(d / "t.gguf")
                self.assertEqual(gguf.validate(gguf.read(p)), [])
                self.assertFalse((d / "t.gguf.part").exists())
                w = GGUFWriter()
                w.add("general.architecture", "llama")
                real = w.write

                def racing(stream):
                    (d / "u.gguf").write_bytes(b"precious")
                    return real(stream)

                w.write = racing
                with self.assertRaises(FileExistsError):
                    w.write_file(d / "u.gguf")
                self.assertEqual((d / "u.gguf").read_bytes(), b"precious")

    def test_any_lone_surrogate_can_be_written(self):
        self.assertEqual(clean_str("a\ud800b"), "a\\ud800b")
        self.assertEqual(clean_str(b"caf\xe9".decode("utf-8", "surrogateescape")), "caf\\xe9")
        self.assertEqual(clean_str("plain é"), "plain é")

    def test_a_partial_block_is_not_called_an_unknown_type(self):
        from athanor.tabs import inspect as I
        with TempDir() as d:
            p = tiny_gguf(d / "t.gguf")
            g = gguf.read(p)
            g.tensors[1].ggml_type = int(GGMLType.Q4_K)
            msgs = " ".join(f.message for f in I.checks(g))
            self.assertIn("not a whole number of Q4_K blocks", msgs)
            self.assertNotIn("unknown to llama.cpp", msgs)


class TheLogKeepsNoText(unittest.TestCase):

    def test_typed_text_is_redacted(self):
        out = redact_argv(["tokenize", "m.gguf", "--text", "SECRET name 555-0142", "--json"])
        self.assertEqual(out[:3], ["tokenize", "m.gguf", "--text"])
        self.assertTrue(out[3].startswith("<text: 20 chars"))
        self.assertNotIn("SECRET", " ".join(out))
        self.assertNotIn("SECRET", " ".join(redact_argv(["tokenize", "m", "--text=SECRET"])))
        note = redact_argv(["notebook", "note", "RUN1", "the", "informant"])
        self.assertEqual(note[:3], ["notebook", "note", "RUN1"])
        self.assertNotIn("informant", " ".join(note))

    def test_a_cli_run_leaves_no_text_in_either_log(self):
        with TempDir() as d:
            env = dict(os.environ, ATHANOR_DATA=str(d))
            subprocess.run([sys.executable, "-m", "athanor", "template", "chatml", "--no-record",
                            "--json", "--conversation", "missing-SECRET.json"],
                           cwd=ROOT, env=env, capture_output=True, timeout=120)
            subprocess.run([sys.executable, "-m", "athanor", "tokenize", str(VOCABS["phi-3"]),
                            "--text", "SECRET informant", "--no-record", "--json"],
                           cwd=ROOT, env=env, capture_output=True, timeout=120)
            logs = "".join(p.read_text(encoding="utf-8") for p in (d / "logs").iterdir())
            self.assertNotIn("SECRET informant", logs)

    def test_a_deleted_working_folder_does_not_stop_the_cli(self):
        with TempDir() as d:
            gone = d / "gone"
            gone.mkdir()
            code = ("import os, sys\n"
                    f"os.chdir({str(gone)!r}); os.rmdir({str(gone)!r})\n"
                    f"sys.path.insert(0, {str(ROOT)!r})\n"
                    "from athanor.cli import main\n"
                    "sys.exit(main(['version', '--json']))\n")
            r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, ATHANOR_DATA=str(d)),
                               capture_output=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))


@needs_llama
class ScriptsWithoutSpaces(unittest.TestCase):

    def test_thai_is_measured_in_characters(self):
        from athanor.tabs import tokenize as K
        from athanor.vocab import Vocab
        with Vocab(VOCABS["llama-spm"]) as v:
            m = K.measure(v, "ภาษาไทยเป็นภาษาที่ยาก and English")
        thai = m["by_script"]["THAI"]
        self.assertIsNone(thai["tokens_per_word"])
        self.assertGreater(thai["tokens_per_char"], 0)
        self.assertTrue(any("without spaces" in n for n in m["notes"]))


if __name__ == "__main__":
    unittest.main()
