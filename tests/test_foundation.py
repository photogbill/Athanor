"""The host port, the labels and the notebook."""

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from athanor import host as H
from athanor.labels import DECLARED, MEASURED, Figure, Finding
from athanor.notebook import Notebook
from fixtures import TempDir


class HostPort(unittest.TestCase):

    def tearDown(self):
        H.reset_host()

    def test_null_host_needs_nothing(self):
        h = H.NullHost()
        with h.borrow_gpu("test"):
            pass
        self.assertIsNone(h.vram_plan())
        self.assertEqual(h.settings(), {})
        self.assertIsInstance(h.data_dir(), Path)

    def test_data_dir_follows_the_environment(self):
        with mock.patch.dict(os.environ, {"ATHANOR_DATA": "/somewhere/else"}):
            self.assertEqual(H.NullHost().data_dir(), Path("/somewhere/else"))

    def test_model_dirs_from_the_environment(self):
        with mock.patch.dict(os.environ, {"ATHANOR_MODELS": os.pathsep.join(["/a", "/b"])}):
            self.assertEqual(H.NullHost().model_dirs(), [Path("/a"), Path("/b")])

    def test_a_host_is_anything_with_the_five_methods(self):
        class Mine:
            def borrow_gpu(self, reason):
                import contextlib
                return contextlib.nullcontext()

            def vram_plan(self):
                return {"free": 1}

            def model_dirs(self):
                return []

            def data_dir(self):
                return Path("/tmp/x")

            def settings(self):
                return {"k": 1}

        H.set_host(Mine())
        self.assertEqual(H.get_host().vram_plan(), {"free": 1})

    def test_a_half_host_is_refused_by_name(self):
        class Half:
            def borrow_gpu(self, reason):
                return None

        with self.assertRaisesRegex(TypeError, "missing vram_plan, model_dirs, data_dir, settings"):
            H.set_host(Half())


class Labels(unittest.TestCase):

    def test_figures_carry_their_label(self):
        self.assertEqual(Figure(3, MEASURED, "s", "timed").to_dict(),
                         {"value": 3, "label": "MEASURED", "unit": "s", "how": "timed"})
        with self.assertRaises(ValueError):
            Figure(3, "GUESSED")

    def test_findings_are_checked(self):
        f = Finding("x", "warn", "a sentence", DECLARED)
        self.assertEqual(f.to_dict()["status"], "warn")
        with self.assertRaises(ValueError):
            Finding("x", "bad", "no")


class NotebookRecord(unittest.TestCase):

    def test_record_note_get(self):
        with TempDir() as d:
            nb = Notebook(d / "nb.jsonl")
            rid = nb.record("inspect", question="q?", inputs=[{"path": "m.gguf"}],
                            settings={"a": 1}, result={"n": 2})
            self.assertEqual(nb.get(rid)["result"], {"n": 2})
            self.assertIn("athanor", nb.get(rid)["versions"])
            nid = nb.note(rid, "looked fine")
            self.assertEqual(nb.notes(rid)[0]["id"], nid)
            self.assertEqual([r["id"] for r in nb.runs()], [rid])
            self.assertEqual(nb.runs("tokenize"), [])

    def test_append_only(self):
        with TempDir() as d:
            nb = Notebook(d / "nb.jsonl")
            r1 = nb.record("a", question="", inputs=[], settings={}, result={})
            first = (d / "nb.jsonl").read_bytes()
            nb.note(r1, "later")
            nb.record("b", question="", inputs=[], settings={}, result={})
            self.assertTrue((d / "nb.jsonl").read_bytes().startswith(first))

    def test_a_note_needs_a_run(self):
        with TempDir() as d:
            with self.assertRaises(KeyError):
                Notebook(d / "nb.jsonl").note("nope", "x")

    def test_a_torn_last_line_is_reported_not_fatal(self):
        with TempDir() as d:
            p = d / "nb.jsonl"
            nb = Notebook(p)
            nb.record("a", question="", inputs=[], settings={}, result={})
            with open(p, "a", encoding="utf-8") as f:
                f.write('{"type": "run", "id": "x"')  # power cut mid-write
            kinds = [x["type"] for x in nb.lines()]
            self.assertEqual(kinds, ["run", "damaged"])
            self.assertEqual(len(nb.runs()), 1)

    def test_every_line_is_json(self):
        with TempDir() as d:
            nb = Notebook(d / "nb.jsonl")
            nb.record("a", question="é", inputs=[], settings={}, result={"x": [1, 2]})
            for line in (d / "nb.jsonl").read_text(encoding="utf-8").splitlines():
                json.loads(line)


if __name__ == "__main__":
    unittest.main()
