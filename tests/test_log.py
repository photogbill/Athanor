"""Athanor's own log: what happened, kept locally so failures can be learned from."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from athanor import log
from fixtures import TempDir, VOCABS, needs_llama

ROOT = Path(__file__).resolve().parents[1]


class TheLog(unittest.TestCase):

    def setUp(self):
        self._tmp = TempDir()
        self.data = self._tmp.__enter__()
        self._env = mock.patch.dict(os.environ, {"ATHANOR_DATA": str(self.data)})
        self._env.start()
        log.configure(enabled=None)

    def tearDown(self):
        self._env.stop()
        self._tmp.__exit__(None, None, None)
        log.reset()

    def test_an_event_is_one_json_line(self):
        log.event("test", value=3, text="é", sync=True)
        lines = log.log_path().read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[-1])
        self.assertEqual((rec["event"], rec["value"], rec["text"]), ("test", 3, "é"))
        self.assertEqual(rec["pid"], os.getpid())
        self.assertTrue(rec["time"].endswith("Z"))
        self.assertEqual(log.log_path().parent, self.data / "logs")

    def test_off_means_off(self):
        with mock.patch.dict(os.environ, {"ATHANOR_LOG": "0"}):
            log.event("test")
        self.assertFalse(log.log_path().exists())
        log.configure(enabled=False)
        log.event("test")
        self.assertFalse(log.log_path().exists())

    def test_a_host_can_send_it_elsewhere(self):
        log.configure(directory=self.data / "elsewhere")
        log.event("test")
        self.assertTrue((self.data / "elsewhere").is_dir())

    def test_exceptions_carry_their_traceback(self):
        try:
            raise ValueError("bad thing")
        except ValueError as exc:
            log.exception("failed", exc, step="x")
        rec = log.read_events()[-1]
        self.assertEqual(rec["error"], "ValueError")
        self.assertIn("raise ValueError", rec["traceback"])

    def test_a_log_that_cannot_be_written_breaks_nothing(self):
        log.configure(directory=self.data / "file")
        (self.data / "file").write_text("not a folder")
        log.event("test")      # must not raise

    def test_a_native_crash_leaves_a_stack(self):
        code = ("import faulthandler, os\n"
                "from athanor import log\n"
                "log.enable_crash_log()\n"
                "log.breadcrumb('about to crash', where='test')\n"
                "faulthandler._sigsegv()\n")
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                           env=dict(os.environ, ATHANOR_DATA=str(self.data)), timeout=120)
        self.assertNotEqual(r.returncode, 0)
        crash = next((self.data / "logs").glob("crash-*.log")).read_text(encoding="utf-8")
        self.assertIn("Fatal Python error", crash)
        crumbs = [e for e in log.read_events() if e["event"] == "breadcrumb"]
        self.assertEqual(crumbs[-1]["what"], "about to crash", "the last thing done is on disk")

    @needs_llama
    def test_a_vocab_load_is_logged_in_full(self):
        from athanor.vocab import Vocab
        with Vocab(VOCABS["gpt-neox"]):
            pass
        ev = [e for e in log.read_events() if e["event"] in ("breadcrumb", "vocab-load")]
        self.assertEqual(ev[-2]["what"], "vocab-load")
        load = ev[-1]
        self.assertEqual(load["n_tokens"], 50432)
        levels = {lvl for lvl, _t in load["llama_cpp_log"]}
        self.assertTrue({"debug", "info", "warn"} <= levels, levels)
        self.assertIn("missing pre-tokenizer", " ".join(t for _l, t in load["llama_cpp_log"]))


if __name__ == "__main__":
    unittest.main()
