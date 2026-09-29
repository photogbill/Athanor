"""The command line: JSON on stdout, stable exit codes, the notebook."""

import contextlib
import io
import json
import os
import unittest
from unittest import mock

from athanor import cli
from fixtures import TempDir, VOCABS, needs_llama, tiny_gguf


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CommandLine(unittest.TestCase):

    def setUp(self):
        self._tmp = TempDir()
        self.data = self._tmp.__enter__()
        self._env = mock.patch.dict(os.environ, {"ATHANOR_DATA": str(self.data)})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.__exit__(None, None, None)

    def test_json_is_all_that_is_printed(self):
        p = tiny_gguf(self.data / "t.gguf")
        code, out, _ = run("inspect", str(p), "--no-llama", "--json")
        self.assertEqual(code, 0)
        r = json.loads(out)
        self.assertEqual(r["kind"], "inspect")

    def test_runs_are_recorded_unless_asked_not_to(self):
        p = tiny_gguf(self.data / "t.gguf")
        run("inspect", str(p), "--no-llama", "--json")
        run("inspect", str(p), "--no-llama", "--json", "--no-record")
        code, out, _ = run("notebook", "list", "--json")
        runs = json.loads(out)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["kind"], "inspect")
        self.assertEqual(runs[0]["inputs"][0]["name"], "t.gguf")

    def test_exit_codes(self):
        self.assertEqual(run("inspect", str(self.data / "missing.gguf"))[0], cli.EXIT_INPUT)
        bad = self.data / "bad.gguf"
        bad.write_bytes(b"not a gguf at all")
        self.assertEqual(run("inspect", str(bad))[0], cli.EXIT_INPUT)
        self.assertEqual(run("template")[0], cli.EXIT_USAGE)
        with self.assertRaises(SystemExit):
            run("no-such-command")
        few = tiny_gguf(self.data / "few.gguf", emb_rows=60)
        self.assertEqual(run("inspect", str(few), "--no-llama")[0], cli.EXIT_OK)
        self.assertEqual(run("inspect", str(few), "--no-llama", "--strict")[0], cli.EXIT_STRICT)

    def test_the_template_list(self):
        code, out, _ = run("template", "--list", "--json")
        self.assertEqual(len(json.loads(out)), 55)

    def test_roundtrip(self):
        p = tiny_gguf(self.data / "t.gguf")
        code, out, _ = run("roundtrip", str(p), "--whole", "--json")
        r = json.loads(out)
        self.assertTrue(r["header_identical"] and r["file_identical"])
        self.assertEqual(code, 0)

    def test_capabilities_answers_either_way(self):
        code, out, _ = run("capabilities", "--json")
        self.assertEqual(code, 0)
        self.assertIn("features", json.loads(out))

    def test_notebook_note_and_show(self):
        p = tiny_gguf(self.data / "t.gguf")
        run("inspect", str(p), "--no-llama", "--json")
        rid = json.loads(run("notebook", "list", "--json")[1])[0]["id"]
        self.assertEqual(run("notebook", "note", rid, "checked", "by", "hand")[0], 0)
        shown = json.loads(run("notebook", "show", rid, "--json")[1])
        self.assertEqual(shown["notes"][0]["text"], "checked by hand")

    @needs_llama
    def test_the_other_commands_in_json(self):
        a, b = str(VOCABS["llama-spm"]), str(VOCABS["phi-3"])
        for argv in (("tokenize", a, b, "--text", "hello", "--context", "4096"),
                     ("splits", a), ("compare", a, b), ("overlap", a, b),
                     ("template", "phi3", "--model", b), ("template", "phi3", "--vs", "chatml",
                                                          "--model", b)):
            code, out, err = run(*argv, "--json")
            self.assertEqual(code, 0, (argv, err))
            json.loads(out)

    def test_a_cp1252_pipe_does_not_kill_the_report(self):
        """Windows' default for a redirected stdout. '✓' is not in cp1252."""
        import subprocess
        import sys
        from pathlib import Path
        p = tiny_gguf(self.data / "t.gguf")
        env = dict(os.environ, PYTHONIOENCODING="cp1252:strict")
        r = subprocess.run([sys.executable, "-m", "athanor", "inspect", str(p), "--no-llama",
                            "--no-record"], cwd=Path(__file__).resolve().parents[1],
                           capture_output=True, env=env, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))
        self.assertIn("✓".encode("utf-8"), r.stdout)

    def test_a_token_that_is_not_utf8_does_not_lose_the_result(self):
        from fixtures import tiny_tokens
        bad = b"caf\xe9".decode("utf-8", "surrogateescape")
        a = tiny_gguf(self.data / "a.gguf")
        b = tiny_gguf(self.data / "b.gguf", tokens=tiny_tokens(63) + [bad])
        code, out, err = run("compare", str(a), str(b), "--json")
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertIn("caf\\xe9", r["tokenizer"]["only_in_b"]["sample"])
        runs = json.loads(run("notebook", "list", "--json")[1])
        self.assertEqual(runs[0]["kind"], "compare", "and it was recorded")

    def test_bad_input_files_are_exit_3(self):
        short = self.data / "short.gguf"
        short.write_bytes(b"GGUF\x03\x00")
        self.assertEqual(run("inspect", str(short))[0], cli.EXIT_INPUT)

    def test_a_byte_order_mark_does_not_break_a_conversation_file(self):
        conv = self.data / "c.json"
        conv.write_text('\ufeff[{"role": "user", "content": "hi"}]', encoding="utf-8")
        code, out, err = run("template", "chatml", "--conversation", str(conv), "--json")
        self.assertEqual(code, 0, err)
        self.assertIn("hi", json.loads(out)["render"])

    @needs_llama
    def test_stdin_is_utf8_whatever_the_console_says(self):
        import subprocess
        import sys
        from pathlib import Path
        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        r = subprocess.run([sys.executable, "-m", "athanor", "tokenize", str(VOCABS["llama-spm"]),
                            "--json", "--no-record"],
                           input="\ufeffВстреча перенесена на четверг".encode("utf-8"),
                           cwd=Path(__file__).resolve().parents[1], capture_output=True, env=env,
                           timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))
        m = json.loads(r.stdout.decode("utf-8"))["models"][0]
        self.assertEqual(m["words"], 4)
        self.assertIn("CYRILLIC", m["by_script"])

    def test_the_run_is_logged(self):
        from athanor import log
        run("inspect", str(self.data / "missing.gguf"))
        events = [e["event"] for e in log.read_events()]
        self.assertIn("cli-start", events)
        self.assertIn("cli-input-error", events)
        end = [e for e in log.read_events() if e["event"] == "cli-end"][-1]
        self.assertEqual(end["exit_code"], cli.EXIT_INPUT)

    def test_a_missing_llama_is_exit_4(self):
        with mock.patch.dict("sys.modules", {"llama_cpp": None}):
            code, _, err = run("tokenize", str(VOCABS["phi-3"]), "--text", "x")
        self.assertEqual(code, cli.EXIT_LLAMA, err)
        self.assertIn("llama-cpp-python", err)


if __name__ == "__main__":
    unittest.main()
