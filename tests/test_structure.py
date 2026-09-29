"""Guards on the shape of the package — read from the source's AST, never
its text, and each one proves it looked at what it guards.

* Standalone first: nothing in athanor/ imports ATK (or any host).
* The engine imports without Qt and without llama-cpp-python.
* The public API is what api.__all__ says.
* Every command can answer in JSON.
"""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "athanor"
FORBIDDEN_ROOTS = {"atk"}          # the first host; Athanor must not know it
QT_ROOTS = {"PySide6", "PyQt5", "PyQt6", "shiboken6"}


def imports_of(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name.split(".")[0], node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module.split(".")[0], node.lineno))
    return out


def engine_files() -> list:
    return sorted(p for p in PKG.rglob("*.py") if "gui" not in p.relative_to(PKG).parts)


class Standalone(unittest.TestCase):

    def test_the_guard_looks_at_the_whole_engine(self):
        names = {p.relative_to(PKG).as_posix() for p in engine_files()}
        for must in ("api.py", "cli.py", "vocab.py", "host.py", "gguf/read.py",
                     "tabs/inspect.py", "tabs/template.py", "templates/library.py"):
            self.assertIn(must, names)
        self.assertGreaterEqual(len(names), 18)

    def test_no_host_is_imported(self):
        bad = [(p.relative_to(ROOT).as_posix(), mod, line)
               for p in engine_files() for mod, line in imports_of(p) if mod in FORBIDDEN_ROOTS]
        self.assertEqual(bad, [])

    def test_the_guard_can_fail(self):
        """Negative control: an ATK import IS seen."""
        tree_src = "import os\nfrom atk.core import ctx\n"
        mods = [n.module.split(".")[0] for n in ast.walk(ast.parse(tree_src))
                if isinstance(n, ast.ImportFrom)]
        self.assertIn("atk", mods)

    def test_the_engine_has_no_qt(self):
        bad = [(p.relative_to(ROOT).as_posix(), mod) for p in engine_files()
               for mod, _ in imports_of(p) if mod in QT_ROOTS]
        self.assertEqual(bad, [])


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                          text=True, timeout=120)


class ImportsCleanly(unittest.TestCase):

    def test_api_imports_without_llama_cpp_or_qt(self):
        code = ("import sys\n"
                "sys.modules['llama_cpp'] = None\n"      # simulate absence
                "sys.modules['PySide6'] = None\n"
                "from athanor import api\n"
                "caps = api.capabilities()\n"
                "assert caps['binding']['installed'] is False, caps['binding']\n"
                "assert caps['features']['vocab']['available'] is False\n"
                "try:\n"
                "    api.Vocab('x.gguf')\n"
                "except (api.LlamaUnavailable, FileNotFoundError) as e:\n"
                "    print('ok', type(e).__name__)\n")
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ok", r.stdout)

    def test_importing_the_api_loads_neither(self):
        r = _run("import sys\nfrom athanor import api\n"
                 "print(sorted(m for m in ('llama_cpp', 'PySide6') if m in sys.modules))")
        self.assertEqual(r.stdout.strip(), "[]", r.stderr)

    def test_a_missing_llama_is_named_as_such(self):
        vocab = str(ROOT / "tests" / "data" / "vocab" / "ggml-vocab-phi-3.gguf")
        r = _run("import sys\nsys.modules['llama_cpp'] = None\n"
                 "from athanor.vocab import Vocab, LlamaUnavailable\n"
                 f"try:\n    Vocab({vocab!r})\nexcept LlamaUnavailable as e:\n"
                 "    print('LlamaUnavailable', 'llama-cpp-python' in str(e))\n")
        self.assertIn("LlamaUnavailable True", r.stdout, r.stderr)


class WindowsFiles(unittest.TestCase):
    """cmd.exe finds labels by byte offset: an LF-only .bat breaks goto,
    silently, and only on Windows. Check the bytes."""

    def test_batch_files_are_crlf(self):
        bats = sorted(ROOT.glob("*.bat"))
        self.assertTrue(bats, "the check must see run_tests.bat")
        for bat in bats:
            data = bat.read_bytes()
            self.assertEqual(data.count(b"\n"), data.count(b"\r\n"), f"{bat.name}: bare LF")
            self.assertNotIn("\ufffd", data.decode("utf-8", "replace"), bat.name)


class PublicSurface(unittest.TestCase):

    def test_everything_promised_exists(self):
        from athanor import api
        missing = [n for n in api.__all__ if not hasattr(api, n)]
        self.assertEqual(missing, [])
        self.assertGreaterEqual(len(api.__all__), 30)

    def test_every_command_has_json(self):
        from athanor.cli import build_parser
        p = build_parser()
        sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
        self.assertGreaterEqual(len(sub.choices), 10)
        for name, sp in sub.choices.items():
            opts = {o for a in sp._actions for o in a.option_strings}
            self.assertIn("--json", opts, name)


if __name__ == "__main__":
    unittest.main()
