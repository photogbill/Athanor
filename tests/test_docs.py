"""The guide cannot drift from the code: every ```python block in docs/
runs here, and every `athanor <command>` the docs mention exists."""

import contextlib
import io
import re
import unittest
from pathlib import Path

from athanor import log
from athanor.host import reset_host
from fixtures import HAVE_LLAMA, TempDir, VOCABS

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "docs" / "INTEGRATING.md", ROOT / "docs" / "QUICKSTART.md", ROOT / "README.md"]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def blocks(path: Path) -> list:
    return BLOCK.findall(path.read_text(encoding="utf-8"))


class TheGuideRuns(unittest.TestCase):

    def test_there_are_examples_to_run(self):
        self.assertGreaterEqual(len(blocks(DOCS[0])), 10)

    def test_every_python_example(self):
        ran = 0
        for doc in DOCS:
            for i, code in enumerate(blocks(doc)):
                if code.lstrip().startswith("# needs: llama") and not HAVE_LLAMA:
                    continue
                with self.subTest(doc=doc.name, block=i), TempDir() as d:
                    ns = {"MODEL": str(VOCABS["phi-3"]), "MODEL_B": str(VOCABS["llama-spm"]),
                          "DATA_DIR": d, "__name__": "__docs__"}
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            exec(compile(code, f"{doc.name}[{i}]", "exec"), ns)
                    finally:
                        reset_host()
                        log.reset()
                    ran += 1
        self.assertGreater(ran, 5)

    def test_commands_mentioned_exist(self):
        from athanor.cli import build_parser
        p = build_parser()
        sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
        known = set(sub.choices)
        mentioned = set()
        for doc in DOCS:
            mentioned |= set(re.findall(r"(?:^|`)[ \t]*(?:python -m )?athanor ([a-z]+)",
                                        doc.read_text(encoding="utf-8"), re.M))
        self.assertTrue(mentioned, "the check must find commands")
        self.assertEqual(sorted(mentioned - known), [])


if __name__ == "__main__":
    unittest.main()
