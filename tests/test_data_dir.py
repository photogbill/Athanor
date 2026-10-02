"""Where Athanor writes when nobody told it: beside its code, never in a user
profile it was not pointed at (2026-10-02)."""

import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from athanor import cli, host


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class DataFolder(unittest.TestCase):

    def test_athanor_data_comes_first(self):
        with mock.patch.dict(os.environ, {"ATHANOR_DATA": "/some/where"}):
            path, how = host.data_dir_choice()
        self.assertEqual((path, how), (Path("/some/where"), "ATHANOR_DATA"))

    def test_a_checkout_keeps_its_data_beside_its_code(self):
        env = {k: v for k, v in os.environ.items() if k != "ATHANOR_DATA"}
        with mock.patch.dict(os.environ, env, clear=True):
            checkout = host.source_checkout()
            self.assertIsNotNone(checkout, "the tests run from a checkout")
            self.assertTrue((checkout / "pyproject.toml").is_file())
            path, how = host.data_dir_choice()
            self.assertEqual(how, "checkout")
            self.assertEqual(path, checkout / "data")
            self.assertEqual(host.default_data_dir(), checkout / "data")
            self.assertEqual(host.NullHost().data_dir(), checkout / "data")

    def test_only_an_installed_package_falls_back_to_the_user_profile(self):
        env = {k: v for k, v in os.environ.items() if k != "ATHANOR_DATA"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(host, "source_checkout", return_value=None):
            path, how = host.data_dir_choice()
            self.assertEqual(how, "per-user")
            self.assertTrue(str(path).lower().endswith("athanor"))

    def test_the_command_says_which_rule(self):
        code, out, err = run("data", "--json", "--data", str(Path("/tmp/athanor-x")))
        self.assertEqual(code, 0, err)
        r = json.loads(out)
        self.assertEqual(r["chosen_by"], "ATHANOR_DATA")
        self.assertEqual(Path(r["path"]), Path("/tmp/athanor-x"))
        self.assertTrue(r["folders"]["recordings"].endswith("recordings"))
        code, out, _err = run("data", "--data", str(Path("/tmp/athanor-x")))
        self.assertIn("because ATHANOR_DATA is set", out)


if __name__ == "__main__":
    unittest.main()
