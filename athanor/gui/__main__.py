"""``python -m athanor.gui [RECORDING | FOLDER]`` — the Waterfall, in a window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m athanor.gui",
                                description="Play Waterfall recordings.")
    p.add_argument("target", nargs="?",
                   help="a recording (.athrec-meta) or a folder of them (default: "
                        "<data>/recordings)")
    args = p.parse_args(argv)
    from PySide6.QtWidgets import QApplication
    from ..waterfall import default_folder
    from .browser import WaterfallWindow
    folder, path = default_folder(), None
    if args.target:
        t = Path(args.target)
        if t.is_dir():
            folder = t
        else:
            path, folder = t, t.parent
    app = QApplication.instance() or QApplication(sys.argv[:1])
    w = WaterfallWindow(folder, path)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
