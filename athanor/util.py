"""Small shared helpers: JSON for numpy values, file identity, versions."""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Recursively turn results into plain JSON types."""
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {clean_str(str(k)): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, str):
        return clean_str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "backslashreplace")
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return str(obj)
    return obj


def clean_str(s: str) -> str:
    """A string that can always be written as UTF-8.

    The GGUF reader keeps bytes that are not UTF-8 as lone surrogates (so a
    file round-trips exactly). Written out, they become visible escapes —
    ``caf\\xe9`` — instead of an encoding error that would lose the result.
    """
    try:
        s.encode("utf-8")
        return s
    except UnicodeEncodeError:
        pass
    try:  # bytes the reader kept (U+DC80–U+DCFF) become \\xNN
        return s.encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")
    except UnicodeEncodeError:  # any other lone surrogate becomes \\udNNN
        return s.encode("utf-8", "backslashreplace").decode("utf-8")


def dumps(obj: Any, **kw) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False, **kw)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_identity(path: str | os.PathLike, header_sha256: str | None = None,
                  full_sha256: str | None = None) -> dict:
    """How a result names the file it was made from.

    ``header_sha256`` (cheap) identifies the metadata, tokenizer and tensor
    layout; ``full_sha256`` (slow on a large model) the whole file.
    """
    p = Path(path)
    st = p.stat()
    d = {"path": str(p.resolve()), "name": p.name, "size": st.st_size,
         "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if header_sha256:
        d["header_sha256"] = header_sha256
    if full_sha256:
        d["sha256"] = full_sha256
    return d


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def versions() -> dict:
    """What produced a result — recorded with every run."""
    from . import LLAMA_CPP_TAG, __version__
    return {
        "athanor": __version__,
        "llama_cpp_tag_tested": LLAMA_CPP_TAG,
        "llama_cpp_python": package_version("llama_cpp_python") or package_version("llama-cpp-python"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
    }


def human_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(f) < 1024 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{n} B"


def stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
