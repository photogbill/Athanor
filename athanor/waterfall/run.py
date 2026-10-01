"""Recording without a host: load a GGUF, generate once, save the recording.

This is what ``athanor record`` runs. A host that already has a model
loaded (ATK does) uses ``attach`` on its own ``Llama`` instead and never
loads a second copy.
"""

from __future__ import annotations

import time
from pathlib import Path

from .. import log
from ..vocab import ModelLoadError, _capturing, llama
from .attach import attach
from .fileformat import DEFAULT_K


def auto_gpu_layers() -> int:
    """Every layer on the GPU when this build can offload, else none."""
    try:
        return -1 if llama().llama_supports_gpu_offload() else 0
    except Exception:
        return 0


def load_model(path, *, n_ctx: int = 4096, gpu_layers: int | None = None, seed: int = 0):
    """A full ``llama_cpp.Llama`` (weights and all), with llama.cpp's log
    captured instead of printed. ``gpu_layers`` None: every layer on the GPU
    when this build can offload, else none."""
    lc = llama()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no model at {p}")
    if gpu_layers is None:
        gpu_layers = auto_gpu_layers()
    lines: list = []
    log.breadcrumb("record-load", path=str(p), n_ctx=n_ctx, gpu_layers=gpu_layers)
    started = time.perf_counter()
    try:
        with _capturing(lc, lines):
            llm = lc.Llama(model_path=str(p), n_ctx=int(n_ctx), n_gpu_layers=int(gpu_layers),
                           seed=int(seed), verbose=False)
    except Exception as exc:
        log.event("record-load-failed", path=str(p), error=f"{type(exc).__name__}: {exc}",
                  llama_cpp_log="".join(t for _l, t in lines))
        raise ModelLoadError(p, lines) from exc
    log.event("record-load", path=str(p), seconds=round(time.perf_counter() - started, 3),
              gpu_layers=gpu_layers)
    return llm


def record_once(model_path, *, messages: list | None = None, prompt: str | None = None,
                max_tokens: int = 256, temperature: float = 0.8, top_k: int | None = None,
                top_p: float | None = None, min_p: float | None = None,
                repeat_penalty: float | None = None, seed: int | None = None,
                n_ctx: int = 4096, gpu_layers: int | None = None, k: int = DEFAULT_K,
                folder=None, llm=None, tap=None) -> dict:
    """Generate once with the recorder attached and save the recording.

    ``messages`` goes through the model's chat template; ``prompt`` is sent
    as raw text. ``tap``: also record with the Tap (``"experts"``,
    ``"residual"``, patterns) — the model is made tappable first, which
    rebuilds a passed-in ``llm``'s context (its cache starts empty).
    Returns the saved recording's summary."""
    from ..host import get_host
    from . import default_folder
    from .reading import read
    if (messages is None) == (prompt is None):
        raise ValueError("give either messages (a chat) or prompt (raw text), not both")
    if llm is None and gpu_layers is None:
        gpu_layers = auto_gpu_layers()
    settings = {"max_tokens": max_tokens, "temperature": temperature, "top_k": top_k,
                "top_p": top_p, "min_p": min_p, "repeat_penalty": repeat_penalty,
                "seed": seed, "n_ctx": n_ctx, "gpu_layers": gpu_layers,
                "mode": "chat" if messages is not None else "raw", "tap": tap}
    sampling = {k2: v for k2, v in (("temperature", temperature), ("top_k", top_k),
                                    ("top_p", top_p), ("min_p", min_p),
                                    ("repeat_penalty", repeat_penalty)) if v is not None}
    if seed is not None:
        sampling["seed"] = int(seed)
    with get_host().borrow_gpu("athanor record"):
        own = llm is None
        if own:
            llm = load_model(model_path, n_ctx=n_ctx, gpu_layers=gpu_layers,
                             seed=seed if seed is not None else 0)
        try:
            if tap is not None:
                from ..tap import TapUnavailable, make_tappable
                try:
                    make_tappable(llm)
                except TapUnavailable as exc:      # recorded without; the meta says why
                    log.event("record-tap-unavailable", error=str(exc))
            with attach(llm, k=k, meta={"settings": settings, "messages": messages,
                                        "host": {"name": "athanor record"}}, tap=tap) as rec:
                log.breadcrumb("record-generate", model=str(model_path), max_tokens=max_tokens)
                if messages is not None:
                    out = llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                                     **sampling)
                    choice = out["choices"][0]
                    text = (choice.get("message") or {}).get("content") or ""
                else:
                    out = llm.create_completion(prompt=prompt, max_tokens=max_tokens, **sampling)
                    choice = out["choices"][0]
                    text = choice.get("text") or ""
            path = rec.save(folder or default_folder(), finish_reason=choice.get("finish_reason"),
                            host_reply=text)
        finally:
            if own:
                try:
                    llm.close()
                except Exception:
                    pass
    log.event("recorded-waterfall", path=str(path), steps=rec.n_steps, error=rec.error)
    return read(path).summary()
