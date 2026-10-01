"""What the installed llama.cpp binding can do — and therefore which tabs run.

Athanor does not install llama-cpp-python: a host (or the user) brings its
own build, CPU or CUDA, and builds differ. ``probe()`` looks at the one that
is installed and says, feature by feature, what is available and — when
something is not — why, and what needs it.
"""

from __future__ import annotations

import ctypes
import platform

from . import LLAMA_CPP_TAG, __version__
from .util import package_version

# feature -> (binding symbols it needs, what uses it)
FEATURES = {
    "vocab": (
        ("llama_model_load_from_file", "llama_model_get_vocab", "llama_tokenize",
         "llama_token_to_piece", "llama_detokenize", "llama_vocab_get_attr",
         "llama_vocab_get_text", "llama_vocab_n_tokens"),
        "Inspect's tokenizer checks, Tokenize, Compare's tokenizer diff, "
        "Template's tokenized render"),
    "chat_template_reference": (
        ("llama_chat_apply_template",),
        "cross-checking Template's renders against llama.cpp's own formatter"),
    "logits": (
        ("llama_batch_init", "llama_decode", "llama_get_logits_ith"),
        "Next Token, Surprise (Phase 2); the Waterfall's recorder reads them"),
    "state": (
        ("llama_state_seq_get_size", "llama_state_seq_get_data", "llama_state_seq_set_data"),
        "the Waterfall's branching (Phase 2)"),
    "control_vectors": (
        ("llama_set_adapter_cvec",),
        "the Wheel (M2), M21, spike S6 road 1"),
    "lora": (
        ("llama_adapter_lora_init", "llama_set_adapters_lora"),
        "the Adapters tab (Phase 3)"),
    "training": (
        ("llama_opt_init", "llama_opt_epoch"),
        "spike S6 road 2 (exact gradients)"),
}


def _model_param_fields(lc) -> set:
    try:
        return {f[0] for f in lc.llama_model_params._fields_}
    except Exception:
        return set()


def _context_param_fields(lc) -> set:
    try:
        return {f[0] for f in lc.llama_context_params._fields_}
    except Exception:
        return set()


def probe() -> dict:
    """A plain dict: versions, the binding, and each feature's availability."""
    out = {
        "athanor": __version__,
        "llama_cpp_tag_tested": LLAMA_CPP_TAG,
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
        "jinja2": package_version("jinja2"),
        "binding": {"installed": False, "version": None, "error": None},
        "gpu_offload": None,
        "system_info": None,
        "features": {},
    }
    try:
        import llama_cpp as lc
    except Exception as exc:
        out["binding"]["error"] = f"{type(exc).__name__}: {exc}"
        for name, (_syms, used_by) in FEATURES.items():
            out["features"][name] = {"available": False, "used_by": used_by,
                                     "why": "llama-cpp-python is not importable"}
        out["features"]["eval_callback"] = {
            "available": False, "used_by": "the Tap (spike S1)",
            "why": "llama-cpp-python is not importable"}
        out["features"]["recording"] = {
            "available": False, "used_by": "the Waterfall's recorder",
            "why": "llama-cpp-python is not importable"}
        return out

    out["binding"] = {"installed": True, "error": None,
                      "version": getattr(lc, "__version__", None) or package_version("llama_cpp_python")}
    try:
        out["gpu_offload"] = bool(lc.llama_supports_gpu_offload())
    except Exception:
        pass
    try:
        out["system_info"] = lc.llama_print_system_info().decode("utf-8", "replace").strip()
    except Exception:
        pass

    for name, (syms, used_by) in FEATURES.items():
        missing = [s for s in syms if not callable(getattr(lc, s, None))]
        feat = {"available": not missing, "used_by": used_by}
        if missing:
            feat["why"] = "this binding lacks " + ", ".join(missing)
        out["features"][name] = feat

    mp = _model_param_fields(lc)
    if "vocab_only" not in mp:
        out["features"]["vocab"] = {
            "available": False, "used_by": FEATURES["vocab"][1],
            "why": "llama_model_params has no vocab_only field in this binding"}

    from .waterfall.attach import check_binding
    why = check_binding()
    out["features"]["recording"] = {
        "available": why is None,
        "used_by": "the Waterfall's recorder (athanor.waterfall.attach, athanor record)"}
    if why:
        out["features"]["recording"]["why"] = why

    from .tap import check_tap
    why = check_tap()
    out["features"]["eval_callback"] = {
        "available": why is None,
        "used_by": ("the Tap (athanor.tap, athanor record --tap): the expert map, the logit "
                    "lens, attention maps — M1–M3, M9, M16, M20, M28–M34")}
    if why:
        out["features"]["eval_callback"]["why"] = why
    return out


def available(feature: str) -> bool:
    return bool(probe()["features"].get(feature, {}).get("available"))
