"""The Tap — a model's insides, copied out token by token (spike S1).

    from athanor import tap
    tap.make_tappable(llm)                       # once, right after loading
    with tap.capture(llm, "experts") as t:       # or tap.RESIDUAL, "l_out-12", …
        llm.create_completion("The river", max_tokens=16)
    t.decodes[0]["tensors"]["ffn_moe_topk-0"]    # the experts layer 0 chose

With the Waterfall, the Tap records beside every token:
``attach(llm, tap="experts")`` — see docs/INTEGRATING.md.

Read-only by construction: it copies what llama.cpp computes and never
writes into it.
"""

from .core import (EXPERTS, LOGITS, PRESETS, RESIDUAL, Dispatcher, Tap, capture, check_tap,
                   is_tappable, make_plain, make_tappable, model_facts, patterns, split_name,
                   tensor_names, token_axis, watching)
from .ggml import TapUnavailable

__all__ = ["Tap", "Dispatcher", "TapUnavailable", "EXPERTS", "RESIDUAL", "LOGITS", "PRESETS",
           "check_tap", "is_tappable", "make_tappable", "make_plain", "watching", "capture",
           "tensor_names", "model_facts", "split_name", "token_axis", "patterns"]
