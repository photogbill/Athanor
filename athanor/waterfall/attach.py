"""Recording a llama-cpp-python ``Llama`` while it generates.

    from athanor.waterfall import attach
    with attach(llm) as rec:
        out = llm.create_chat_completion(messages, max_tokens=512)
    path = rec.save("recordings", finish_reason=out["choices"][0]["finish_reason"])

How: every token llama-cpp-python generates is picked by ``Llama.sample``
(``Llama.generate`` calls it once per position). For the length of the
``with`` block this instance's ``sample`` is wrapped: the original runs
unchanged and picks the token, then the wrapper reads the logits that token
was picked from — ``llama_get_logits_ith``, the same buffer and index the
sampler read — and hands both to the Recorder. The sampler chain is not
touched (no custom sampler is inserted), so a recorded generation is the
same generation, token for token, as one that was not recorded.

What it cannot see: a runtime that samples without ``Llama.sample``. If a
generation ends with no steps while tokens were produced, the recording
says so (``error``) instead of pretending to be empty.

A failure inside the recorder stops the RECORDING, never the generation:
the host's reply is worth more than its recording.

With ``tap=`` (a preset such as ``"experts"``, or tensor-name patterns) the
Tap records beside every step as well — the rows of the model's own
tensors that produced that step's logits (``athanor.tap``). The model must
have been made tappable first (``athanor.tap.make_tappable``); if it was
not, the recording says so and goes on without. A Tap failure costs the
Tap, never the recording.
"""

from __future__ import annotations

import ctypes
import inspect
import time
from contextlib import ExitStack, contextmanager

import numpy as np

from . import fileformat as ff
from .recorder import Recorder

_REQUIRED = ("llama_get_logits_ith", "llama_model_get_vocab", "llama_token_to_piece",
             "llama_vocab_is_eog", "llama_vocab_is_control")


class RecorderUnavailable(RuntimeError):
    """This binding (or this object) cannot be recorded; the message says why."""


def check_binding(llm=None) -> str | None:
    """None when recording will work; otherwise the reason it will not."""
    try:
        import llama_cpp as lc
    except Exception as exc:
        return f"llama-cpp-python is not importable ({type(exc).__name__}: {exc})"
    missing = [s for s in _REQUIRED if not callable(getattr(lc, s, None))]
    if missing:
        return "this binding lacks " + ", ".join(missing)
    Llama = getattr(lc, "Llama", None)
    if Llama is None or not callable(getattr(Llama, "sample", None)):
        return "this binding has no Llama.sample"
    try:
        if "idx" not in inspect.signature(Llama.sample).parameters:
            return "Llama.sample takes no idx, so the logits it samples from cannot be located"
    except (TypeError, ValueError):
        return "Llama.sample's signature cannot be read"
    if llm is not None:
        ctx = getattr(getattr(llm, "_ctx", None), "ctx", None)
        if not ctx:
            return "this model has no live context (closed, or not a llama_cpp.Llama)"
        if getattr(llm, "model", None) is None:
            return "this model has no live model handle"
        for attr in ("n_tokens", "input_ids", "n_vocab", "sample"):
            if not hasattr(llm, attr):
                return f"this object has no {attr}; it is not a llama_cpp.Llama"
    return None


def _reads_images(llm) -> bool:
    """Does this Llama carry a vision chat handler (llava / mtmd)?"""
    h = getattr(llm, "chat_handler", None)
    return h is not None and any(hasattr(h, a) for a in ("clip_model_path", "mtmd_ctx"))


def _piece_fn(lc, vocab, n_vocab: int):
    def piece(tid: int) -> bytes:
        if not 0 <= tid < n_vocab:        # llama.cpp ends the process on a bad id
            return b""
        buf = ctypes.create_string_buffer(64)
        n = lc.llama_token_to_piece(vocab, tid, buf, 64, 0, True)
        if n < 0:
            buf = ctypes.create_string_buffer(-n)
            n = lc.llama_token_to_piece(vocab, tid, buf, -n, 0, True)
        return buf.raw[:max(n, 0)]
    return piece


def _tap_for(llm, rec: Recorder, tap, limit_bytes: int, stack: ExitStack):
    """The Tap session for this block (or None), and the track it feeds."""
    from ..tap import Tap, TapUnavailable, check_tap, is_tappable, model_facts, patterns, watching
    from ..tap.track import TapTrack
    streams = patterns(tap)
    track = rec.tap_track
    if track is None:
        facts = {}
        try:
            facts = model_facts(llm)
        except Exception:
            pass
        track = rec.tap_track = TapTrack(streams, limit_bytes=limit_bytes, facts=facts)
    if not is_tappable(llm):
        why = check_tap(llm) or ("the model was loaded without the Tap: call "
                                 "athanor.tap.make_tappable(llm) after loading")
        track.unavailable = why
        return None, track
    session = Tap(streams, rows="outputs")
    try:
        stack.enter_context(watching(llm, session))
    except TapUnavailable as exc:
        track.unavailable = str(exc)
        return None, track
    stack.callback(lambda: track.sessions.append(session.report()))
    return session, track


@contextmanager
def attach(llm, *, k: int = ff.DEFAULT_K, meta: dict | None = None,
           clock=time.perf_counter, recorder: Recorder | None = None, tap=None,
           tap_limit_bytes: int = 2 << 30):
    """Record every token ``llm`` samples inside the block; yields the
    Recorder. On leaving, the token texts are resolved (the model is still
    loaded) and the Recorder is detached from the model; ``llm`` is exactly
    as it was.

    ``recorder``: carry on an earlier recording — one reply generated in
    several parts (a reply continued after it hit its length limit). Each
    new part is marked with a ``segment`` annotation.

    ``tap``: also record the model's insides with the Tap — ``"experts"``
    (a mixture-of-experts model's routing), ``"residual"`` (each layer's
    output), or tensor-name patterns. ``tap_limit_bytes`` caps what the Tap
    holds in memory; past it, the Tap stops and the recording says where."""
    why = check_binding(llm)
    if why:
        raise RecorderUnavailable(why)
    import llama_cpp as lc
    vocab = lc.llama_model_get_vocab(llm.model)
    n_vocab = int(llm.n_vocab())
    piece = _piece_fn(lc, vocab, n_vocab)

    def is_eog(t):
        return bool(lc.llama_vocab_is_eog(vocab, t))

    def is_control(t):
        return bool(lc.llama_vocab_is_control(vocab, t))

    if recorder is None:
        rec = Recorder(n_vocab=n_vocab, k=k, piece=piece, is_eog=is_eog,
                       is_control=is_control, clock=clock)
        rec.meta["model"] = {"path": getattr(llm, "model_path", None)}
    else:
        rec = recorder
        if rec.n_vocab != n_vocab:
            raise RecorderUnavailable(f"the model changed during this recording (a vocabulary "
                                      f"of {rec.n_vocab} tokens, now {n_vocab})")
        rec._piece, rec._is_eog, rec._is_control = piece, is_eog, is_control
    if meta:
        rec.meta.update(meta)
    new_part = {"first": rec.n_steps > 0}
    ctx = llm._ctx.ctx
    had_own = "sample" in vars(llm)
    previous = vars(llm).get("sample")
    original = llm.sample
    sig = inspect.signature(original)
    stack = ExitStack()                   # the Tap, if any: closed on every way out
    session = track = None

    def sample(*args, **kwargs):
        n_before = llm.n_tokens
        token = original(*args, **kwargs)
        if rec.error is not None:
            return token
        try:
            t = clock()
            if "idx" in kwargs or not args:
                idx = kwargs.get("idx")
            else:
                idx = sig.bind_partial(*args, **kwargs).arguments.get("idx")
            if rec.n_steps == 0 and rec.prompt_ids is None:
                if _reads_images(llm):
                    # a vision handler advances n_tokens over the image
                    # positions without writing input_ids there, so the ids
                    # below it are partly stale: say so rather than record them
                    rec.begin([])
                    rec.meta["prompt_tokens"] = int(n_before)
                    rec.meta.setdefault("prompt_note", (
                        "this model reads images: the prompt's token ids are not "
                        "recorded, because the binding does not keep them for "
                        "the image positions"))
                else:
                    rec.begin(llm.input_ids[:n_before].tolist())
            elif new_part["first"]:
                rec.mark("segment", prompt_tokens=int(n_before))
            new_part["first"] = False
            ridx = idx - n_before if idx is not None else -1
            ptr = lc.llama_get_logits_ith(ctx, ridx)
            if not ptr:
                raise RuntimeError(f"llama.cpp returned no logits for output {ridx}")
            logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,))
            rec.step(logits, int(token), t=t)
        except Exception as exc:          # the recording stops; the reply does not
            rec.fail(exc)
            return token
        if track is not None:
            try:
                if session is not None and session.error:
                    track.stop(session.error)
                track.step(rec.n_steps - 1, session.output(ridx) if session is not None else None)
            except Exception as exc:      # the Tap stops; the recording does not
                track.fail(exc)
        return token

    try:
        if tap is not None:
            try:
                session, track = _tap_for(llm, rec, tap, tap_limit_bytes, stack)
            except Exception as exc:      # the Tap is off; the recording goes on
                stack.close()
                session = None
                track = rec.tap_track
                if track is not None:
                    track.fail(exc)
        llm.sample = sample
        with stack:
            yield rec
    finally:
        stack.close()
        if had_own:
            llm.sample = previous
        else:
            vars(llm).pop("sample", None)
        rec.resolve()
        rec.detach()
