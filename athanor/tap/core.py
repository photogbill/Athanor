"""The Tap: a model's intermediate tensors, copied out while llama.cpp computes them.

    from athanor import tap
    tap.make_tappable(llm)                 # once, right after loading
    with tap.capture(llm, tap.EXPERTS) as t:
        llm.create_chat_completion(messages, max_tokens=64)
    t.decodes[-1]["tensors"]["ffn_moe_topk-0"]     # the experts layer 0 chose

How: llama.cpp calls an evaluation callback for every tensor it computes
(``cb_eval`` in the context parameters — what ``examples/eval-callback``
prints with). It asks first ("do you want this one?"), computes up to the
tensors that are wanted, and hands each over before going on. The callback
is fixed when a context is created, so ``make_tappable`` rebuilds the
model's context once with Athanor's dispatcher in it — same parameters,
same weights (they are shared, not copied), an empty cache. The dispatcher
does nothing until a Tap is active on it; then it copies the wanted tensors
with ``ggml_backend_tensor_get``, from whichever device holds them.

What it costs: one Python call per graph node per forward pass while the
model is tappable (about 1,000 for a 32-layer model, a fraction of a
microsecond each when idle); while copying, the graph is computed in pieces
so each wanted tensor can be read. ``athanor tap probe`` measures both on
the machine it runs on.

What it changes: nothing it computes. The callback returns before
llama.cpp goes on and never writes; tapped logits are compared with
untapped ones in the tests and by ``probe``. Two honest caveats, both
measured by ``probe``: on a GPU, computing the graph in pieces can stop the
backend fusing some kernels, which may round differently in the last bits;
and every ask is a Python call, so a busy Python thread elsewhere in the
host can slow each token.

A failure inside the Tap stops the Tap — never the generation.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager

import numpy as np

from .ggml import TapUnavailable, ggml

# ------------------------------------------------------------- presets
EXPERTS = ("ffn_moe_topk-*", "ffn_moe_weights-*", "ffn_moe_probs-*")
RESIDUAL = ("l_out-*",)
LOGITS = ("result_output",)
PRESETS = {"experts": EXPERTS, "residual": RESIDUAL, "logits": LOGITS}

# Which ggml axis runs over the tokens. Almost every tensor llama.cpp names
# is [features, tokens]; the MoE tensors that carry one row per chosen
# expert are [features, n_used, tokens].
_TOKENS_ON_AXIS_1 = {
    "ffn_moe_logits", "ffn_moe_logits_biased", "ffn_moe_probs", "ffn_moe_probs_biased",
    "ffn_moe_probs_masked", "ffn_moe_argsort", "ffn_moe_topk", "ffn_moe_group_topk",
    "ffn_moe_weights_norm", "ffn_moe_weights_sum", "ffn_moe_weights_sum_clamped",
    "ffn_moe_out"}
_LAYER = re.compile(r"^(.*)-(\d+)$")

#: kinds whose token axis is known from llama.cpp's graph builders — the
#: tie-breaker when a tensor's shape alone cannot say (a prompt as long as
#: the model has heads), and the rule when the batch is unknown
KNOWN_AXES = {"l_out", "result_norm", "result_output", "attn_norm", "ffn_norm", "ffn_inp",
              "ffn_out", "attn_out", "kqv_out", "embd", "inp_embd", "norm",
              "ffn_moe_logits", "ffn_moe_logits_biased", "ffn_moe_probs", "ffn_moe_probs_biased",
              "ffn_moe_probs_masked", "ffn_moe_argsort", "ffn_moe_topk", "ffn_moe_group_topk",
              "ffn_moe_weights_norm", "ffn_moe_weights_sum", "ffn_moe_weights_sum_clamped",
              "ffn_moe_out", "ffn_moe_weights", "ffn_moe_weights_softmax",
              "ffn_moe_weights_scaled", "ffn_moe_weighted", "ffn_moe_gate", "ffn_moe_up",
              "ffn_moe_down"}


def split_name(name: str) -> tuple:
    """('ffn_moe_topk', 3) for 'ffn_moe_topk-3'; (name, None) for a tensor
    that belongs to no layer ('result_output')."""
    m = _LAYER.match(name)
    return (m.group(1), int(m.group(2))) if m else (name, None)


def token_axis(name: str) -> int:
    kind = split_name(name)[0]
    if kind.startswith("ffn_moe_") and kind not in _TOKENS_ON_AXIS_1:
        return 2
    return 1


def _glob(pattern: str) -> str:
    """A glob as a regex in which ``*`` and ``?`` never cross a space, so
    ``l_out-*`` takes ``l_out-12`` but not ``l_out-12 (view)`` — llama.cpp
    names its reshapes, views and copies by adding a space and a note."""
    out = []
    for ch in pattern:
        out.append("[^ ]*" if ch == "*" else "[^ ]" if ch == "?" else re.escape(ch))
    return "".join(out) + r"\Z"


def patterns(streams) -> tuple:
    """A preset's name ('experts'), one pattern, or several → patterns."""
    if isinstance(streams, str):
        return PRESETS.get(streams, (streams,))
    out = []
    for s in streams:
        out.extend(PRESETS.get(s, (s,)))
    return tuple(out)


# ------------------------------------------------------------ a session
class Tap:
    """What to copy, and what was copied.

    ``streams``: glob patterns over llama.cpp's tensor names (``l_out-*``,
    ``ffn_moe_topk-12``) or a preset (``"experts"``, ``"residual"``,
    ``"logits"``). ``rows``: ``"outputs"`` keeps, for each forward pass, the
    rows that produce logits (one per token generated); ``"all"`` keeps every
    token's row (a prompt's worth — large). ``keep``: keep every forward
    pass in ``decodes`` (for scripts); otherwise only the latest is kept,
    which is all a recording needs. ``limit_bytes`` caps what ``keep``
    holds: the Tap stops copying past it and says so."""

    def __init__(self, streams=EXPERTS, *, rows: str = "outputs", keep: bool = False,
                 limit_bytes: int = 1 << 30, collect_names: bool = False):
        if rows not in ("outputs", "all"):
            raise ValueError("rows is 'outputs' or 'all'")
        self.streams = patterns(streams)
        self._rx = (re.compile("|".join(f"(?:{_glob(p)})" for p in self.streams))
                    if self.streams else None)
        self._memo: dict = {}
        self.rows = rows
        self.keep = keep
        self.limit_bytes = int(limit_bytes)
        self.collect_names = collect_names
        self.names: list = []            # every node name asked about, in order (collect_names)
        self.current: dict = {}          # name -> [rows, elements], the latest forward pass
        self.n_outputs = 1
        self.decodes: list = []          # keep=True: {"n_tokens", "outputs", "tensors"}
        self.n_decodes = 0
        self._batch = None               # (n_tokens, outputs) of a decode the Tap was told of
        self._pass_tokens = None         # n_tokens of the forward pass in ``current``
        self.n_ubatch = 0                # the context's micro-batch size (0: unknown)
        self._ubatch = 0                 # which micro-batch of the announced decode
        self._first_name = None          # the first node of the graph being computed
        self._taken: set = set()         # names copied from the graph being computed
        self._after_announced = False
        self.error: str | None = None
        self.notes: dict = {}            # message -> times seen
        self.bytes_kept = 0
        self.bytes_copied = 0
        self.copy_seconds = 0.0
        self.asks = 0
        self._g = ggml()

    # -- called by the dispatcher, on the thread running llama_decode ----
    #
    # One llama_decode can run its graph several times: a batch longer than
    # the context's n_ubatch is computed in micro-batches of n_ubatch
    # consecutive tokens, each a whole graph. A graph's first node is asked
    # about first, so the first name asked after an announcement marks where
    # each graph (micro-batch, or unannounced forward pass) begins.
    def wants(self, t) -> bool:
        self.asks += 1
        name = self._g.name(t)
        if self._first_name is None:
            self._first_name = name
            if self._batch is None and self._after_announced:
                self._start_unannounced()        # a pass nobody announced, after one we knew
        elif name == self._first_name:
            self._next_graph()
        w = self._memo.get(name)
        if w is None:
            w = self._memo[name] = bool(self._rx is not None and self._rx.match(name))
            if self.collect_names:
                self.names.append(name)
        return w and self.error is None

    def _start_unannounced(self) -> None:
        self._finish()
        self._after_announced = False
        self._pass_tokens = None
        self.n_outputs = 1

    def _next_graph(self) -> None:
        self._taken = set()
        if self._batch is not None:
            self._ubatch += 1                    # the next micro-batch of the same decode
        else:
            self._start_unannounced()            # the next unannounced forward pass

    def _rows(self, name: str, ne: tuple):
        """(axis, rows to copy) for this tensor in the current graph, or
        (None, why) when its token axis cannot be told."""
        b = self._batch
        if b is None:
            # nobody said what the batch was (a vision handler decoding by
            # itself): the last token's row — the only one with logits
            kind_axis = token_axis(name) if split_name(name)[0] in KNOWN_AXES else None
            nonsingle = [d for d in range(4) if ne[d] > 1]
            axis = kind_axis if kind_axis is not None else (nonsingle[-1] if nonsingle else 3)
            if axis == 0:
                axis = 3                         # one row: the whole tensor
            n = int(ne[axis])
            return axis, (range(n) if self.rows == "all" else ([n - 1] if n else []))
        n_tokens, outs = b
        n_ub = self.n_ubatch if self.n_ubatch and self.n_ubatch > 0 else n_tokens
        start = self._ubatch * n_ub
        stop = min(n_tokens, start + n_ub)
        T = stop - start
        if T <= 0:
            return None, "more graphs ran than this batch has micro-batches"
        local = [i - start for i in outs if start <= i < stop]
        O = len(local)
        want_all = self.rows == "all"
        if split_name(name)[0] in KNOWN_AXES:
            # llama.cpp's own layout: the token axis is known; it holds either
            # the micro-batch's tokens or (at the last layer) only its outputs
            axis = token_axis(name)
            n = int(ne[axis])
            if n == T:
                return axis, (range(T) if want_all else local)
            if n == O:
                return axis, range(O)
            return None, (f"{n} rows along axis {axis}, but the micro-batch had {T} tokens "
                          f"and {O} outputs")
        if T == 1:                               # one token: the whole tensor is its row
            axis = next((d for d in (3, 2, 1) if ne[d] == 1), None)
            if axis is None:
                return None, f"a one-token tensor with no axis of size 1 ({list(ne)})"
            return axis, ([0] if (want_all or O) else [])
        on_tokens = [d for d in (1, 2, 3) if ne[d] == T]
        if on_tokens:                            # the outermost: [features…, tokens]
            return on_tokens[-1], (range(T) if want_all else local)
        if O == 0:
            empty = next((d for d in (3, 2, 1) if ne[d] == 0), None)
            if empty is not None:                # nothing to copy from this micro-batch
                return empty, []
        if O == 1:                               # a tensor that kept only the output row
            axis = next((d for d in (3, 2, 1) if ne[d] == 1), None)
            if axis is not None:
                return axis, [0]
        on_outputs = [d for d in (1, 2, 3) if ne[d] == O] if O else []
        if on_outputs:
            return on_outputs[-1], range(O)
        return None, (f"no axis of {list(ne)} matches the micro-batch's {T} tokens or "
                      f"{O} outputs")

    def take(self, t) -> None:
        started = time.perf_counter()
        g = self._g
        name = g.name(t)
        if name in self._taken:
            self.note(f"{name}: llama.cpp names more than one tensor so; the first is kept")
            return
        _typ, ne, _nb = g.head(t)
        axis, sel = self._rows(name, tuple(int(x) for x in ne))
        if axis is None:
            self.note(f"{name}: {sel} — not copied")
            return
        try:
            arr = g.read(t, axis, sel)
        except ValueError as exc:
            self.note(f"{name}: {exc} — not copied")
            return
        self._taken.add(name)
        self.bytes_copied += arr.nbytes
        if self.keep:
            if self.bytes_kept + arr.nbytes > self.limit_bytes:
                self.fail(RuntimeError(f"the Tap reached its limit of {self.limit_bytes:,} bytes "
                                       f"after {len(self.decodes)} forward passes"))
                return
            self.bytes_kept += arr.nbytes
        prev = self.current.get(name)
        if prev is not None and self._batch is not None and self._ubatch > 0:
            if prev.shape[1:] != arr.shape[1:]:
                self.note(f"{name}: its rows changed width between micro-batches — "
                          "the later ones are not kept")
            else:
                self.current[name] = np.concatenate([prev, arr])
        else:
            self.current[name] = arr
        self.copy_seconds += time.perf_counter() - started

    # -- forward-pass boundaries ------------------------------------------
    def before_decode(self, n_tokens: int | None, outputs=None) -> None:
        """A forward pass is about to run: ``n_tokens`` tokens, of which
        ``outputs`` (batch indices) produce logits. None when unknown."""
        self._finish()
        self._pass_tokens = n_tokens
        self._first_name = None
        self._ubatch = 0
        self._taken = set()
        self._after_announced = False
        if n_tokens is None:
            self._batch = None
            self.n_outputs = 1
        else:
            outs = tuple(sorted(int(i) for i in (outputs or ())))
            self._batch = (int(n_tokens), outs)
            self.n_outputs = len(outs) if self.rows == "outputs" else int(n_tokens)

    def after_decode(self) -> None:
        self._batch = None
        self._first_name = None
        self._taken = set()
        self._after_announced = True

    def _finish(self) -> None:
        if self.current:
            self.n_decodes += 1
            if self.keep:
                self.decodes.append({"n_tokens": self._pass_tokens, "outputs": self.n_outputs,
                                     "tensors": self.current})
        self.current = {}
        self._taken = set()

    def close(self) -> None:
        self._finish()
        self._batch = None

    # -- reading ------------------------------------------------------------
    def output(self, ridx: int = -1) -> dict:
        """The latest forward pass's row for logits row ``ridx`` (negative:
        from the end, as llama.cpp counts them): name -> 1-D array."""
        o = int(ridx) if ridx >= 0 else self.n_outputs + int(ridx)
        return {name: arr[o] for name, arr in self.current.items() if 0 <= o < len(arr)}

    # -- trouble ------------------------------------------------------------
    def note(self, message: str) -> None:
        self.notes[message] = self.notes.get(message, 0) + 1

    def fail(self, exc: BaseException) -> None:
        if self.error is None:
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                from .. import log
                log.exception("tap-failed", exc)
            except Exception:
                pass

    def report(self) -> dict:
        return {"streams": list(self.streams), "rows": self.rows, "error": self.error,
                "notes": [f"{m} (×{n})" if n > 1 else m for m, n in self.notes.items()],
                "forward_passes": self.n_decodes, "asks": self.asks,
                "bytes_copied": self.bytes_copied,
                "copy_seconds": round(self.copy_seconds, 6)}


# ----------------------------------------------------------- dispatcher
class Dispatcher:
    """The one C callback a tappable context carries. Idle until a Tap is
    set on it; never raises into llama.cpp; always lets it go on."""

    def __init__(self):
        import llama_cpp as lc
        self.session: Tap | None = None
        self.cfunc = lc.ggml_backend_sched_eval_callback(self._call)

    def _call(self, t, ask, _user_data):
        s = self.session
        if s is None:
            return not ask
        try:
            if ask:
                return s.wants(t)
            s.take(t)
        except BaseException as exc:          # noqa: BLE001 — nothing may reach llama.cpp
            try:
                s.fail(exc)
            except BaseException:             # noqa: BLE001
                pass
            return not ask
        return True                          # False here would make llama.cpp stop computing


_REQUIRED = ("llama_init_from_model", "ggml_backend_sched_eval_callback", "llama_context_params")


def check_tap(llm=None) -> str | None:
    """None when the Tap will work (on ``llm``, if given); otherwise why not."""
    try:
        import llama_cpp as lc
        from llama_cpp import _internals
    except Exception as exc:
        return f"llama-cpp-python is not importable ({type(exc).__name__}: {exc})"
    missing = [s for s in _REQUIRED if getattr(lc, s, None) is None]
    if missing:
        return "this binding lacks " + ", ".join(missing)
    if "cb_eval" not in {f[0] for f in lc.llama_context_params._fields_}:
        return "llama_context_params has no cb_eval field in this binding"
    if not hasattr(_internals, "LlamaContext"):
        return "this binding has no _internals.LlamaContext"
    try:
        ggml()
    except TapUnavailable as exc:
        return str(exc)
    if llm is not None:
        for attr in ("_ctx", "_model", "context_params", "_stack", "reset", "sample"):
            if not hasattr(llm, attr):
                return f"this object has no {attr}; it is not a llama_cpp.Llama"
        if not getattr(llm._ctx, "ctx", None):
            return "this model has no live context (closed?)"
    return None


def is_tappable(llm) -> bool:
    d = vars(llm).get("_athanor_tap") if hasattr(llm, "__dict__") else None
    ctx = getattr(llm, "_ctx", None)
    return d is not None and getattr(ctx, "_athanor_dispatcher", None) is d


def dispatcher(llm) -> Dispatcher | None:
    return vars(llm).get("_athanor_tap") if is_tappable(llm) else None


def make_tappable(llm) -> Dispatcher:
    """Rebuild ``llm``'s context with the Tap's dispatcher in it (a no-op if
    it already has one). Call it while nothing is generating — right after
    loading is best. The cache starts empty: the next generation reads its
    whole prompt again. Weights are shared, not reloaded; the old context is
    freed before the new one is made, so memory never holds two."""
    if is_tappable(llm):
        return vars(llm)["_athanor_tap"]
    why = check_tap(llm)
    if why:
        raise TapUnavailable(why)
    d = Dispatcher()
    params = type(llm.context_params).from_buffer_copy(llm.context_params)
    params.cb_eval = d.cfunc
    params.cb_eval_user_data = None
    ctx = _replace_context(llm, params)
    ctx._athanor_dispatcher = d          # the C callback lives as long as the context
    llm._athanor_tap = d
    from .. import log
    log.event("tap-installed", model=str(getattr(llm, "model_path", "")))
    return d


def make_plain(llm) -> None:
    """Undo ``make_tappable``: a context without the dispatcher (the cache
    starts empty again)."""
    if not is_tappable(llm):
        vars(llm).pop("_athanor_tap", None)
        return
    if vars(llm)["_athanor_tap"].session is not None:
        raise TapUnavailable("the Tap is in use on this model")
    _replace_context(llm, llm.context_params)
    vars(llm).pop("_athanor_tap", None)


class _GoneContext:
    """Stands in for a context that could not be rebuilt: any use raises in
    Python instead of handing llama.cpp a freed pointer."""

    ctx = None

    def __init__(self, why: str):
        self._why = why

    def __getattr__(self, attr):
        raise TapUnavailable(f"this model has no context ({self._why}); reload the model")

    def close(self):
        pass


def _replace_context(llm, params):
    """Free ``llm``'s context, then make one with ``params``. The old one
    goes first so memory never holds two (on a full card, the only order
    that fits). If neither the new one nor a plain one can be made, the
    model is left with a context that raises on use — never a freed one."""
    from llama_cpp import _internals
    old = llm._ctx
    old.close()
    try:
        new = _internals.LlamaContext(model=llm._model, params=params,
                                      verbose=getattr(llm, "verbose", False))
    except Exception as exc:
        try:
            plain = _internals.LlamaContext(model=llm._model, params=llm.context_params,
                                            verbose=getattr(llm, "verbose", False))
        except Exception as exc2:
            why = f"its context could not be rebuilt: {exc2}"
            llm._ctx = _GoneContext(why)
            raise TapUnavailable(f"the model's {why}; reload the model") from exc
        _install(llm, plain)
        raise TapUnavailable(f"a context with the Tap could not be made ({exc}); the "
                             "model has a plain context again") from exc
    _install(llm, new)
    return new


def _install(llm, ctx) -> None:
    llm._ctx = ctx
    llm._stack.callback(ctx.close)
    adapter = getattr(llm, "_lora_adapter", None)
    if adapter is not None:
        import ctypes

        import llama_cpp as lc
        adapters = (lc.llama_adapter_lora_p_ctypes * 1)(adapter)
        scales = (ctypes.c_float * 1)(float(getattr(llm, "lora_scale", 1.0)))
        if lc.llama_set_adapters_lora(ctx.ctx, adapters, 1, scales):
            raise TapUnavailable("the LoRA adapter could not be applied to the new context")
    llm.reset()


# ------------------------------------------------------------- watching
@contextmanager
def watching(llm, tap: Tap):
    """Copy what ``tap`` wants from every forward pass ``llm`` runs inside
    the block. ``llm`` must be tappable (``make_tappable``)."""
    d = dispatcher(llm)
    if d is None:
        raise TapUnavailable("this model was loaded without the Tap: call "
                             "athanor.tap.make_tappable(llm) first (it rebuilds the "
                             "model's context once)")
    if d.session is not None:
        raise TapUnavailable("the Tap is already in use on this model")
    ctxobj = llm._ctx
    had_own = "decode" in vars(ctxobj)
    previous = vars(ctxobj).get("decode")
    original = ctxobj.decode

    def decode(batch, *args, **kwargs):
        try:
            b = batch.batch
            n = int(b.n_tokens)
            outs = [i for i in range(n) if b.logits[i]]
        except Exception:                      # noqa: BLE001 — an unknown batch: last rows
            tap.before_decode(None)
        else:
            tap.before_decode(n, outs)
        try:
            return original(batch, *args, **kwargs)
        finally:
            tap.after_decode()

    try:
        import llama_cpp as lc
        tap.n_ubatch = int(lc.llama_n_ubatch(ctxobj.ctx))
    except Exception:                          # noqa: BLE001 — then a decode is one graph
        tap.n_ubatch = 0
    ctxobj.decode = decode
    d.session = tap
    try:
        yield tap
    finally:
        d.session = None
        if had_own:
            ctxobj.decode = previous
        else:
            vars(ctxobj).pop("decode", None)
        tap.close()


@contextmanager
def capture(llm, streams=EXPERTS, *, rows: str = "outputs", limit_bytes: int = 1 << 30):
    """``watching`` with a Tap that keeps every forward pass (``decodes``)."""
    with watching(llm, Tap(streams, rows=rows, keep=True, limit_bytes=limit_bytes)) as t:
        yield t


def tensor_names(llm, tokens=None) -> list:
    """The name of every tensor llama.cpp computes in one forward pass of
    ``llm`` (in graph order, without repeats) — what can be tapped. Runs one
    forward pass over ``tokens`` (default: the model's BOS token) and
    empties the model's cache before and after."""
    t = Tap((), collect_names=True)
    if tokens is None:
        tokens = [int(llm.token_bos())] if int(llm.token_bos()) >= 0 else [0]
    llm.reset()
    try:
        with watching(llm, t):
            llm.eval(list(tokens))
    finally:
        llm.reset()
        try:
            llm._ctx.kv_cache_clear()
        except Exception:
            pass
    return t.names


def model_facts(llm) -> dict:
    """What the Tap's readers need to know about the model: architecture,
    layers, width, experts. From the GGUF's own metadata (DECLARED)."""
    md = getattr(llm, "metadata", None) or {}
    arch = md.get("general.architecture")

    def num(key):
        v = md.get(f"{arch}.{key}") if arch else None
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    out = {"arch": arch, "n_layer": num("block_count"), "n_embd": num("embedding_length"),
           "n_expert": num("expert_count"), "n_expert_used": num("expert_used_count")}
    try:
        out["n_vocab"] = int(llm.n_vocab())
    except Exception:
        pass
    return out


__all__ = ["Tap", "Dispatcher", "TapUnavailable", "EXPERTS", "RESIDUAL", "LOGITS", "PRESETS",
           "check_tap", "is_tappable", "make_tappable", "make_plain", "watching", "capture",
           "tensor_names", "model_facts", "split_name", "token_axis", "patterns"]
