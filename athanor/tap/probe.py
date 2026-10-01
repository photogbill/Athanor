"""Spike S1, answered on the machine it runs on: does the Tap work here, what
does it cost, and does it change anything?

    python -m athanor tap probe D:\\models\\model.gguf

Loads the model once and makes two small contexts on it (the weights are
shared): a plain one, and one carrying the Tap's dispatcher. Generates the
same greedy continuation in each, several ways, and compares:

- **exactness** — the logits of every step, plain against tapped (idle, and
  copying each preset). 0 means bit-identical. On a GPU a small difference
  can come from the graph being computed in pieces (kernels that fuse
  across the piece boundary no longer fuse); the tokens are compared too.
- **the Tap reads the truth** — llama.cpp's own ``result_output`` tensor,
  copied by the Tap, against the logits llama.cpp returns.
- **the logit lens** — the last layer's output, put through the model's
  final norm and its output matrix by Athanor, against llama.cpp's logits
  (for the 16 most likely tokens: only those rows of the output matrix are
  decoded). This is the check the lens (M1, M30) stands on.
- **cost** — milliseconds per generated token: plain, tapped but idle, and
  copying each preset; bytes copied per token.
- **experts** — on a mixture-of-experts model, whether each layer's chosen
  experts are the router's top-scoring ones (true for Mixtral and Qwen;
  DeepSeek V3's selection bias makes them differ, by design).

Every number is MEASURED on this machine and this build.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .core import Dispatcher, Tap, model_facts, patterns

PROMPT = ("The history of radio begins with the discovery of electromagnetic waves. "
          "In the years that followed,")


class _Ctx:
    """A raw llama.cpp context on a loaded model, with our own greedy loop."""

    def __init__(self, llm, n_ctx: int, dispatcher: Dispatcher | None):
        import llama_cpp as lc
        self.lc = lc
        params = type(llm.context_params).from_buffer_copy(llm.context_params)
        params.n_ctx = n_ctx
        params.n_batch = max(int(params.n_batch), n_ctx)
        params.n_ubatch = min(int(params.n_ubatch) or 512, params.n_batch)
        if dispatcher is not None:
            params.cb_eval = dispatcher.cfunc
            params.cb_eval_user_data = None
        self.dispatcher = dispatcher
        self.ctx = lc.llama_init_from_model(llm.model, params)
        if not self.ctx:
            raise RuntimeError("llama.cpp could not make a context for the probe")
        self.n_vocab = int(llm.n_vocab())
        self.batch = lc.llama_batch_init(n_ctx, 0, 1)

    def close(self):
        if self.batch is not None:
            self.lc.llama_batch_free(self.batch)
            self.batch = None
        if self.ctx:
            self.lc.llama_free(self.ctx)
            self.ctx = None

    def _decode(self, tokens, pos0: int, tap: Tap | None):
        b = self.batch
        n = len(tokens)
        b.n_tokens = n
        for i, t in enumerate(tokens):
            b.token[i] = int(t)
            b.pos[i] = pos0 + i
            b.n_seq_id[i] = 1
            b.seq_id[i][0] = 0
            b.logits[i] = 1 if i == n - 1 else 0
        if tap is not None:
            tap.before_decode(n, [n - 1])
        try:
            rc = self.lc.llama_decode(self.ctx, b)
        finally:
            if tap is not None:
                tap.after_decode()
        if rc != 0:
            raise RuntimeError(f"llama_decode returned {rc}")
        ptr = self.lc.llama_get_logits_ith(self.ctx, -1)
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()

    def greedy(self, prompt_ids, n: int, tap: Tap | None = None, *, forced=None):
        """(tokens, logits per step, seconds per generated token, tap rows
        per step). ``forced``: feed these tokens instead of the argmax, so
        two runs see the same inputs even if their choices would differ."""
        self.lc.llama_memory_clear(self.lc.llama_get_memory(self.ctx), True)
        if self.dispatcher is not None:
            self.dispatcher.session = tap
        if tap is not None:
            tap.n_ubatch = int(self.lc.llama_n_ubatch(self.ctx))
        try:
            logits = [self._decode(prompt_ids, 0, tap)]
            rows = [tap.output(-1) if tap is not None else None]
            toks = []
            started = time.perf_counter()
            for i in range(n):
                t = int(np.argmax(logits[-1])) if forced is None else int(forced[i])
                toks.append(t)
                if i == n - 1:
                    break
                logits.append(self._decode([t], len(prompt_ids) + i, tap))
                rows.append(tap.output(-1) if tap is not None else None)
            per = (time.perf_counter() - started) / max(n - 1, 1)
        finally:
            if self.dispatcher is not None:
                self.dispatcher.session = None
            if tap is not None:
                tap.close()
        return toks, logits, per, rows


def _max_diff(a: list, b: list) -> float:
    return float(max((np.abs(x - y).max() for x, y in zip(a, b)), default=0.0))


def _lens_check(llm, rows: list, logits: list, facts: dict) -> dict:
    """The last layer's output → final norm → output rows, against llama.cpp's logits."""
    from ..gguf import read as read_gguf
    from ..gguf.dequant import NoDequantizer, dequantize_rows
    path = getattr(llm, "model_path", None)
    L = facts.get("n_layer")
    if not path or not L:
        return {"checked": False, "why": "the model's path or layer count is unknown"}
    last = f"l_out-{L - 1}"
    pairs = [(r, lg) for r, lg in zip(rows, logits) if r and last in r and "result_norm" in r]
    if not pairs:
        return {"checked": False, "why": f"the Tap did not see {last} and result_norm"}
    try:
        g = read_gguf(path)
    except Exception as exc:
        return {"checked": False, "why": f"the GGUF could not be read ({exc})"}
    arch = facts.get("arch")
    eps = g.get(f"{arch}.attention.layer_norm_rms_epsilon")
    norm_t = g.tensor("output_norm.weight")
    out_t = g.tensor("output.weight") or g.tensor("token_embd.weight")
    if eps is None or norm_t is None or out_t is None:
        return {"checked": False, "why": "not an RMS-norm model with output_norm.weight "
                                         "(this check covers llama-style models)"}
    try:
        from ..gguf.dequant import dequantize_row
        w = dequantize_row(g.tensor_bytes(norm_t), norm_t.ggml_type, int(norm_t.shape[0]))
        w = w.astype(np.float64)
    except Exception as exc:
        return {"checked": False, "why": f"output_norm.weight could not be read ({exc})"}
    norm_err, logit_err, scale = 0.0, 0.0, 0.0
    for r, lg in pairs[:4]:
        x = r[last].astype(np.float64)
        n = x / math.sqrt(float((x * x).mean()) + float(eps)) * w
        norm_err = max(norm_err, float(np.abs(n - r["result_norm"]).max()))
        top = np.argsort(-lg)[:16]
        try:
            W = dequantize_rows(g, out_t, [int(t) for t in top])
        except (NoDequantizer, Exception) as exc:        # noqa: BLE001
            return {"checked": False, "rms_norm_max_error": norm_err,
                    "why": f"the output matrix's rows could not be decoded ({exc})"}
        mine = W.astype(np.float64) @ r["result_norm"].astype(np.float64)
        logit_err = max(logit_err, float(np.abs(mine - lg[top]).max()))
        scale = max(scale, float(np.abs(lg[top]).max()))
    return {"checked": True, "rms_norm_max_error": norm_err,
            "logit_max_error_top16": logit_err, "logit_scale": scale,
            "reproduces": logit_err <= 1e-3 * max(scale, 1.0),
            "note": ("the last layer's output, normed with the file's output_norm and "
                     "multiplied by the output matrix in Athanor, against llama.cpp's logits")}


def probe(llm, *, tokens: int = 32, prompt: str = PROMPT) -> dict:
    """Run the S1 measurements on a loaded ``llama_cpp.Llama``. Its own
    context is not touched: two small contexts are made beside it."""
    from ..labels import MEASURED
    from ..util import versions
    from .core import check_tap
    why = check_tap(llm)
    if why:
        from .ggml import TapUnavailable
        raise TapUnavailable(why)
    import llama_cpp as lc
    facts = model_facts(llm)
    ids = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=False)
    n_ctx = int(math.ceil((len(ids) + tokens + 8) / 256) * 256)
    tokens = max(2, int(tokens))
    plain = tapped = None
    out = {"kind": "tap-probe", "label": MEASURED, "model": facts,
           "model_path": str(getattr(llm, "model_path", "")), "versions": versions(),
           "system_info": None, "gpu_offload": None, "prompt_tokens": len(ids),
           "generated_tokens": tokens}
    try:
        out["system_info"] = lc.llama_print_system_info().decode("utf-8", "replace").strip()
        out["gpu_offload"] = bool(lc.llama_supports_gpu_offload())
    except Exception:
        pass
    try:
        plain = _Ctx(llm, n_ctx, None)
        d = Dispatcher()
        tapped = _Ctx(llm, n_ctx, d)
        plain.greedy(ids, 2)                                # warm both up
        tapped.greedy(ids, 2)
        base_toks, base_logits, base_s, _ = plain.greedy(ids, tokens)
        idle_toks, idle_logits, idle_s, _ = tapped.greedy(ids, tokens)
        timing = {"plain": base_s, "tapped_idle": idle_s}
        exact = {"idle_vs_plain_max_abs": _max_diff(base_logits, idle_logits),
                 "idle_tokens_identical": idle_toks == base_toks}
        copied = {}
        runs = [("logits", ("result_output", "result_norm")), ("residual", patterns("residual"))]
        if facts.get("n_expert"):
            runs.append(("experts", patterns("experts")))
        L = facts.get("n_layer")
        lens_rows = lens_logits = None
        names_seen = None
        moe = None
        for label, streams in runs:
            want = tuple(streams) + ((f"l_out-{L - 1}", "result_norm") if label == "logits" and L
                                     else ())
            t = Tap(want, rows="outputs", collect_names=label == "logits")
            toks, logits, s, rows = tapped.greedy(ids, tokens, t, forced=base_toks)
            timing[label] = s
            exact[f"{label}_vs_plain_max_abs"] = _max_diff(base_logits, logits)
            copied[label] = {"bytes_per_token": t.bytes_copied / tokens,
                             "copy_ms_per_token": 1000 * t.copy_seconds / tokens,
                             "tensors_per_token": (sum(len(r) for r in rows if r) / tokens),
                             "error": t.error, "notes": t.report()["notes"]}
            if label == "logits":
                names_seen = t.names
                out["asks_per_token"] = t.asks / (tokens + 0.0)
                errs = [float(np.abs(r["result_output"] - lg).max())
                        for r, lg in zip(rows, logits) if r and "result_output" in r]
                out["result_output_vs_logits_max_abs"] = max(errs) if errs else None
                lens_rows, lens_logits = rows, logits
            if label == "experts":
                moe = _moe_check(rows, facts)
        out["ms_per_token"] = {k: round(1000 * v, 3) for k, v in timing.items()}
        out["overhead_percent"] = {k: round(100 * (v / base_s - 1), 1)
                                   for k, v in timing.items() if k != "plain" and base_s > 0}
        out["exactness"] = exact
        out["copied"] = copied
        out["lens"] = _lens_check(llm, lens_rows or [], lens_logits or [], facts)
        out["experts"] = moe
        if names_seen is not None:
            kinds: dict = {}
            from .core import split_name
            for nm in names_seen:
                if " (" in nm:
                    continue
                k, layer = split_name(nm)
                kinds.setdefault(k, set()).add(layer)
            out["graph"] = {"nodes": len(names_seen),
                            "kinds": {k: (len([x for x in v if x is not None]) or None)
                                      for k, v in sorted(kinds.items())}}
    finally:
        for c in (tapped, plain):
            if c is not None:
                c.close()
    out["verdict"] = _verdict(out)
    return out


def _moe_check(rows: list, facts: dict) -> dict:
    L = facts.get("n_layer") or 0
    checked = mismatched = 0
    used = facts.get("n_expert_used")
    for r in rows:
        if not r:
            continue
        for li in range(L):
            top, probs = r.get(f"ffn_moe_topk-{li}"), r.get(f"ffn_moe_probs-{li}")
            if top is None or probs is None:
                continue
            checked += 1
            best = np.argsort(-probs, kind="stable")[: len(top)]
            if set(int(x) for x in best) != set(int(x) for x in top):
                mismatched += 1
    return {"layer_steps_checked": checked, "chosen_not_router_top": mismatched,
            "n_expert": facts.get("n_expert"), "n_expert_used": used,
            "note": ("chosen experts that are not the router's top-scoring ones: 0 for "
                     "Mixtral / Qwen-style routers; DeepSeek V3's selection bias and grouped "
                     "routing make some differ by design")}


def _verdict(o: dict) -> str:
    ex = o.get("exactness") or {}
    parts = []
    if o.get("result_output_vs_logits_max_abs") == 0.0:
        parts.append("the Tap reads llama.cpp's own logits exactly")
    elif o.get("result_output_vs_logits_max_abs") is not None:
        parts.append(f"the Tap's copy of the logits differs by up to "
                     f"{o['result_output_vs_logits_max_abs']:.3g}")
    diffs = [v for k, v in ex.items() if k.endswith("_max_abs")]
    if diffs and max(diffs) == 0.0:
        parts.append("tapping changed nothing (bit-identical logits)")
    elif diffs:
        parts.append(f"tapping changed the logits by at most {max(diffs):.3g} "
                     f"(tokens identical: {ex.get('idle_tokens_identical')})")
    lens = o.get("lens") or {}
    if lens.get("checked"):
        parts.append("the logit lens reproduces the logits" if lens.get("reproduces")
                     else "the logit lens does NOT reproduce the logits here")
    oh = o.get("overhead_percent") or {}
    if oh:
        parts.append("cost: " + ", ".join(f"{k} {v:+.1f}%" for k, v in oh.items()))
    return "; ".join(parts)
