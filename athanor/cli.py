"""The command line: ``python -m athanor <command>``.

Every command prints a readable report, or with ``--json`` the same result
as JSON on stdout (and nothing else on stdout), so a program in any
language can use Athanor by running it.

Exit codes (stable, documented in docs/INTEGRATING.md):
  0  ran (findings, if any, are in the report)
  1  ran, and --strict was given and there were problem findings
  2  usage error
  3  an input file is missing or is not a GGUF Athanor can read
  4  llama.cpp (llama-cpp-python) is needed and not available
  5  anything else went wrong
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import LLAMA_CPP_TAG, __version__

EXIT_OK, EXIT_STRICT, EXIT_USAGE, EXIT_INPUT, EXIT_LLAMA, EXIT_ERROR = 0, 1, 2, 3, 4, 5


def _out(args, result, human):
    run = None
    if getattr(args, "record", False) and isinstance(result, dict) and result.get("kind"):
        from .api import record
        try:
            run = record(result, settings={k: v for k, v in vars(args).items()
                                           if k not in ("func", "json", "pretty", "record")
                                           and v is not None and not callable(v)})
        except Exception as exc:  # the notebook must never cost the user the result
            from . import log
            log.exception("record-failed", exc)
            print(f"(not recorded in the notebook: {type(exc).__name__}: {exc})", file=sys.stderr)
    if args.json:
        from .util import dumps
        sys.stdout.write(dumps(result, indent=2 if args.pretty else None) + "\n")
    else:
        human(result)
        if run:
            print(f"\nrecorded in the notebook as {run}")


def _strict(args, result) -> int:
    if getattr(args, "strict", False):
        fs = result.get("findings") or []
        if any(f.get("status") == "problem" for f in fs):
            return EXIT_STRICT
    return EXIT_OK


# ----------------------------------------------------------------- human
_MARK = {"problem": "✗", "warn": "!", "info": "·", "ok": "✓", "skipped": "–"}


def _print_findings(findings):
    for f in findings:
        print(f"  {_MARK.get(f['status'], '?')} [{f['status']:7s}] {f['message']}  ({f['label']})")


def _val(fig):
    if isinstance(fig, dict) and "value" in fig:
        v = fig["value"]
        unit = f" {fig['unit']}" if fig.get("unit") else ""
        return f"{v:,}{unit}" if isinstance(v, int) and not isinstance(v, bool) else f"{v}{unit}"
    return str(fig)


def _human_inspect(r):
    a = r["anatomy"]
    print(f"{r['file']['name']}  ({r['size_human']}, GGUF v{r['gguf']['version']}, "
          f"{r['gguf']['n_kv']} keys, {r['gguf']['n_tensors']} tensors)")
    for key in ("architecture", "name", "size_label", "file_type", "context_length",
                "embedding_length", "block_count", "head_count", "head_count_kv",
                "parameters", "bits_per_weight", "kv_cache_per_token"):
        fig = a.get(key)
        if fig and fig.get("value") not in (None, 0, ""):
            print(f"  {key:20s} {_val(fig):>24s}   {fig['label']}")
    d = r["tokenizer"]["declared"]
    print(f"  tokenizer            {d['model']['value']} / pre {d['pre']['value']}, "
          f"{d['n_tokens']['value']:,} tokens")
    m = r["tokenizer"]["measured"]
    if m:
        eog = ", ".join(x["text"] for x in m["end_of_generation"][:6])
        print(f"  llama.cpp sees       {m['type']['value']}, {m['n_tokens']['value']:,} tokens; "
              f"stops on: {eog}; vocab loaded in {m['load_seconds']['value']} s")
    print("\nfindings:")
    _print_findings(r["findings"])


def _human_metadata(rows):
    for row in rows:
        if "preview" in row:
            print(f"  {row['key']:48s} {row['type']:18s} [{row['length']}] {row['preview'][:6]}")
        else:
            print(f"  {row['key']:48s} {row['type']:18s} {row['value']!r}")


def _human_tokenize(r):
    for m in r["models"]:
        print(f"{m['model']['name']}  ({m['vocab_type']}, {m['n_vocab']:,} tokens)")
        print(f"  {m['n_tokens']['value']:,} tokens for {m['words']:,} words — "
              f"{m['tokens_per_word']} tokens/word, {m['chars_per_token']} chars/token")
        for s, v in sorted(m["by_script"].items()):
            print(f"    {s:10s} {v['tokens_per_word']} tokens/word ({v['words']} words)")
        if m.get("ruler"):
            ru = m["ruler"]
            print(f"  a {ru['context']:,}-token context holds about {ru['words']['value']:,} words "
                  f"of this material ({ru['words']['label']})")
        if m["tokens"]:
            shown = " | ".join(t["piece"] for t in m["tokens"][:40])
            print(f"  pieces: {shown}{' …' if m['tokens_shown'] > 40 else ''}")
        print()


def _human_splits(r):
    for row in r["strings"]:
        per = "  ".join(f"{x['model'][:28]}={x['n_tokens']}" for x in row["models"])
        print(f"  {row['category']:26s} {row['string'][:32]:34s} {per}")
    print(f"\n{r['note']}")


def _human_compare(r):
    t, m, ten, lin = r["tokenizer"], r["metadata"], r["tensors"], r["lineage"]
    print(f"A {r['a']['name']}\nB {r['b']['name']}\n")
    print(f"tokenizer: {'IDENTICAL' if t['identical'] else 'different'} — "
          f"{t['n_tokens']['a']:,} vs {t['n_tokens']['b']:,} tokens, overlap {t['overlap']['value']}, "
          f"{t['ids_that_differ']:,} shared ids hold different tokens")
    for k, v in t["special_ids_differ"].items():
        print(f"  special {k}: {v['a']} vs {v['b']}")
    print(f"metadata: {len(m['changed'])} keys differ, {len(m['only_in_a'])} only in A, "
          f"{len(m['only_in_b'])} only in B")
    for c in m["changed"][:12]:
        print(f"  {c['key']}: {c['a']!r} → {c['b']!r}")
    print(f"tensors: {ten['n_tensors']['a']} vs {ten['n_tensors']['b']}; "
          f"{ten['n_shape_differs']} shapes and {ten['n_type_differs']} types differ")
    print(f"lineage: {lin.get('verdict') or lin.get('why')}")
    if "cosine" in lin:
        print(f"  mean cosine {lin['cosine']['mean']:.4f} over {lin['rows_compared']} rows "
              f"of {lin['tensor']} (numbers MEASURED; the verdict is {lin['verdict_label']})")


def _human_overlap(r):
    names = r["models"]
    w = max(len(n) for n in names)
    for i, n in enumerate(names):
        cells = " ".join(("=" if r["identical_vocabulary"][i][j] else f"{r['overlap'][i][j]:.2f}")
                         .rjust(5) for j in range(len(names)))
        print(f"  {n.ljust(w)} {cells}")
    print(f"\n'=' identical vocabulary. {r['note']}")


def _human_template(r):
    if r.get("kind") == "template-compare":
        print(f"A {r['a']['template']['name']}  ({r['a'].get('n_tokens')} tokens)")
        print(f"B {r['b']['template']['name']}  ({r['b'].get('n_tokens')} tokens)")
        print("identical renders" if r["identical"] else f"{len(r['differences'])} differences:")
        for d in r["differences"][:20]:
            print(f"  {d['op']:8s} A {d['a']!r}  →  B {d['b']!r}")
        for side in ("a", "b"):
            print(f"\nfindings for {side.upper()}:")
            _print_findings(r[side]["findings"])
        return
    print(f"template: {r['template']['name']} ({r['template']['label']}); "
          f"llama.cpp calls it '{r['detected_format']}'")
    if r.get("render") is not None:
        print("\n--- render ---")
        print(r["render"])
        print("--- end ---")
    if r.get("segments"):
        print(f"\n{r['n_tokens']} tokens. markers and specials:")
        for s in r["segments"]:
            if s["kind"] != "text":
                print(f"  {s['kind']:12s} {s['text']!r} (id {s['id']})")
    print("\nfindings:")
    _print_findings(r["findings"])


def _human_library(rows):
    for t in rows:
        print(f"  {t['name']:24s} {t['label']:40s} {t['note'][:60]}")
    print(f"\n{len(rows)} templates, from llama.cpp {LLAMA_CPP_TAG}")


def _human_caps(r):
    b = r["binding"]
    print(f"athanor {r['athanor']} (tested against llama.cpp {r['llama_cpp_tag_tested']}), "
          f"python {r['python']}")
    print(f"llama-cpp-python: {b['version'] if b['installed'] else 'NOT AVAILABLE — ' + str(b['error'])}")
    if r.get("gpu_offload") is not None:
        print(f"GPU offload: {'yes' if r['gpu_offload'] else 'no (CPU build)'}")
    for name, f in r["features"].items():
        mark = "✓" if f["available"] else "✗"
        why = "" if f["available"] else f" — {f.get('why')}"
        print(f"  {mark} {name:24s} {f['used_by']}{why}")


def _human_roundtrip(r):
    print(f"{r['file']}: header {'IDENTICAL' if r['header_identical'] else 'DIFFERS'}"
          + (f", whole file {'IDENTICAL' if r['file_identical'] else 'DIFFERS'}"
             if r.get("file_identical") is not None else ""))


def _human_notebook(r):
    if isinstance(r, list):
        for run in r:
            print(f"  {run['id']}  {run['time']}  {run['kind']:16s} {run['question'][:60]}")
        print(f"\n{len(r)} runs")
    else:
        print(json.dumps(r, indent=2, ensure_ascii=False)[:20000])


# -------------------------------------------------------------- commands
def cmd_version(args):
    args.record = False
    _out(args, {"athanor": __version__, "llama_cpp_tag_tested": LLAMA_CPP_TAG},
         lambda r: print(f"athanor {r['athanor']} (llama.cpp {r['llama_cpp_tag_tested']})"))
    return EXIT_OK


def cmd_capabilities(args):
    args.record = False
    from .capabilities import probe
    _out(args, probe(), _human_caps)
    return EXIT_OK


def cmd_inspect(args):
    from .tabs.inspect import inspect, metadata
    if args.metadata:
        _out(args, metadata(args.model, max_items=args.preview), _human_metadata)
        return EXIT_OK
    r = inspect(args.model, run_llama=not args.no_llama, full_hash=args.full_hash,
                projector=args.projector)
    _out(args, r, _human_inspect)
    return _strict(args, r)


def cmd_tokenize(args):
    from .tabs.tokenize import compare_text, load_corpus
    corpus = None
    if args.text is not None:
        text = args.text
    elif args.file:
        corpus = load_corpus(args.file)
        text = corpus.pop("text")
        if corpus["truncated"]:
            print(f"note: read {corpus['bytes_read']:,} of {corpus['bytes_total']:,} bytes "
                  f"({corpus['files_read']} of {corpus['files']} files) — the limit is 64 MB",
                  file=sys.stderr)
    else:
        # stdin as UTF-8 whatever the console says (Windows pipes default to
        # the ANSI code page); a byte-order mark is dropped
        text = sys.stdin.buffer.read().decode("utf-8-sig", "replace")
    r = compare_text(args.models, text, context=args.context, show_tokens=args.show)
    if corpus is not None:
        r["input"] = dict(corpus, path=str(args.file))
    _out(args, r, _human_tokenize)
    return EXIT_OK


def cmd_splits(args):
    from .tabs.tokenize import worst_splits
    strings = None
    if args.strings_file:
        lines = Path(args.strings_file).read_text(encoding="utf-8-sig").splitlines()
        strings = [("custom", s) for s in lines if s.strip()]
    r = worst_splits(args.models, strings)
    _out(args, r, _human_splits)
    return EXIT_OK


def cmd_compare(args):
    from .tabs.compare import compare
    r = compare(args.a, args.b, lineage_rows=args.rows)
    _out(args, r, _human_compare)
    return EXIT_OK


def cmd_overlap(args):
    from .tabs.compare import overlap_matrix
    r = overlap_matrix(args.models)
    _out(args, r, _human_overlap)
    return EXIT_OK


def cmd_template(args):
    from .tabs import template as T
    if args.list:
        args.record = False
        _out(args, T.library(), _human_library)
        return EXIT_OK
    messages = None
    if args.conversation:
        messages = json.loads(Path(args.conversation).read_text(encoding="utf-8-sig"))
    spec = args.spec or ("gguf" if args.model else None)
    if spec is None:
        print("give a template (a name from --list, 'gguf', a file) and/or --model", file=sys.stderr)
        return EXIT_USAGE
    if args.vs:
        r = T.side_by_side(spec, args.vs, model=args.model, messages=messages,
                           add_generation_prompt=not args.no_gen)
    else:
        r = T.analyse(spec, model=args.model, messages=messages,
                      add_generation_prompt=not args.no_gen)
    _out(args, r, _human_template)
    return _strict(args, r if not args.vs else {"findings": r["a"]["findings"] + r["b"]["findings"]})


def cmd_roundtrip(args):
    from . import gguf
    g = gguf.read(args.model)
    rt = gguf.round_trip_check(g)
    r = {"kind": "roundtrip", "file": str(g.path), "header_identical": rt["identical"],
         "first_difference": rt["first_difference"], "header_bytes": rt["header_bytes"]}
    if args.whole:
        a, b = gguf.copy_stream_hash(g)
        r.update(file_identical=a == b, sha256=a)
    args.record = False
    _out(args, r, _human_roundtrip)
    return EXIT_OK if r["header_identical"] and r.get("file_identical", True) else EXIT_STRICT


def cmd_notebook(args):
    from .notebook import Notebook
    nb = Notebook(args.path) if args.path else Notebook()
    args.record = False
    if args.action == "list":
        _out(args, nb.runs(args.kind), _human_notebook)
    elif args.action == "show":
        run = nb.get(args.id)
        if run is None:
            print(f"no run {args.id}", file=sys.stderr)
            return EXIT_USAGE
        run = dict(run, notes=nb.notes(args.id))
        _out(args, run, _human_notebook)
    elif args.action == "note":
        nid = nb.note(args.id, " ".join(args.text))
        _out(args, {"note": nid, "run": args.id}, lambda r: print(f"note {r['note']} added"))
    elif args.action == "where":
        _out(args, {"path": str(nb.path)}, lambda r: print(r["path"]))
    return EXIT_OK


def _human_recording(r):
    m = r.get("model") or {}
    t = r.get("timing") or {}
    print(f"{r['path']}")
    print(f"  model: {m.get('name') or m.get('path') or '?'}")
    tps = t.get("tokens_per_second")
    ovh = t.get("overhead_ms_per_step")
    print(f"  {r['n_steps']} tokens, finish: {r.get('finish_reason')}"
          + (f", {tps:.1f} tokens/s" if tps else "")
          + (f", recorder {ovh:.2f} ms/token" if ovh is not None else ""))
    if r.get("mean_entropy_bits") is not None:
        print(f"  mean entropy {r['mean_entropy_bits']:.2f} bits; the model's favourite was "
              f"NOT chosen at {r['not_the_favourite']} of {r['n_steps']} tokens")
    if r.get("error"):
        print(f"  ! {r['error']}")
    tap = r.get("tap")
    if tap:
        if tap.get("recorded"):
            ovh = tap.get("overhead_ms_per_step")
            print(f"  the Tap recorded {', '.join(tap['recorded'])} beside every token"
                  + (f" ({ovh:.2f} ms/token)" if ovh is not None else ""))
        for key in ("unavailable", "error"):
            if tap.get(key):
                print(f"  ! the Tap: {tap[key]}")
        if tap.get("stopped_at") is not None:
            print(f"  ! the Tap stopped at step {tap['stopped_at']}")
    text = r.get("text") or ""
    print("\n" + (text if len(text) <= 2000 else text[:2000] + " …"))
    forks = r.get("forks")
    if forks:
        print("\nwhere the sampler did not take the favourite:")
        for f in forks:
            print(f"  step {f['step']:5d}  took {f['chosen']!r} ({100 * f['p']:.1f}%, rank "
                  f"{f['rank']})  over {f['favourite']!r} ({100 * f['favourite_p']:.1f}%)")
    step = r.get("step")
    if step:
        print(f"\nstep {step['step']}: chose {step['piece']!r} — rank {step['rank']}, "
              f"p {100 * (step['p'] or 0):.2f}%, entropy {step['entropy_bits']:.2f} bits")
        for c in r.get("candidates", []):
            mark = "▶" if c["chosen"] else " "
            print(f"   {mark} {100 * c['p']:7.3f}%  {c['piece']!r}  (id {c['id']})")


def _human_recordings(rows):
    for r in rows:
        err = f"  ! {r['error']}" if r.get("error") else ""
        print(f"  {r.get('started') or r.get('created') or '?':27s} {str(r.get('model')):40s} "
              f"{str(r.get('n_steps')):>6s} tokens{err}")
        print(f"      {r['path']}")
    print(f"\n{len(rows)} recordings")


def cmd_record(args):
    from .waterfall.run import record_once
    messages = prompt = None
    if args.raw:
        if args.messages or args.system:
            raise ValueError("--raw sends --prompt as it is; it takes no --messages or --system")
        if not args.prompt:
            raise ValueError("--raw needs --prompt")
        prompt = args.prompt
    elif args.messages:
        if args.prompt or args.system:
            raise ValueError("give --messages, or --prompt (with --system), not both")
        with open(args.messages, encoding="utf-8-sig") as f:
            messages = json.load(f)
        if not isinstance(messages, list) or not all(
                isinstance(m, dict) and "role" in m and "content" in m for m in messages):
            raise ValueError(f"{args.messages}: expected a JSON list of {{role, content}}")
    else:
        if not args.prompt:
            raise ValueError("record needs --prompt, or --messages FILE")
        messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
            {"role": "user", "content": args.prompt}]
    r = record_once(args.model, messages=messages, prompt=prompt, max_tokens=args.max_tokens,
                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                    min_p=args.min_p, repeat_penalty=args.repeat_penalty, seed=args.seed,
                    n_ctx=args.n_ctx, gpu_layers=args.gpu_layers, k=args.k, folder=args.out,
                    tap=args.tap or None)
    r["kind"] = "record"
    _out(args, r, _human_recording)
    return EXIT_OK


def _human_probe(r):
    m = r.get("model") or {}
    print(f"{r.get('model_path')}")
    print(f"  {m.get('arch')}, {m.get('n_layer')} layers, {m.get('n_embd')} wide"
          + (f", {m['n_expert']} experts ({m.get('n_expert_used')} used per token)"
             if m.get("n_expert") else "")
          + f"; GPU offload in this build: {r.get('gpu_offload')}")
    print(f"  {r.get('prompt_tokens')} prompt tokens, {r.get('generated_tokens')} generated, "
          f"{r.get('asks_per_token', 0):.0f} graph nodes asked about per token")
    print("\n  ms per generated token:")
    oh = r.get("overhead_percent") or {}
    for k, v in (r.get("ms_per_token") or {}).items():
        print(f"    {k:14s} {v:9.3f}" + (f"   {oh[k]:+.1f}%" if k in oh else ""))
    print("\n  exactness (largest logit difference against the plain context; 0 = identical):")
    for k, v in (r.get("exactness") or {}).items():
        print(f"    {k:32s} {v}")
    print(f"    result_output (the Tap's copy) vs logits: {r.get('result_output_vs_logits_max_abs')}")
    lens = r.get("lens") or {}
    if lens.get("checked"):
        print(f"\n  logit lens: {'reproduces' if lens.get('reproduces') else 'does NOT reproduce'} "
              f"llama.cpp's logits (largest error {lens['logit_max_error_top16']:.3g} on values "
              f"up to {lens['logit_scale']:.3g}; final norm error {lens['rms_norm_max_error']:.3g})")
    else:
        print(f"\n  logit lens: not checked — {lens.get('why')}")
    ex = r.get("experts")
    if ex:
        print(f"  experts: {ex['chosen_not_router_top']} of {ex['layer_steps_checked']} layer-steps "
              "chose experts other than the router's top-scoring ones")
    for label, c in (r.get("copied") or {}).items():
        extra = f"  ! {c['error']}" if c.get("error") else ""
        print(f"  copied for {label}: {c['tensors_per_token']:.0f} tensors, "
              f"{c['bytes_per_token'] / 1024:.1f} KB, {c['copy_ms_per_token']:.3f} ms per token{extra}")
        for n in c.get("notes") or []:
            print(f"    · {n}")
    print(f"\n  {r.get('verdict')}")


def _human_names(r):
    for k, layers in r["kinds"].items():
        print(f"  {k:32s} {('layers ' + layers) if layers else ''}")
    print(f"\n{r['nodes']} tensors in one forward pass of {r['model_path']}; "
          "tap any of them by name (e.g. --tap 'l_out-*')")


def _layer_ranges(layers: list) -> str:
    out, start, prev = [], None, None
    for x in sorted(layers):
        if start is None:
            start = prev = x
        elif x == prev + 1:
            prev = x
        else:
            out.append(f"{start}" if start == prev else f"{start}–{prev}")
            start = prev = x
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}–{prev}")
    return ", ".join(out)


def cmd_tap(args):
    from .tap import make_tappable, split_name, tensor_names
    from .tap.probe import probe
    from .waterfall.run import load_model
    llm = load_model(args.model, n_ctx=args.n_ctx, gpu_layers=args.gpu_layers)
    try:
        if args.action == "probe":
            from .util import file_identity
            r = probe(llm, tokens=args.tokens)
            try:
                r["file"] = file_identity(args.model)
            except OSError:
                pass
        else:
            args.record = False
            make_tappable(llm)
            names = tensor_names(llm)
            kinds: dict = {}
            unnamed = 0
            for n in names:
                base, _sp, note = n.partition(" (")
                if base.startswith("node_") and base[5:].isdigit():
                    unnamed += 1
                    continue
                k, layer = split_name(base)
                k = f"{k} ({note}" if note else k
                kinds.setdefault(k, [])
                if layer is not None:
                    kinds[k].append(layer)
            if unnamed:
                kinds[f"(unnamed nodes: {unnamed})"] = []
            r = {"kind": "tap-names", "model_path": str(args.model), "nodes": len(names),
                 "names": names,
                 "kinds": {k: _layer_ranges(v) for k, v in kinds.items()}}
    finally:
        try:
            llm.close()
        except Exception:
            pass
    _out(args, r, _human_probe if args.action == "probe" else _human_names)
    return EXIT_OK


def _forks(rec, limit: int) -> list:
    import math
    out = []
    for i in range(rec.n_steps):
        row = rec.steps[i]
        if int(row["rank"]) > 0:
            fav = int(row["ids"][0])
            out.append({"step": i, "chosen": rec.piece(int(row["chosen"])),
                        "p": math.exp(float(row["logprob"])), "rank": int(row["rank"]),
                        "favourite": rec.piece(fav),
                        "favourite_p": math.exp(float(row["logprobs"][0]))})
            if len(out) >= limit:
                break
    return out


def cmd_recording(args):
    from .waterfall import read
    args.record = False
    rec = read(args.path, verify=args.verify)
    r = rec.summary()
    r["forks"] = _forks(rec, args.forks)
    if args.step is not None:
        if not 0 <= args.step < rec.n_steps:
            raise ValueError(f"step {args.step} is outside 0–{rec.n_steps - 1}")
        r["step"] = rec.chosen(args.step)
        r["candidates"] = rec.candidates(args.step, args.top)
    _out(args, r, _human_recording)
    return EXIT_OK


def cmd_recordings(args):
    from .waterfall import default_folder, list_recordings
    args.record = False
    _out(args, list_recordings(args.folder or default_folder()), _human_recordings)
    return EXIT_OK


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    # NOTE: never set_defaults(record=...) on a subparser — the parent's
    # actions are shared objects, and argparse would change the default for
    # every command. Commands that never record set args.record themselves.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print the result as JSON only")
    common.add_argument("--pretty", action="store_true", help="indent the JSON")
    common.add_argument("--no-record", dest="record", action="store_false",
                        help="do not write this run to the notebook")
    common.add_argument("--data", help="data folder (default: ATHANOR_DATA or the per-user folder)")

    p = argparse.ArgumentParser(prog="athanor", description="Take a language model apart and "
                                "measure it. Every number says how it was made.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("version", parents=[common], help="versions")
    s.set_defaults(func=cmd_version)

    s = sub.add_parser("capabilities", parents=[common],
                       help="what the installed llama.cpp binding can do")
    s.set_defaults(func=cmd_capabilities)

    s = sub.add_parser("inspect", parents=[common], help="a file's anatomy and tokenizer health")
    s.add_argument("model")
    s.add_argument("--projector", help="an mmproj file to check against this model")
    s.add_argument("--no-llama", action="store_true", help="header only; do not run llama.cpp")
    s.add_argument("--full-hash", action="store_true", help="also hash the whole file (slow)")
    s.add_argument("--metadata", action="store_true", help="list every key instead")
    s.add_argument("--preview", type=int, default=16, help="array items shown with --metadata")
    s.add_argument("--strict", action="store_true", help="exit 1 if there are problem findings")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("tokenize", parents=[common], help="the context ruler: one text, many models")
    s.add_argument("models", nargs="+")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--text")
    g.add_argument("--file", help="a text file, or a folder of them")
    s.add_argument("--context", type=int, help="context length for the ruler, e.g. 32768")
    s.add_argument("--show", type=int, default=200, help="pieces to include per model")
    s.set_defaults(func=cmd_tokenize)

    s = sub.add_parser("splits", parents=[common], help="the analyst's strings that split worst")
    s.add_argument("models", nargs="+")
    s.add_argument("--strings-file", help="one string per line (default: the built-in set)")
    s.set_defaults(func=cmd_splits)

    s = sub.add_parser("compare", parents=[common], help="two files: tokenizer, metadata, tensors, lineage")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--rows", type=int, default=64, help="embedding rows compared for lineage")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("overlap", parents=[common], help="vocabulary overlap across a library")
    s.add_argument("models", nargs="+")
    s.set_defaults(func=cmd_overlap)

    s = sub.add_parser("template", parents=[common], help="the prompt microscope")
    s.add_argument("spec", nargs="?", help="a library name, 'gguf', a .jinja file, or Jinja text")
    s.add_argument("--model", help="tokenize the render with this model")
    s.add_argument("--vs", help="a second template to compare with")
    s.add_argument("--conversation", help="a JSON file: [{\"role\":…, \"content\":…}, …]")
    s.add_argument("--no-gen", action="store_true", help="no generation prompt at the end")
    s.add_argument("--list", action="store_true", help="list the 55 llama.cpp templates")
    s.add_argument("--strict", action="store_true", help="exit 1 if there are problem findings")
    s.set_defaults(func=cmd_template)

    s = sub.add_parser("roundtrip", parents=[common],
                       help="spike S5: does Athanor's writer reproduce this file?")
    s.add_argument("model")
    s.add_argument("--whole", action="store_true", help="hash the whole file too (reads it all)")
    s.set_defaults(func=cmd_roundtrip)

    s = sub.add_parser("notebook", parents=[common], help="the record of runs")
    s.add_argument("action", choices=["list", "show", "note", "where"])
    s.add_argument("id", nargs="?")
    s.add_argument("text", nargs="*")
    s.add_argument("--kind", help="only runs of this kind (list)")
    s.add_argument("--path", help="a notebook file other than the default")
    s.set_defaults(func=cmd_notebook)

    s = sub.add_parser("record", parents=[common],
                       help="the Waterfall: generate once and record every token's choices")
    s.add_argument("model")
    s.add_argument("--prompt", help="the user's message (or, with --raw, the whole prompt)")
    s.add_argument("--system", help="a system message before --prompt")
    s.add_argument("--messages", help="a JSON file: [{\"role\":…, \"content\":…}, …]")
    s.add_argument("--raw", action="store_true", help="send --prompt as raw text, no template")
    s.add_argument("--max-tokens", type=int, default=256)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int)
    s.add_argument("--top-p", type=float)
    s.add_argument("--min-p", type=float)
    s.add_argument("--repeat-penalty", type=float)
    s.add_argument("--seed", type=int)
    s.add_argument("--n-ctx", type=int, default=4096)
    s.add_argument("--gpu-layers", type=int,
                   help="layers on the GPU (default: all, if this build can offload)")
    s.add_argument("--k", type=int, default=256, help="candidates kept per token (default 256)")
    s.add_argument("--out", help="folder for the recording (default: <data>/recordings)")
    s.add_argument("--tap", action="append",
                   help="also record the model's insides: 'experts', 'residual', 'logits', or "
                        "tensor-name patterns such as 'l_out-*' (repeatable)")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("tap", parents=[common],
                       help="the Tap (spike S1): probe what it costs here, or list what it can read")
    s.add_argument("action", choices=["probe", "names"])
    s.add_argument("model")
    s.add_argument("--tokens", type=int, default=32, help="tokens generated per run (probe)")
    s.add_argument("--n-ctx", type=int, default=1024)
    s.add_argument("--gpu-layers", type=int,
                   help="layers on the GPU (default: all, if this build can offload)")
    s.set_defaults(func=cmd_tap)

    s = sub.add_parser("recording", parents=[common], help="read a Waterfall recording")
    s.add_argument("path", help="the .athrec-meta file (or its stem)")
    s.add_argument("--step", type=int, help="show the candidates at this step")
    s.add_argument("--top", type=int, default=20, help="candidates shown with --step")
    s.add_argument("--forks", type=int, default=40,
                   help="list up to N steps where the favourite was not taken")
    s.add_argument("--verify", action="store_true", help="check the data file's SHA-256")
    s.set_defaults(func=cmd_recording)

    s = sub.add_parser("recordings", parents=[common], help="list Waterfall recordings")
    s.add_argument("folder", nargs="?", help="default: <data>/recordings")
    s.set_defaults(func=cmd_recordings)
    return p


def _utf8_streams() -> None:
    """stdout and stderr in UTF-8, whatever the console or pipe says.

    On Windows a redirected stdout defaults to the ANSI code page (cp1252):
    the first '✓', Cyrillic name or Arabic token in a report would raise
    UnicodeEncodeError and kill the run — ATK lost its audio service to
    exactly that (one arrow, cp1252). JSON on stdout is UTF-8, always.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_TEXT_FLAGS = ("--text", "--prompt", "--system")


def redact_argv(argv: list) -> list:
    """The arguments as the log keeps them: text the user typed (``--text``,
    ``--prompt``, ``--system``, a notebook note) is replaced by its length and a hash — the log records
    WHAT was run, not the material. Paths are kept."""
    import hashlib

    def mask(v: str) -> str:
        h = hashlib.sha256(v.encode("utf-8", "surrogateescape")).hexdigest()[:12]
        return f"<text: {len(v)} chars, sha256 {h}>"

    out, skip = [], False
    note = len(argv) >= 2 and argv[0] == "notebook" and argv[1] == "note"
    for i, a in enumerate(argv):
        if skip:
            out.append(mask(a))
            skip = False
        elif a in _TEXT_FLAGS:
            out.append(a)
            skip = True
        elif a.split("=", 1)[0] in _TEXT_FLAGS and "=" in a:
            flag, value = a.split("=", 1)
            out.append(flag + "=" + mask(value))
        elif note and i >= 3 and not a.startswith("--"):
            out.append(mask(a))
        else:
            out.append(a)
    return out


def main(argv=None) -> int:
    import time

    from . import log
    _utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "data", None):
        os.environ["ATHANOR_DATA"] = args.data
    if args.command == "notebook" and args.action in ("show", "note") and not args.id:
        parser.error(f"notebook {args.action} needs a run id")
    from .gguf import GGUFError
    from .util import versions
    from .vocab import LlamaUnavailable, ModelLoadError
    from .tap import TapUnavailable
    from .waterfall.attach import RecorderUnavailable
    from .waterfall.fileformat import RecordingError
    if argv is None:           # a real process, not a call from Python
        log.enable_crash_log()
    started = time.perf_counter()
    try:
        cwd = os.getcwd()
    except OSError:
        cwd = None  # the working folder was deleted; nothing else needs it
    log.event("cli-start", argv=redact_argv(sys.argv[1:] if argv is None else list(argv)),
              cwd=cwd, versions=versions())
    code = EXIT_ERROR

    def fail(exc, rc, kind):
        log.exception(kind, exc, exit_code=rc)
        print(f"athanor: {exc}", file=sys.stderr)
        return rc

    try:
        code = args.func(args)
    except (FileNotFoundError, IsADirectoryError, GGUFError, RecordingError) as exc:
        code = fail(exc, EXIT_INPUT, "cli-input-error")
    except (LlamaUnavailable, RecorderUnavailable, TapUnavailable) as exc:
        code = fail(exc, EXIT_LLAMA, "cli-no-llama")
    except ModelLoadError as exc:
        code = fail(exc, EXIT_INPUT, "cli-model-load-error")
    except (UnicodeError, OSError) as exc:
        code = fail(exc, EXIT_ERROR, "cli-error")
    except ValueError as exc:
        code = fail(exc, EXIT_USAGE, "cli-usage-error")
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:  # the last resort: say what, not a traceback
        log.exception("cli-error", exc, exit_code=EXIT_ERROR)
        print(f"athanor: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("ATHANOR_DEBUG"):
            raise
        code = EXIT_ERROR
    finally:
        log.event("cli-end", exit_code=code, seconds=round(time.perf_counter() - started, 3))
    return code
