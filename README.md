# Athanor

Take a language model apart and measure it — its file, its tokenizer, its
prompt format, its behaviour, its speed and its internals — on your own
hardware and your own material, with every result recorded so it can be
reproduced.

Athanor works on **GGUF** models through llama.cpp. It stands on its own: a
Python library, a command line (`python -m athanor`), and an optional set of
Qt widgets — and it is built to be put inside other programs. The
**Analyst Toolkit (ATK)** is its first host, where it appears as the
experimental Model Lab workspace, but nothing in Athanor needs ATK.

*An athanor was the alchemist's slow furnace, built to hold one steady heat
for days so that long, patient work could go on inside it.*

**Status: design only.** The plan is [ATHANOR_PLAN.md](ATHANOR_PLAN.md) — the
tabs, the Mad Science Wing, the order they are built in, and the decisions
still open.

## The flagship: the Waterfall

An RF waterfall, pointed at a model. Time runs down the screen, the
vocabulary runs across it, colour is probability: every token the model
writes is one line — the whole distribution it was choosing from at that
moment. A reply becomes a recording you can scrub back and forth through,
like an I/Q capture: step through it, bookmark the moment it changed its
mind, rewind to any token and branch down a road it didn't take, or stack
several answers to the same question and watch where they part.

## Using it in your own program (planned)

Five ways in — take as much or as little as you need:

1. **Python API** — `athanor.api`: plain functions and dataclasses, no Qt,
   semantic versioning from 1.0.
2. **Command line with `--json`** — for tools written in any language.
3. **Open file formats** — recordings, the notebook and every result are
   specified in `docs/formats/`; you can read a recording without Athanor.
4. **The host port** — lend Athanor your GPU scheduling, model folders and
   data folder; the default needs nothing.
5. **Qt widgets** (optional, `pip install athanor[gui]`) — the Waterfall
   player and each tab as a widget for a PySide6 application.

The integration guide, `docs/INTEGRATING.md`, arrives with the first
release, and every code example in it runs in the test suite.

## The rules it keeps

- **Measure, don't assume** — every number says how it was made:
  MEASURED, ESTIMATE, or EXPERIMENTAL.
- **Never modify a model file** — changes are written to new files, with
  both hashes recorded.
- **One model on the card at a time** — comparisons run in turn and cache to
  disk; a host's own model is put back afterwards.
- **Held to llama.cpp** — wherever llama.cpp has a reference tool, Athanor's
  number is checked against it.
- **Offline and local** — nothing is downloaded, nothing is uploaded, no
  telemetry. Recordings and results stay on your disk.
- **Nothing ships with a model** — bring your own GGUF files.

## Layout (planned)

```
athanor/       the engine (no Qt, no ATK): GGUF reader/writer, vocab-only
               loads, its own generation loop, the recorder, the notebook,
               one module per tab, the public API, the CLI, the host port
athanor/gui/   optional Qt widgets and a standalone window
tests/         unittest, with references produced by llama.cpp itself
docs/          the integration guide, the format specs, per-tab notes
```

## Licence

MIT — see [LICENSE](LICENSE).
"# Athanor" 
