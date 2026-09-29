# Athanor

Take a language model apart and measure it — its file, its tokenizer, its
prompt format, its behaviour, its speed and its internals — on your own
hardware and your own material, with every result recorded so it can be
reproduced.

Athanor works on **GGUF** models through llama.cpp itself. It stands on its
own: a Python library, a command line (`python -m athanor`) and, soon, a set
of Qt widgets. It is built to be put inside other programs. The **Analyst
Toolkit (ATK)** is its first host, where it appears as the experimental Model
Lab workspace, but nothing in Athanor needs ATK.

*An athanor was the alchemist's slow furnace, built to hold one steady heat
for days so that long, patient work could go on inside it. Named by Bill and
Claude together, 2026-09-27.*

## Status — 0.1, Phase 1: the file

These work now, without loading any weights:

| tab | question |
|---|---|
| **Inspect** | What is this file, really, and will its tokenizer and template work together? |
| **Tokenize** | How much of *my* material fits, on which model? Which of my strings get shredded? |
| **Compare** | How do these two files differ? Are they the same family? |
| **Template** | What does the model actually see, under any of llama.cpp's 55 chat formats or its own? |

All four come with the **notebook** (every run recorded), a **log** of
everything that happened (llama.cpp's own messages included, and a crash
log if something native fails), and a JSON interface for other programs. The plan for everything else is
[ATHANOR_PLAN.md](ATHANOR_PLAN.md): the Waterfall, retrieval by thought, and
the Mad Science Wing.

```text
python -m athanor capabilities
python -m athanor inspect my-model.gguf
python -m athanor tokenize a.gguf b.gguf --file my_documents/ --context 32768
python -m athanor template chatml --model my-model.gguf
```

Five minutes: [docs/QUICKSTART.md](docs/QUICKSTART.md). Using it from your
own program: [docs/INTEGRATING.md](docs/INTEGRATING.md).

## The flagship, coming next: the Waterfall

An RF waterfall, pointed at a model. Time runs down the screen, the
vocabulary runs across it, and colour is probability. Every token the model
writes is one line: the whole distribution it was choosing from at that
moment. A reply becomes a recording you can scrub back and forth through,
like an I/Q capture. You can step through it, bookmark the moment the model
changed its mind, rewind to any token and branch down a road it didn't take,
or stack several answers to the same question and watch where they part.

## The rules it keeps

- **Measure, don't assume.** Every number says how it was made: MEASURED,
  DECLARED (what the file says about itself), ESTIMATE or EXPERIMENTAL.
- **Never modify a model file.** Changes are written to new files, and a file
  that exists is never overwritten.
- **One model on the card at a time.** Comparisons run in turn; a host's own
  model is put back afterwards.
- **Held to llama.cpp.** Tokenizers are llama.cpp's own, never
  re-implemented. The 55 chat templates match llama.cpp's C++ byte for byte.
- **Offline and local.** Nothing is downloaded, nothing is uploaded, no
  telemetry. Results stay on your disk.
- **Nothing ships with a model.** Bring your own GGUF files. The tests use
  three of llama.cpp's vocabulary-only files, which contain no weights.

## Layout

```
athanor/          the engine (no Qt, no ATK)
  api.py          the public Python API
  cli.py          python -m athanor …
  gguf/           GGUF reader and writer (every key, every array; byte-exact round trip)
  vocab.py        llama.cpp's tokenizer, vocab-only
  templates/      llama.cpp's 55 chat templates as Jinja, and rendering
  tabs/           Inspect, Tokenize, Compare, Template
  notebook.py     the record
  log.py          what happened, kept to learn from
  host.py         the port a host application fills
docs/             the integration guide, the quickstart, the format specs
tests/            unittest; run_tests.bat on Windows
```

## Tests

```text
python -m unittest discover -s tests -v          # or run_tests.bat on Windows
```

Tests that need llama.cpp are skipped, and say so, when llama-cpp-python is
not installed. Set `ATHANOR_TEST_MODEL` to one of your GGUFs to check a real
model as well.

## Licence

MIT. See [LICENSE](LICENSE). The three test vocabularies in
`tests/data/vocab` are llama.cpp's (MIT, © The ggml authors).
