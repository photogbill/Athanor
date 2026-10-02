# Athanor

Take a language model apart and measure it — its file, its tokenizer, its
prompt format, its behaviour, its speed and its internals — on your own
hardware and your own material, with every result recorded so it can be
reproduced.

Athanor works on **GGUF** models through llama.cpp itself. It stands on its
own: a Python library, a command line (`python -m athanor`) and a set of Qt
widgets. It is built to be put inside other programs. The **Analyst
Toolkit (ATK)** is its first host, where it appears as the experimental Model
Lab workspace, but nothing in Athanor needs ATK.

*An athanor was the alchemist's slow furnace, built to hold one steady heat
for days so that long, patient work could go on inside it. Named by Bill and
Claude together, 2026-09-27.*

## Status — 0.4: the file, the Waterfall, the Tap, and the lens

**The Waterfall** records every token a model writes, with everything it
was choosing between at that moment, and plays it back: the reply along
the top, time down the screen, the candidates across, the token actually
taken highlighted on every row, and the moments the model hesitated one
key away. Recorded live from a model your program already has loaded,
without changing a single token of the reply.

**The Tap** copies the model's own tensors out while llama.cpp computes
them, beside every token: which experts a mixture-of-experts model used
at every layer (the **expert map**, shown next to the Waterfall), each
layer's output, the final norm, the logits. It reads and never writes;
`athanor tap probe` checks, on your machine and your card, that it
changes nothing and what it costs.

**The lens** reads a recording's residual stream layer by layer: each
layer's output through the model's own final norm and output matrix —
what the model would say if it stopped there (the logit lens). From it,
every token's **decision depth**: the first layer from which the model's
eventual answer stays its answer all the way up. It checks itself against
the recording's own logits at the top layer, and says MEASURED only when
they agree.

```text
python -m athanor record my-model.gguf --prompt "Describe the relay plan." --tap experts
python -m athanor record my-model.gguf --prompt "Describe the relay plan." --lens
python -m athanor lens <that recording>.athrec-meta --step 12
python -m athanor tap probe my-model.gguf
python -m athanor.gui
```

And, without loading any weights:

| tab | question |
|---|---|
| **Inspect** | What is this file, really, and will its tokenizer and template work together? |
| **Tokenize** | How much of *my* material fits, on which model? Which of my strings get shredded? |
| **Compare** | How do these two files differ? Are they the same family? |
| **Template** | What does the model actually see, under any of llama.cpp's 55 chat formats or its own? |

All four come with the **notebook** (every run recorded), a **log** of
everything that happened (llama.cpp's own messages included, and a crash
log if something native fails), and a JSON interface for other programs. The plan for everything else is
[ATHANOR_PLAN.md](ATHANOR_PLAN.md): the rest of the behaviour tabs,
retrieval by thought, and the Mad Science Wing.

```text
python -m athanor capabilities
python -m athanor inspect my-model.gguf
python -m athanor tokenize a.gguf b.gguf --file my_documents/ --context 32768
python -m athanor template chatml --model my-model.gguf
```

Five minutes: [docs/QUICKSTART.md](docs/QUICKSTART.md). Using it from your
own program: [docs/INTEGRATING.md](docs/INTEGRATING.md).

## Next: the picture, branching, and the calibrated lens

The Waterfall is an RF waterfall pointed at a model: time down the screen,
the candidates across, colour for probability, a reply you can scrub like
an I/Q capture. The lens gives it a third axis — depth — and the next
step is drawing it: the grid of layers × tokens in the player, a depth
trace under the Waterfall, and the column at the cursor. Then rewinding to
any token and branching down a road the model didn't take, stacking
several answers to the same question to watch where they part, and the
calibrated (tuned) lens so the early layers read as well as the late ones.

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
  telemetry. Results stay on your disk — beside the code when Athanor runs
  from a checkout, in the host's folder inside a host, where `ATHANOR_DATA`
  says otherwise; `athanor data` tells you which.
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
  waterfall/      the recorder, the recording format, reading it back
  tap/            the Tap: ggml read from Python, and the probe
  lens/           the lens: the unembedding (decoded once, cached), the pass, the track
  gui/            the Qt widgets: the player, the expert map (optional)
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
