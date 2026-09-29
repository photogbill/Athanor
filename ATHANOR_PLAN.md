# Athanor — the plan

*Scoped 2026-09-27. **Phase 1 built 2026-09-28 (0.1.0)** — the engine,
Inspect, Tokenize, Compare, Template, the notebook, the CLI and the public
API; see CHANGELOG.md. Next: Phase 1.1 (numbers, slack, variants), the
Phase 1 widgets (athanor.gui), ATK's host side, then spike S1 and Phase 2.*
*Repository: `D:\Analyst_Toolkit\Athanor` → `github.com/photogbill/Athanor`;
package `athanor`. **Open source, MIT** (decided 2026-09-27). **Standalone
first**: any program can use it — ATK is one host among many, and hosts it
as the experimental **Model Lab** workspace.*

**The name.** An athanor was the alchemist's slow furnace, built to hold one
steady heat for days so that long, patient work could go on inside it. The
Lab's character is the same: overnight batteries, a hundred seeds, a model
taken apart one layer at a time with everything written down. The ⚗ on ATK's
rail is the alembic that sat on top of it. Named by Bill and Claude together,
2026-09-27.

Bill, 2026-09-26: *"I think I will want to add an experimental sidebar item,
with a series of tabs. At some point in the future, I want to be able to
test models with various tokenizers, not just the one that comes bundled in
the GGUF."* Then, 2026-09-27: *"if you can think of any additional Lab tabs,
things that may be useful. Please exercise creative control on this one"* —
and *"if you can think of any truly mad scientist things we can add in
there. I think it could be both fun and educational right? Stuff that others
aren't considering really."*

---

## 0. What the Lab is — and the one fact that shapes it

A place to take a model apart and measure it — its file, its tokenizer, its
prompt format, its behaviour, its speed, its internals — on Bill's own
hardware and Bill's own material, and to record every result so it can be
reproduced.

**The fact that shapes it:** a trained model's tokenizer cannot be swapped at
load. The embedding table is indexed by the model's own token ids; another
tokenizer produces ids that point at rows meaning something else, and the
output is noise. What *can* be changed at load is narrower — llama.cpp's
`--override-kv` (and the binding's `kv_overrides`) takes integers, floats,
booleans and strings up to 127 characters, and the vocabulary arrays are read
straight from the file (checked in llama.cpp b11093:
`llama_model_kv_override`, `llama-vocab.cpp`). So the Lab is built around
what is real:

- **Tokenizers are measured, compared and diagnosed** (Inspect, Tokenize,
  Compare), and their *metadata* is fixable at load (Override).
- **Changing the vocabulary itself is surgery on a copy**, and measured
  afterwards (Frankentokenizer, in the Mad Science Wing) — which is also the
  honest version of "various tokenizers": it shows how much a model loses,
  in numbers, instead of promising it loses nothing.
- **TokSuite** (arXiv 2512.20757) — fourteen models identical but for the
  tokenizer, plus a multilingual robustness benchmark (English, Chinese,
  Farsi, Italian, Turkish; Farsi is close kin to Dari) — is exactly the
  controlled version of Bill's question. Those are separate models; the Lab
  can load them side by side like any others, if their licences pass
  Bill's licence rule.

## 1. Principles

1. **Measure, don't assume.** Every tab answers a question with a number and
   says how the number was made. Four labels, on screen and in the record:
   **MEASURED**, **DECLARED** (what the file says about itself, unchecked —
   added while building, 2026-09-28), **ESTIMATE** (arithmetic, not
   observation), **EXPERIMENTAL** (research-grade: interesting, not
   evidence).
2. **Never modify a model file.** Anything that changes a model writes a NEW
   file, beside a record of both SHA-256s. The forensic habit, applied to
   models.
3. **One model on the card at a time.** 16 GB. A comparison loads its
   configurations in turn and caches what it needs (top-k log-probabilities,
   hidden states) to disk. Every load asks the HOST for the GPU and hands it
   back afterwards (§2a): inside ATK that is ATK's AI-queue ticket, with the
   operator's model put back exactly as escalate does
   (`second_opinion.escalate`); on its own, the null host simply loads.
4. **No weights when they are not needed; no memory bill that is not
   needed.** Tokenizer work loads the vocabulary only (`vocab_only=True`, in
   the pinned binding) — no VRAM, and (to be measured, S4) seconds per model.
   And **never `logits_all` over a long context**: the binding sizes its
   score buffer as context × vocabulary floats (read in ea3b56b's
   `llama.py`) — 32,768 × 131,072 × 4 bytes is 17 GB of RAM. Athanor
   evaluates in batches and reads the logits it needs, position by position
   (`llama_get_logits_ith`).
5. **Held to llama.cpp.** Wherever llama.cpp has a reference — its chat
   formatter, `llama-perplexity` (perplexity and `--kl-divergence`),
   `llama-bench`, `llama-tokenize`, `llama-quantize` — the Lab's number is
   checked against it once, in the tests, the way ATK's 55 prompt templates
   were held to `llm_chat_apply_template` byte for byte on 2026-09-26.
6. **The notebook records everything.** Question, files (hashes), settings,
   versions (Lab, llama.cpp tag, binding commit), result, notes. A result
   that cannot be reproduced is an anecdote.
7. **Offline, and local.** Nothing is downloaded by Athanor and nothing
   leaves the machine: no telemetry, no update check, no upload. Recordings,
   the notebook and caches stay on local disk, and the repo's `.gitignore`
   keeps them out of git. Anything external (TokSuite, reference corpora)
   arrives as files the operator supplies. **Nothing ships with a model**:
   Athanor works on the GGUF files the operator already has — any GGUF the
   pinned llama.cpp loads.
8. **Bill's licence rule applies to every dependency**: nothing that
   restricts commercial use — permissive and weak copyleft pass; GPL / AGPL
   / non-commercial do not. llama.cpp (MIT), gguf-py (MIT), numpy (BSD) pass.
9. **The boundary.** The Lab studies models; it does not remove their safety
   training. No refusal-direction ablation ("abliteration"), no jailbreak
   search. The steering and probing experiments below are about language,
   style, truthfulness and understanding.
   **Studying refusals is in scope** — Bill, 2026-09-27: *"nothing that
   removes it. Finding it sure, so we can see how it works"*, and *"if
   nobody researches this stuff … we will never learn from that data, both
   how to prevent and how to improve."* The existing instruments already
   show it: the Waterfall records the moment a reply tilts into a refusal
   and the roads not taken around it; the logit lens (M1) shows how deep in
   the stack it forms; attention maps (M16) show which words of the prompt
   it rests on; the multiverse (M15) shows how often it happens across
   seeds; Compare / Quantization show whether a quant, a template or an
   override changes it; the Report Card counts over-refusals on legitimate
   work. What Athanor does not produce is a refusal direction as a saved,
   reusable vector — that artifact is the ingredient removal needs.
10. **Standalone first.** Nothing in `athanor/` imports ATK — a test walks
   the package's AST and fails on any `atk` import. Everything ATK supplies
   (the GPU hand-off, model folders, a discussion's messages, a case's
   entities) reaches Athanor as plain data through the host port or a
   function argument, so any other program can supply the same thing.

## 2. Architecture

```
athanor/                   the engine — no Qt, no ATK
  api.py                   THE public Python API (§2a) — the only names
                           other programs are promised; everything else is
                           internal
  capabilities.py          probes the installed llama.cpp binding once:
                           which features it has (cb_eval, state save,
                           control vectors …) → which tabs can run
  templates/               Athanor's own prompt-template library (the 55
                           llama.cpp templates held byte for byte + the
                           binding's formats), ported from ATK's
                           prompt_templates.py under MIT
  gguf/        read.py     full GGUF reader: every key, every array (token
                           lists, merges, types), tensor infos — ATK's
                           gguf_meta keeps arrays <= 4096 on purpose, a
                           vocabulary is ~130k
               write.py    new files only (S5)
  vocab.py                 vocab-only loads; tokenize / detokenize / token
                           attributes through llama.cpp itself — never a
                           Python re-implementation of a tokenizer
  run.py                   one loaded configuration and ATHANOR'S OWN
                           generation loop: per-position logits in batches,
                           kv overrides, LoRA, the tap (S1); caches to disk
  record.py                the thought recorder (3.6): writes and reads
                           recordings; branches
  notebook.py              append-only runs + notes (JSON Lines)
  tabs/…                   one module per tab: pure functions over the above
  cli.py                   python -m athanor inspect|tokenize|compare|…
                           every command also takes --json
  host.py                  the port a host fills: borrow_gpu() (e.g. queue
                           ticket + unload + restore), vram_plan(),
                           model_dirs(), data_dir(), settings; NullHost is
                           the default and needs nothing
  gui/                     OPTIONAL (pip install athanor[gui], PySide6 —
                           LGPL, passes the licence rule): the tabs and the
                           Waterfall player as embeddable widgets, plus
                           python -m athanor.gui, a standalone window
tests/                     unittest; references from llama.cpp's own tools;
                           every example in docs/ runs as a test
docs/                      the integration guide and the format specs (§2a),
                           then per-tab notes as they are built
pyproject.toml             pip-installable; the GUI is an extra
LICENSE                    MIT
```

The pages live in Athanor (`athanor.gui`), so a program that is not ATK gets
the Waterfall too. ATK keeps a thin page and one adapter, as it does for the
Forensics Workshop: `atk/ui/lab_panel.py` places Athanor's widgets in ATK's
rail, `atk/core/lab_host.py` fills `host.py`, all through the airlock
(`atk/ui/subsystem.py`, key = the PACKAGE, `athanor`).

**Where results live:** in the host's data folder. On its own, the per-user
data folder (`%LOCALAPPDATA%\Athanor` on Windows, `~/.local/share/athanor`
on Linux), or wherever `--data` / `data_dir()` says; inside ATK,
`ATK\data\athanor\`. Caches and new model files are large (a logit cache of
a long corpus is gigabytes, a surgery copy of a 24B is 14 GB) and go to a
folder the operator chooses.

## 2a. Built to be embedded

Bill, 2026-09-27: *"make sure that we include a detailed guide and
incorporate whatever is needed so that other software will be able to
incorporate it into their own tool. If it requires ATK, then it defeats the
purpose."*

Five ways in, so a program can take as much or as little as it wants:

1. **The Python API** (`athanor.api`) — plain functions and dataclasses,
   no Qt, no global state: `inspect(path)`, `tokenize(paths, text)`,
   `compare(a, b)`, `render(template, messages)`, `record(model, messages,
   …)`, `replay(model, messages, reply)`, `open_recording(path)`, … Semantic
   versioning from 1.0: a name in `api` is removed or changed only in a
   major version, with a deprecation warning a minor version before.
2. **The command line, with `--json`** — every operation, machine-readable
   output and documented exit codes, so a tool in C#, Rust, Go or
   JavaScript uses Athanor by running it, with no Python bindings of its own.
3. **Open file formats, specified** — `docs/formats/`: the recording
   (`.athrec-meta` JSON + `.athrec-data` binary; SigMF-shaped, versioned,
   every field defined, with a reader in ~50 lines of plain Python and
   numpy in the guide as proof), the notebook (JSON Lines) and every
   result's JSON schema. Another program can read a recording without
   importing Athanor at all.
4. **The host port** (`athanor.host.Host`) — how an application lends
   Athanor its GPU, model folders and data folder. `NullHost` needs nothing.
   The guide shows a complete host in under 40 lines; ATK's `lab_host.py`
   is the full-size example.
5. **The widgets** (`athanor.gui`, optional) — the Waterfall player and each
   tab as a QWidget a PySide6 application drops into its own window.

**The guide** — `docs/INTEGRATING.md`, written with Phase 1 and kept whole
as the phases land: install (`pip install
git+https://github.com/photogbill/Athanor`, and the llama-cpp-python build it
needs, CPU or CUDA); the capability probe (which tabs a given binding can
run, and why not); a worked example for each of the five ways in; writing a
host; reading a recording from another language; the versioning promise.
**Every code example in it runs in the test suite**, so the guide cannot
drift from the code. `docs/QUICKSTART.md` is the five-minute version.

**Held to it by tests:** no `atk` import anywhere in `athanor/` (AST, not
text — the ATK stale-needle lesson); the whole suite runs under `NullHost`;
the non-GUI package imports with PySide6 absent; the format specs are checked
against files the recorder actually writes.

**Tests without shipping a model.** Plumbing tests use a tiny GGUF the test
suite writes itself with random weights (S5's writer) — enough to exercise
loading, the loop, recording and replay, meaningless as a model. Tests that
need a real model read its path from `ATHANOR_TEST_MODEL` and skip when it
is unset.

## 2b. Scale — the models it is for

Bill, 2026-09-27: *"will it work on models of significant enough size to have
thoughts. Models like the dolphin series, magistral, etc."* It is built for
exactly those: 24B models such as Magistral Small and
Dolphin-Mistral-24B-Venice-Edition (40 layers, 5,120 wide, a 131,072-token
vocabulary) on a 16 GB card with 64 GB of RAM. The numbers below are
ESTIMATES until S1 and Speed measure them.

- **The file tabs** read the vocabulary only — any size, no VRAM.
- **Behaviour and the Waterfall** run on anything ATK already runs, at
  ATK's speed. The logits of one step are 512 KB (131,072 floats); a
  recording keeps the top 256 per token, about 2 KB, so a 10,000-token
  Magistral reasoning trace is ~20 MB — or ~1.3 GB with the optional full
  8-bit distribution.
- **The Tap** copies each layer's output: 40 × 5,120 floats, about 800 KB a
  token — small beside the forward pass. The logit lens then projects every
  layer through the output matrix (131,072 × 5,120, ~2.7 GB at 32-bit in
  RAM) — tens of GFLOP per token, batched on the CPU after the fact; well
  under a second a token, to be measured. The tuned lens, which reads early
  layers more faithfully, trains small per-layer maps from Tap data alone —
  no gradients through the model.
- **Attention maps** need flash attention off and grow with the context;
  recorded for chosen layers and heads, not all 40 × 32.
- **Bigger than the card** (a 70B at 4-bit, ~40 GB): everything works with
  layers in RAM, slowly — a few tokens a second or less.
- **Two honest limits.** A readout of a 4-bit model is a readout of the
  4-bit model — the Quantization tab says how far that is from 16-bit, and
  the small open models (Pythia, OLMo) can be studied at 16-bit to check.
  And M19's gradients are the hardest part to scale — not by memory (the
  lens needs gradients with respect to activations only: no weight
  gradients, no optimizer state; a 24B GGUF fits the card or 64 GB of RAM)
  but by time and by plumbing. S6 has two GGUF roads for it; 1–8B models
  come first because they calibrate fastest.

## 3. The tabs

Five groups, left to right the way the work goes: the file → how it behaves →
changing it → judging it → embedders. The Notebook is the spine under all of
them.

### 3.0 Notebook — the spine (Phase 1)
Every run of every tab, automatically: the question, the files and their
hashes, every setting, the versions, the result, and the operator's notes.
Compare two runs side by side; "run it again" reproduces one; export
Markdown / JSON. Append-only; editing a note keeps the original.

### Group 1 — THE FILE (no model on the card)

#### 3.1 Inspect (Phase 1)
**Question:** what IS this file, and is anything wrong with it?
- Anatomy: architecture, parameters, layers, heads, context declared vs
  trained (and the RoPE scaling that makes the difference), the
  quantisation mix per tensor class (a bar: how much is Q4_K, Q6_K, F16…),
  size, SHA-256, provenance keys (`general.name`, `base_model.*`,
  `quantized_by`, `license`, source URL).
- Tokenizer: model type, pre-tokenizer, vocabulary size, special tokens with
  ids and attributes, add-BOS / add-EOS, the chat template and what it looks
  like (`prompt_templates.detect`, llama.cpp's own detection).
- **Health checks**, each with its rule and its evidence, never a bare
  verdict:
  - pre-tokenizer missing or unknown (llama.cpp prints "GENERATION QUALITY
    WILL BE DEGRADED" and loads anyway);
  - EOS is not the template's end-of-turn token (replies that never stop);
  - BOS twice (the template writes `bos_token` AND add-BOS is on);
  - **template markers that are not special tokens** — render the template,
    tokenize the render: `<|im_end|>` must come out as ONE control token, not
    as `<`, `|`, `im`, `_end`…; if it splits, the model has never seen the
    marker it is being prompted with;
  - vocabulary size ≠ rows in `token_embd`;
  - the name disagreeing with the header ("1M" in the filename; what the
    header declares; what the card can actually hold — ATK's measured
    usable-context table);
  - with a projector: does the mmproj belong to this text model (base-model
    name, projection width = embedding width)?

#### 3.2 Tokenize — the context ruler (Phase 1)
**Question:** how much of MY material fits, on which model?
- One text — typed, a file, a folder, or ATK's own (a discussion, a
  document set, the RF log) — through every chosen model's tokenizer at
  once (vocab-only). Tokens, the pieces, tokens per word and per character,
  by language and script.
- **"Your 32k context holds about 9,100 words of Pashto on model A and
  23,400 on model B"** — the number that should decide which model reads
  which material.
- **The analyst's own strings**: IP addresses, MGRS, callsigns, hashes,
  transliterated names, Arabic-script names with and without diacritics — the
  worst-split list. A model that sees `10.0.0.5` as seven pieces is more
  likely to corrupt it when it copies it.
- Fertility across the whole library as one table, re-run whenever a model
  arrives.
- **Numbers** (Phase 1.1). How each tokenizer cuts digits: one at a time
  (Llama 2's SentencePiece, Qwen2's `\p{N}`), up to three from the left
  (Llama 3 and its kin, `\p{N}{1,3}`), or wherever BPE learned to merge
  (GPT-2's `\p{N}+`). Found by tokenizing numbers rather than trusting the
  pre-tokenizer's name, and shown on the analyst's numbers: coordinates,
  frequencies, IP addresses, phone numbers, dates. It matters: grouping
  digits from the right measurably helps arithmetic (Singh & Strouse, 2024;
  Zheng et al., 2025).
- **Slack** (Phase 1.1). A text has many spellings in the same vocabulary;
  llama.cpp picks one (the canonical). The shortest spelling of the same
  bytes, found by dynamic programming over the vocabulary, says how much
  room there is. MEASURED, vocab-only. Whether the model still understands
  the other spellings is M24.
- **Variants** (Phase 1.1). Every form of a word the vocabulary holds
  (`analyze`, ` analyze`, `Analyze`, ` ANALYZE`) and how close their
  embedding rows sit, by cosine, read from the file. It shows which forms
  the model treats as one word and which as different words. M8 reads the
  same rows.

#### 3.3 Compare — two files (Phase 1)
**Question:** how do these two differ — and are they the same family?
- Tokenizer diff: identical? which ids differ, which special tokens, which
  flags, which pre-tokenizer. This is what catches a fine-tune whose
  tokenizer drifted from its base.
- Metadata diff; tensor diff (names, shapes, quant types).
- **Lineage**: same vocabulary + same shapes + a sample of embedding rows
  compared numerically → "built on the same base" with the evidence;
  otherwise how they differ.
- **The compatibility matrix**: vocabulary overlap (by token string) between
  every pair in the library — which models could even share a draft model,
  a LoRA, or a tokenizer transplant (M4).

#### 3.4 Template — the prompt microscope (Phase 1)
**Question:** what does the model actually see?
- A sample conversation — or a real one from the host (in ATK, a reply's
  `sent`) — rendered through any template (Auto, any in Athanor's library,
  a file); the render tokenized
  and coloured (control tokens / text / BOS / EOS); where generation will
  stop; two templates side by side with the difference marked.
- The Model Adjustments template picker lives here too, with the render
  beside it — choose with the evidence in front of you.

### Group 2 — BEHAVIOUR (one model on the card)

#### 3.5 Next Token — the logit microscope (Phase 2)
**Question:** what was the model choosing between, right here?
- At any point in a prompt: the top 20 next tokens as bars, with
  probabilities and the entropy. Click one to take it and step on — a token
  tree you can walk down several branches of.
- **Sampler shaping, live**: temperature, top-p, top-k, min-p, repetition
  penalty re-shape the bars as they move — exactly which candidates survive
  "temperature 1.0 / top-p 0.95 / top-k 64" (Muse Glimmer's documented
  settings), no longer a matter of trust.
- Logit bias: push a token up or down and watch the continuation change.

#### 3.6 Waterfall — the thought recorder (Phase 2) · THE FLAGSHIP
Bill, 2026-09-27: *"I'm especially excited about the token waterfall.
Imagine how interesting that could be, especially if it is recorded so
someone can go back and forth through time watching its answers evolve."*

**Question:** how did the model get to this answer — token by token — and
what almost happened instead?

ATK's RF waterfall, pointed at a model. Time runs down the screen; the
vocabulary runs across it; colour is probability. Every token the model
writes is one line of the waterfall — the whole distribution it was choosing
from at that moment — and the whole reply is a recording you can scrub
through like an I/Q capture.

- **The recording.** Per step: the chosen token; the top 256 candidates with
  their log-probabilities; the probability left in the tail; the entropy;
  the wall-clock time; and both the model's RAW distribution and what the
  sampler made of it (temperature, top-p, min-p). Optionally the full
  distribution at 8 bits — about 130 KB a token on a 131k vocabulary, so
  ~130 MB per 1,000 tokens; off by default. The header carries the model's
  hash, the template render, the prompt, every setting, the seed and the
  versions. **Shaped like SigMF** — a JSON meta file and a binary data file
  (`.athrec-meta` / `.athrec-data`) — because that is the recording format
  ATK's RF side already speaks.
- **Two ways to get one.**
  - LIVE: Athanor's own generation loop, so the timing is real and the
    thinking pauses are visible. (Not the binding's custom-logits hook: it
    carries its author's own note, "This is probably broken".)
  - **AFTER THE FACT, for any reply ATK has already written**: a
    discussion's `sent` messages and the reply, forced back through the same
    model — the distribution at every step reconstructed, with nothing
    changed in Chat's own path. It matches what happened live only when the
    model, the settings and the arithmetic match (S3 says how often), and it
    has no timing. Chat gets a **[waterfall]** link on every reply.
- **The player.** The waterfall; the entropy as a trace beside it; the text
  so far with the current token lit; the top-20 bars at the cursor (3.5's
  microscope, on a recording). Scrub, step, play at the recorded speed or
  faster, bookmarks and clips — ATK's RF bookmarks and clips, the same verbs.
  The **thinking → answer boundary** is marked — whatever the model's
  markers are (`[THINK]…[/THINK]` in the newer Magistrals, `<think>…</think>`
  in Magistral Small 2506 and many others): the moment it stops thinking and
  starts saying.
- **The axis across.** Three choices: vocabulary by id; by rank (a sorted
  spectrum); or **by meaning** — every token placed along one axis by its
  output embedding (a principal component, or a seriation), computed once per
  model. Then a band of the waterfall is a region of meaning, and the model
  changing its mind is energy moving across the band. EXPERIMENTAL; the most
  interesting view if it works.
- **Thought as a signal.** The entropy trace is a signal, so treat it as one:
  an FFT, an autocorrelation. A repetition loop shows up as a spectral line
  **before** the text is visibly stuck — an early warning Chat could use to
  stop a runaway reply.
- **Comparing.** Two recordings, synchronised: two models, two quants, a
  template change, an override, two seeds. Side by side, and a difference
  waterfall (the KL divergence at each step).
- **Branching — the time machine.** Rewind to any step, pick a different
  candidate, and let it run: a child recording joined to its parent at that
  step. The tree of what could have been — every answer's roads not taken,
  explorable. The model's state is saved at a branch point
  (`llama_state_seq_*`, in the pinned binding) so a branch does not re-read
  the prompt. M15's multiverse is the same tree, grown from seeds.
- **Watching answers evolve.** A stack of recordings of the same question —
  regenerations, seeds, models, before and after an override — aligned step
  by step, scrubbed together: where they agree, the moment they part, and
  which road each one took.
- The instrument an RF analyst already reads every day, turned on a model:
  the same eye for a steady carrier, a drift, a burst, a spur.

#### 3.7 Surprise (Phase 2)
**Question:** where does this text surprise the model — and how much, overall?
- A text through the model, every token coloured by surprise (−log p); the
  perplexity and bits per character of the whole; the most surprising spans
  listed.
- **For an analyst this is an anomaly detector**: in an intercept or a
  seized document, the spans a model finds least predictable are code words,
  unusual phrasing, OCR damage, a different author. EXPERIMENTAL — surprise
  is a measurement of the model, not of meaning — and worth trying on real
  material.
- The honest "did it help?" for every other tab: an override, a template, a
  quant, a LoRA — same text, before and after, one number.
- Held to `llama-perplexity` on the same text (the S2 tools).

#### 3.8 Context — the truth about the window (Phase 2)
**Question:** how much of its declared context does this model actually use?
- Needle-in-a-haystack in Bill's OWN filler (his documents, not lorem ipsum):
  a grid of length × depth, pass / fail, up to what fits on the card with and
  without offload (ATK's VRAM plan). Then harder variants — several needles;
  two facts far apart that must be combined; the needle phrased differently
  from the question.
- The number Chat should compress at, per model — today it is one share for
  every model (`discussions.compress_at_pct`).

#### 3.9 Speed — the bench (Phase 2)
**Question:** what does each setting cost, on this card?
- Prompt processing and generation tok/s, time to first token, across layers
  on the card (offload), KV cache type, flash attention, context size, batch
  — a speed surface per model.
- **This is the measurement ATK's context architecture asked for and never
  got** ("*NOT MEASURED: the speed penalty itself … Time it before setting
  routing defaults*") — and the prompt-cache payoff of stable-first ordering,
  measured instead of argued.
- Held to `llama-bench` (S2).

#### 3.10 Structure — does it keep the shape? (Phase 2)
**Question:** how reliably does this model produce what ATK parses?
- N runs of ATK's own structured asks — entity extraction JSON, the ACH
  matrix, Structured-stance labels, the translation JSON — with and without
  the grammar: valid, schema-conformant, and (against a small gold set) right.
- A GBNF workbench: write or paste a grammar, run it, see where it
  constrains and where the model fights it.
- **Grammar pressure.** With a grammar on, the Waterfall records at each
  step how much probability the grammar removed and what rank the chosen
  token had without it. That trace shows where the format forced the
  model's hand. The same questions are run with and without the grammar,
  because format restrictions can cost reasoning (Tam et al., 2024): a
  grammar guarantees the shape, not the answer.

### Group 3 — CHANGING IT (at load; never the file)

#### 3.11 Override (Phase 2; writing it down Phase 4)
**Question:** does changing this metadata fix what Inspect found?
- The keys llama.cpp lets you override — add-BOS / add-EOS, the
  pre-tokenizer, special-token ids, other scalars — per model profile, applied
  at load through the binding's `kv_overrides`. Each change shows a probe
  string tokenized before and after, the template render before and after,
  and a Surprise number before and after.
- Presets for the known fixes ("Llama-3 fine-tune with no pre-tokenizer →
  `llama-bpe`"), each with the evidence that it helped.
- **Writing it down (Phase 4):** the fix baked into a NEW GGUF; a
  tokenizer copied from a donor into a NEW GGUF when the vocabularies line up
  (Compare says whether they do). Never the original.

#### 3.12 Adapters — anatomy and comparison (Phase 3)
**Question:** what did this fine-tune change — where, how much, what did it
learn, what did it forget — and how do LoRA, QLoRA, DoRA and a full
fine-tune compare on the same job?

Bill, 2026-09-27: *"I wonder if we could compare lora/dora/qlora."*

**What arrives, and how** (checked in b11093): a LoRA or QLoRA adapter
converts to a GGUF adapter with llama.cpp's `convert_lora_to_gguf.py` and
loads over its base (`lora_path`, `lora_scale`, in the pinned binding) —
QLoRA's adapter is an ordinary LoRA that was trained against a 4-bit base.
The converter has no DoRA support, so a DoRA arrives merged into its base
as a full model; so does any full fine-tune (Dolphin). Athanor compares all
of them the same way: as a change to the base. **Athanor does not train** —
llama.cpp's own fine-tuning is FP32-only and marked work in progress — so
the adapters come from Axolotl, Unsloth or PEFT; `docs/` carries the
protocol for a fair comparison (same base, same data, same steps, same
evaluation).

- **Where it changed.** Per layer and per matrix, the size of the change
  (for a LoRA, the product of its two small matrices; otherwise the merged
  weights minus the base) and its effective rank — how many directions the
  change really uses.
- **Magnitude or direction.** DoRA's own paper splits every weight column
  into a length and a direction and shows that LoRA and full fine-tuning
  change them in different patterns. Athanor runs that decomposition on any
  pair of files, so the four methods are compared by the analysis that
  motivated one of them.
- **What it learned, what it forgot.** Surprise on the training data (or a
  sample) before and after; Surprise and KL on the operator's own general
  material before and after — the "learns less, forgets less" trade-off
  measured on Bill's text, not a benchmark's.
- **The dose curve.** The adapter's scale from 0 to 1.5: learned and
  forgotten as two curves; where they cross.
- **The quantization mismatch.** A QLoRA adapter was trained against one
  4-bit base and is usually applied to a different GGUF quant of it. Base
  and adapter at Q4_K_M, Q8_0 and 16-bit: how far each lands from the
  others.
- **Two adapters at once.** llama.cpp stacks adapters with separate scales;
  does the pair interfere — KL of the pair against each alone.
- Later: **draft pairing** — which small model can draft for which big one
  (the vocabularies must match: Compare), acceptance rate and real speed-up.

### Group 4 — JUDGING IT (two or more configurations)

#### 3.13 Quantization (Phase 3)
**Question:** what did this quant cost me — on my material?
- A reference (the higher-precision file) and a candidate run over the same
  corpus; top-k log-probabilities cached from the first so only one model is
  ever on the card; KL divergence, top-1 agreement, the perplexity ratio —
  per corpus type (English prose, Pashto, code, RF logs).
- "Is Q4_K_M enough for translation, or spend the VRAM on Q6_K?" answered
  with a number instead of a forum post. Held to `llama-perplexity
  --kl-divergence` (S2).

#### 3.14 Arena — blind A/B (Phase 3)
**Question:** which do I actually prefer?
- Two configurations (model, quant, template, persona, LoRA), the same
  prompts — Bill's own, harvested from discussions if he likes — answers side
  by side, shuffled and unlabelled; he picks or calls a tie. A preference
  score per category with its count and its uncertainty; nothing ranked on a
  handful. A person judges; no model grades a model.

#### 3.15 Report Card — the overnight battery (Phase 3)
**Question:** how does every model in the library do at MY jobs?
- A fixed battery, deterministic scoring, run overnight through ATK's render
  queue for each configuration:
  - **Extraction**: entity / relationship recall and precision against a
    gold set Bill labels once.
  - **Translation round-trip** in the Bilingual languages, scored by
    character n-gram overlap against reference translations.
  - **Grounding**: unsourced specifics per 1,000 tokens (`grounding.py`).
  - **Structure**: 3.10's numbers.
  - **Robustness** — TokSuite's idea on Bill's material: the same prompts
    perturbed (typos, OCR confusions, diacritics dropped or added,
    Latin↔Arabic-script transliteration, mixed scripts, homoglyphs, Unicode
    normalisation, number formats) and the answer's change measured.
  - **Calibration** — "does it know what it knows?": questions with known
    answers, the model's stated confidence and its token-probability
    confidence against how often it was right. The measured basis for the
    estimative language of Part 3 item 8 (ICD 203).
  - Speed, cut-off and continuation rates.
- One table, sortable by job. This is the controlled twin of the live
  scoreboard in FUTURE_PLANS Part 3 item 12.

### Group 5 — EMBEDDERS

#### 3.16 Embedders (Phase 3)
**Question:** which embedder finds the right passage in MY documents — and at
what cost on the CPU?
- Recall@k and mean reciprocal rank on Bill's corpus (queries with known
  answers, labelled once), cross-lingual (an English question, a Pashto
  passage), milliseconds per chunk on the CPU (Bill's rule: embedders never
  touch the card). Decides Chat Part 4 item C (the RAG cost) with numbers.

---

## 4. The Mad Science Wing

*Everything here is EXPERIMENTAL: research-grade, genuinely educational, and
mostly things nobody runs on their own desktop. Each says what it teaches,
how it would work on a 16 GB card, what it needs first, and where the idea
comes from.*

### The instruments the wing needs

- **The Tap (spike S1).** llama.cpp calls an evaluation callback for every
  tensor it computes (`cb_eval` in the context parameters — present in the
  pinned binding), which is how `examples/eval-callback` prints
  intermediates. From Python: a ctypes callback that answers "yes" for the
  tensors wanted (each layer's output, `l_out-N`) and copies them out with
  ggml's `ggml_backend_tensor_get`. **Not yet proven from Python on Windows
  / CUDA** — the binding loads libggml but binds none of its functions; the
  spike binds four and measures the cost per token. Every "internals"
  experiment below waits on it.
- **The Wheel.** `llama_set_adapter_cvec` adds a vector to the residual
  stream across a range of layers — a control vector. In the pinned binding.
- **Surgery (S2 + S5).** A GGUF writer, and llama.cpp's own `llama-quantize`
  / `llama-imatrix` / `llama-gguf-split` built from the same llama.cpp tree
  ATK already compiles.
- **Input as vectors.** `llama_batch` accepts embeddings in place of token
  ids (its `embd` field, which is how a vision projector feeds an image in).
  That lets an experiment hand the model something no token spells, such as
  a phrase's pieces averaged into one vector. Available in the pinned
  binding.
- **Positions by hand.** Every token in a batch carries its own position
  and sequence ids, so two passages can sit at the same positions, side by
  side, instead of one after the other.
- **Forking the cache.** `llama_memory_seq_cp` / `llama_memory_seq_rm` copy
  and cut sequences in the KV cache, both in the pinned binding. An
  experiment can branch, run, compare and discard without re-reading the
  prompt. It is the in-memory form of the Waterfall's saved branch points.

### M1 · Watch it think — the logit lens
Project every layer's hidden state through the model's final norm and output
matrix and read off the token it "would say" at that depth. A grid —
layers down, tokens across — showing where the answer appears, where it
changes its mind, where a translation switches language in the middle of the
network. The single most striking picture of what a transformer does.
*Needs:* the Tap; the output matrix dequantized (a few GB of RAM for an 8B —
64 GB is plenty). Small models first. *From:* nostalgebraist's "logit lens"
(2020); the "tuned lens", Belrose et al. (2023).

### M2 · The steering wheel
Build a direction from contrasting examples — formal vs casual, English vs
Spanish, hedged vs confident, terse vs expansive — as the mean difference of
hidden states at a layer; add it back through the Wheel with a strength
slider and watch the same question answered differently. Concepts as
directions you can turn. *Needs:* the Tap + the Wheel. *Boundary:* style,
language and tone — never a refusal direction. *From:* Activation Addition,
Turner et al. (2023); Contrastive Activation Addition, Rimsky et al. (2024);
Representation Engineering, Zou et al. (2023) — llama.cpp's
`cvector-generator` implements the PCA form.

### M3 · Mind-reading probes
Record hidden states for labelled sentences — true / false statements,
languages, sentiment, "a fact given in the document" vs "a fact the model
supplied" — fit a linear probe per layer (plain numpy) and plot where each
concept becomes readable. The analyst's question: **can the model's inside
tell when its outside is confabulating?** If a probe predicts ATK's
unsourced-specifics flags better than chance, that is a new instrument for
grounding. *Needs:* the Tap. *From:* "The Geometry of Truth", Marks & Tegmark
(2023); "Language Models (Mostly) Know What They Know", Kadavath et al. (2022).

### M4 · Frankentokenizer
The honest answer to "various tokenizers", in three acts. **One**: load a
model with a foreign tokenizer on a copy and watch it fall apart — a
one-click demonstration of why the embedding table and the vocabulary are
one thing. **Two**: the compatibility matrix (3.3) — how much of a donor
vocabulary even exists in the model. **Three**: a real transplant — a NEW
GGUF whose shared tokens keep their embeddings and whose new tokens get the
average embedding of their pieces under the old tokenizer — then Surprise
and Tokenize on Pashto before and after: did the context stretch, and what
did it cost? Expect it to hurt; the point is to measure how much. *Needs:*
Surgery. *From:* WECHSEL (Minixhofer et al., 2022), FOCUS (Dobler & de Melo,
2023), zero-shot tokenizer transfer (Minixhofer et al., 2024).

### M5 · Brain surgery
Remove or duplicate blocks of layers on a copy and plot quality against what
was cut. Research found the deeper layers of many models can be pruned with
surprisingly little loss — Bill can see the curve for each of his own models,
and try "passthrough" self-merges (a block repeated) that the community
builds by hand. *Needs:* Surgery (a copy with `block_count` and the tensor
names rewritten). Dense architectures first. *From:* "The Unreasonable
Ineffectiveness of the Deeper Layers", Gromov et al. (2024).

### M6 · Model alchemy
Two fine-tunes of the same base (Compare proves they are siblings):
interpolate their weights at α = 0 … 1 and measure Surprise along the line —
the loss landscape between two models, drawn. Sometimes the middle beats
both ends. *Needs:* Surgery + requantisation (llama-quantize) and RAM (64 GB
holds two 8B models in F16). *From:* model soups, Wortsman et al. (2022);
linear mode connectivity, Frankle et al. (2020).

### M7 · Precision torture
Quantize ONE class of tensor harder than the rest — attention values, the
feed-forward down-projections, the embeddings — and measure KL divergence
(3.13) for each: which parts of this brain are fragile, which do not care.
Then read the importance matrix llama.cpp computes (`llama-imatrix`) and see
whether it predicted the same thing. *Needs:* llama-quantize with per-tensor
types + imatrix (S2).

### M8 · The glitch token hunt
Some tokens were barely trained — the "SolidGoldMagikarp" family — and a
model given one can misbehave oddly. Find the candidates from the embedding
and output matrices alone (their statistics give them away), then confirm by
asking the model to repeat each. Practical for ATK: intercepts and scraped
text contain exactly the strange strings that land on these. *Needs:* the
reader + numpy (dequantized rows). *From:* "Fishing for Magikarp", Land &
Bartolo (2024).

### M9 · The waterfall in three dimensions
*The token waterfall itself was promoted to a tab (3.6); this is its mad
extension.* With the Tap, record the logit lens (M1) at EVERY step: for each
token the model writes, what every layer would have said. Time down, depth
across, and the vocabulary as the third axis — scrub time AND depth, and
watch an answer rise through the layers at each moment of the reply: where a
translation decides its language, where a fact is retrieved, where the model
changes its mind halfway up the stack before its output ever does. *Needs:*
the Tap (S1), the recorder (3.6).

### M10 · The LLM compressor
A language model is a compressor: drive an arithmetic coder with its
probabilities, compress a file, decompress it byte for byte — and compare the
size against zip and xz on Bill's own text. The bits per character is just
Surprise in other units; the decompression is the magic trick. *Needs:*
bit-identical logits on every run (spike S3 — GPU arithmetic may not be;
the CPU path or a fixed batch may be required). *From:* "Language Modeling Is
Compression", Delétang et al. (2023).

### M11 · Whose words? — model attribution
Give the Lab a text and it asks every model in the library how surprising it
finds it: the least-surprised family is the likeliest author, and the ratio
between two models' surprise separates machine text from human text better
than either alone. **A forensic instrument** — attributing a seized document
to a model family — and one the Forensics Workshop could call. Probabilistic,
never proof; reported with its error rate on labelled samples. *From:*
"Spotting LLMs With Binoculars", Hans et al. (2024).

### M12 · Telephone
A message passed through a chain — English → Pashto → Dari → Arabic →
English, across models — with the drift measured at every hop (embedding
similarity, Surprise, which specifics survived: `grounding`'s patterns). How
much meaning survives the Bilingual Intermediary on these models? And its
cousin: two models (or two personas) talking for fifty turns, watching them
drift, loop, or collapse into one voice.

### M13 · Time capsule
When does each model's world end? Surprise on dated text, month by month —
the knowledge cutoff shows up as the month the text stops being predictable.
Needs a dated corpus Bill supplies (news, logs, anything with dates). A
small, satisfying measurement no model card states honestly.

### M14 · Signed by ATK — watermarking its own words
Sample ATK's generations with a keyed "green list" watermark (in Athanor's
own sampling loop — the binding's logits hook is not trusted), and detect it
later: **did ATK write this
paragraph of the report?** Provenance for model text, beside the
"model-generated" labels ATK already puts on extracted graph nodes — and a
measured answer to how many edits a watermark survives. *From:* "A Watermark
for Large Language Models", Kirchenbauer et al. (2023).

### M15 · The multiverse
The same question, a hundred seeds. Cluster the answers (embeddings, on the
CPU) and draw the space: how many *different* answers does this model really
hold — one confident mode, three camps, or noise? Which specifics appear in
every universe and which in one? The consistency check (Part 2, three
samples) grown into a map, and a direct look at what "temperature" does to
a model's opinions. *Needs:* logits only; the render queue for the hundred
runs.

### M16 · Where is it looking?
With flash attention off, the Tap can also catch each layer's attention
weights (`kq_soft_max-N`): a heat map of which earlier tokens each word of
the answer was built from — which sentence of the attached report a
conclusion actually rests on. Educational for everyone, and for an analyst a
second opinion beside the grounding check. *Needs:* the Tap, flash attention
off for the run (slower; measured by Speed).
**The causal check.** Attention shows where the model looked, not what
mattered. To test that, keep the cache up to a passage, re-read the context
from there without it (Forking the cache), and measure how the answer's
distribution moves, token by token in the Waterfall (KL). That measures
which source a conclusion rests on by taking the source away. It needs
logits only, so it can come before the Tap.

### M17 · The model's own dossier
Before a model analyses a case about "Brightline Ltd", ask it — with no
documents — what it already believes about every entity in the case: a graph
of its priors, built by the same extraction ATK uses. Then lay it over the
case's Network Link graph built from evidence: where they agree, where the
model "knows" things no document says, where it contradicts the evidence.
**A contamination audit** — the places a model is most likely to smuggle its
own assumptions into an assessment, found before it does. Every analyst
using a model should run it. (Standalone, the entities and the evidence
graph arrive as plain JSON; inside ATK, from the case.)

### M18 · Evolution — a model across its training
Bill, 2026-09-27: *"compare checkpoints from an evolutionary standpoint, see
if there's an indicator that emerges when a model has been overtrained … see
if perhaps the tokenizer needs to evolve with the model, not just remain
static."*

The same model, stopped at many moments of its training, put through the
same instruments — every measurement becomes a curve across training.

- **The series.** EleutherAI's Pythia: eight sizes from 70M to 12B, 154
  checkpoints each (step 0, log-spaced early steps, then every 1,000 steps
  to 143,000), every size trained on the same data in the same order;
  Apache-2.0. AI2's OLMo also publishes intermediate checkpoints. Both
  architectures load in b11093 (`gptneox`, `olmo` / `olmo2`), and llama.cpp's
  own converter (`convert_hf_to_gguf.py`, MIT) turns a checkpoint into a
  GGUF. The operator downloads and converts; Athanor fetches nothing. On
  16 GB: up to 6.9B at 16-bit, 12B quantized.
- **Fine-tuning checkpoints work the same way.** A fine-tune's checkpoints
  (Bill, 2026-09-27: the Dolphin series is typically released as it trains,
  and he knows its researcher) make a series whose step 0 is the base model
  — Dolphin-Mistral-24B-Venice-Edition's base is Mistral-Small-24B-Instruct-
  2501 — so every curve below shows what the fine-tune changed, and when.
  Two practical rules: every checkpoint quantized with the same quant type
  (and the same importance matrix, or none), so the differences are the
  training and not the quantization; and if the training was LoRA, each
  checkpoint is a small adapter loaded over one base (the Adapters tab),
  not a 14 GB file each. Tokens a fine-tune ADDS to its base vocabulary (a
  new chat format's markers) start untrained, and the series shows them
  being learned from nothing.
- **What is public today (checked 2026-09-27).** Eric Hartford's Hugging
  Face page (`ehartford`) has Dolphin checkpoint repos — e.g.
  `dolphin-2.9.4-llama-3.1-8b-checkpoint`: one full-weight checkpoint in
  safetensors plus `trainer_state.json`, the trainer's log (the loss at
  every logged step). A longer series would come from him directly. The
  loss log goes under M18's curves as-is.
- **The training data is public** (Bill found it, 2026-09-27):
  `datasets/ehartford/dolphin-2.9.4` — 10.1 GB, 18 JSONL files in ShareGPT
  form (`conversations`: `from` / `value`), drawn from many public sources
  (math, code, system-prompt chat, agent data and others; one file per
  source). No licence is stated on the page and each source carries its
  own; Athanor never ships it — the operator downloads it and Athanor reads
  it locally. Athanor needs a ShareGPT JSONL reader, and renders each
  conversation through the checkpoint's own chat template so surprise is
  measured on the text exactly as the model was trained on it.
- **The first real target, available today:** Llama 3.1 8B (step 0) →
  `dolphin-2.9.4-llama-3.1-8b-checkpoint`, with its loss log and its
  training data. Two points rather than a film, but every M18 question can
  be asked of it; an 8B fits the card at Q8 and calibrates M19 quickly.
  Tokenizing the whole 10 GB for token frequencies is a vocab-only job —
  minutes to tens of minutes, ESTIMATE.
- **With the training data in hand**, two sharper questions: the
  tokenizer one becomes *how often each token appears in the fine-tuning
  data* against *how far its embedding moved*; and memorization becomes
  measurable — Surprise on the training examples against held-out text,
  checkpoint by checkpoint: when the model starts remembering its data
  rather than learning from it.
- **Does the tokenizer fit the model?** Step 0 is the untrained model, so
  every token's embedding can be measured against where it started: how far
  each row travelled, and when it stopped moving. Tokens that never moved
  are vocabulary the data never taught. Plotted by frequency, script and
  length, that is evidence — per tokenizer, per corpus — of whether the
  vocabulary fitted the training, and how early in training it was already
  clear. M8's under-trained-token hunt, run as a film.
- **An overtraining indicator.** Springer et al. (2025) found that models
  trained longer can become harder to fine-tune — OLMo-1B trained on 3T
  tokens did over 2% worse after fine-tuning than its 2.3T checkpoint — and
  traced it to parameters growing more sensitive to change. The hypothesis
  to test here: the same sensitivity is visible without fine-tuning, as the
  cost of quantizing each checkpoint (the Quantization tab's KL divergence,
  M7 per tensor) rising late in training. If it holds, a sensitivity curve
  that turns upward is the early warning. EXPERIMENTAL until measured.
- **Skills appearing.** A fixed battery — the four-card selection problem,
  route directions scored against a reference route, multi-step arithmetic,
  Bill's own material — recorded with the Waterfall at every checkpoint. The
  question: does the right answer start gaining probability well before the
  checkpoint where it becomes the model's answer?
- With the Tap: the logit lens at every checkpoint — when depth starts to
  matter — and M19's workspace lens, below.
- *Needs:* nothing new for the behaviour and embedding curves (the reader,
  logits, the recorder); the Tap for the lenses. *From:* Biderman et al.,
  "Pythia" (2023); Springer et al., "Overtrained Language Models Are Harder
  to Fine-Tune" (2025); Land & Bartolo (2024).

### M19 · The workspace lens
After Gurnee et al., "Verbalizable Representations Form a Global Workspace in
Language Models" (Anthropic, Transformer Circuits, 2026-07-06). They define a
Jacobian lens: for each token, the average linearized effect of an activation
on the model's likelihood of producing that token, taken over about 1,000
varied contexts — what the model is disposed to say, not what it happened to
say in one prompt. Read through it, a model shows a small working set in its
middle layers (roughly a third to nine-tenths of the way through; 10–25
concepts at once; under 10% of the activation variance) that carries its
step-by-step reasoning: for (4+17)×2+7 the unspoken intermediates 21, 42 and
49 appear in order, at successively deeper layers; a rhyme is chosen before
the line reaches it. The authors take no position on consciousness, and
call their lens approximate and incomplete (single-token concepts only).

What Athanor would add: the same lens on open models on a desktop. The
working set as a strip beside the Waterfall — what the model is holding in
mind at each token, not only what it says. How large the workspace is in
different models, sizes and quants. And, with M18's checkpoints, **when
during training a workspace appears at all.**

*Needs:* how the output responds to each layer's activations, which
llama.cpp's inference path does not report directly — spike S6, which has
two ways to get it on the GGUF itself. A one-time calibration per model,
cached; reading it is then as cheap as the logit lens. EXPERIMENTAL.

### New ways to give a model information
Bill, 2026-09-27: *"or even better, come up with a better way to add
information or context to a model that nobody has tried yet?"* Four
experiments, each labelled honestly for how new it is — every one gets a
literature check before anything is claimed.

### M20 · Retrieval by thought — a programme, not one experiment
Bill, 2026-09-27: *"Honestly, i'm the most excited about retrieval by
thought. How can you take the concept even further. How far can we develop
it and what could we accomplish with it?"*

Retrieval today is triggered by the question, or by what the model has
already written. The closest research uses the model's insides to decide
WHEN to look something up — its uncertainty (SeaKR, 2024), a probe
(CtrlA, 2024), its attention and entropy (DRAGIN, 2024) — but still picks
WHAT to look up from words already in the text. The new source: the
concepts the model is holding in mind **but has not written**. As it
reasons, read its insides; when it starts thinking about an entity, place,
number or event that something outside it can speak to, fetch that and put
it in front of the model before it commits to a claim. *Status:* not found
in the literature on 2026-09-27; a full literature check comes before any
claim of novelty.

**Why only a tool like Athanor can do it:** it needs the model's insides,
so it works on open models run locally and on no cloud model — a
capability that exists only because the weights are on your own disk.

**The stages** — each stands on the one before, and R0 can end the
programme cheaply:

- **R0 · Lead time** (needs only S1 and the logit lens; no retrieval yet).
  Record the middle layers while a model answers questions about a case
  file, and ask: does each entity, place or number appear inside the model
  BEFORE it is written — and how many tokens before? How often do concepts
  appear that are never written (considered and dropped), and how often do
  written ones never appear early? If there is no lead time, the idea is
  dead and the record says so; if there is, its length is the window every
  later stage works in. A result worth publishing either way.
- **R1 · Retrieval by thought, first version** (logit lens). Trigger on a
  concept that persists across several tokens and across layers, not on a
  flicker; match it to the case's entities (multi-token names matched
  piece by piece); fetch; insert the evidence as a marked note — the
  template's tool/document role where the model has one, a bracketed
  evidence block in its thinking where it does not. When several passages
  come back at once, a variant places them side by side at the same
  positions (Positions by hand), so that none of them comes first and the
  context does not grow. That is a known method (Parallel Context Windows,
  2023; Superposition Prompting, 2024), with a known cost: attention
  spreads too thin unless it is corrected (2024). Here it is measured on
  GGUF models against placing the passages in order. Every insertion is a
  Waterfall **branch point**: the recording keeps the counterfactual — the
  same step without the note — so the effect of each piece of evidence is
  measured, not guessed (KL between the two continuations; did the claim
  change?). *Measured by:* grounded-claim rate and M17's prior-versus-
  evidence count, against question-triggered retrieval on the same
  questions.
- **R2 · The gap alarm.** A thought about something the case file has NO
  evidence for is the model working from its priors. That silence is the
  signal: flag the unsupported thought before it becomes an unsupported
  claim. And the reverse — **contradiction retrieval**: when the thought
  matches something the evidence contradicts, fetch the contradicting
  passage first. M17's contamination audit, live, before the words exist;
  M22's evidence-weighting can then act on it.
- **R3 · Tools by thought.** Retrieval is one tool; the trigger works for
  any. A number in the workspace → a calculator checks it (the paper's
  21 → 42 → 49 are exactly the intermediates to check — catch the wrong
  step in the thought before it reaches the words). A place → the host's
  router or gazetteer (in ATK, the offline road graph — Bill's 2024
  directions test, where a model sent the driver to the wrong end of Old
  Town, is the benchmark). A date → date arithmetic. A frequency or a
  protocol → the host's signal library. **Tool use for models never
  trained to call tools** — the harness watches the thinking and supplies
  the facts; the model never has to ask.
- **R4 · The roads not taken, for the analyst.** Concepts that appear in
  the workspace and are dropped are the model's unspoken alternatives: a
  second suspect considered and discarded, a place weighed and set aside.
  Shown beside the answer — and, in ATK, offered as rows for the ACH
  matrix with the evidence the model never looked at. What it considered
  and rejected, made visible.
- **R5 · Memory by thought.** The index need not be a case file. Point it
  at the operator's own past discussions (in ATK, the discussion store):
  when the model thinks of something discussed before, that conversation
  comes back. Recall cued by what is on its mind, the way a person
  remembers, rather than a search someone has to think to run.
- **R6 · The sharper lens** (M19). The same stages with the workspace lens
  instead of the logit lens: fewer false triggers, earlier ones. And one
  speculative step: put fetched evidence into the workspace directly as a
  vector instead of as text (M21 says how much a vector can carry) —
  retrieval from thought to thought. EXPERIMENTAL even by this wing's
  standards.
- **R7 · A property of models.** Lead time, gap rate and the grounding gain
  from R1, measured across models, sizes, quants and fine-tunes, and across
  M18's checkpoints: when in training does a model start thinking ahead of
  its words, and does fine-tuning change how much it relies on its priors?

**What it could accomplish:** local models that ground themselves in the
analyst's evidence without being trained for tools; an early warning for
confabulation that fires on the thought rather than the finished sentence;
and a record of each answer — what the model reached for, what it was
given, what it considered and dropped, what rested on its priors — which
is exactly what analytic tradecraft asks a written product to show: its
sources, the line between evidence and assumption, and the alternatives
considered. Every session also produces a labelled record of what a model
needed and when — training data for teaching small models when to look
things up.

**The honest ceiling.** The lens sees single-token concepts and is
approximate; thinking about something is not needing it; inserting text
mid-reasoning must be done cleanly or it confuses the model; and ordinary
retrieval may already cover most of what the thought would ask for. Every
stage is measured against the plain alternative, and a stage that does
not beat it is recorded as such and not built further.

*Needs:* R0–R5 the Tap (S1), Athanor's own generation loop and the
recorder's branching; R6 M19 and S6; R7 M18. From the host, plain
functions: search, and whichever tools it offers (in ATK: the RAG index,
the road graph, the signal library, the discussion store; standalone, any
function passed in). EXPERIMENTAL.

### M21 · How much fits in a vector?
Read a document with and without it in the context, and keep only the
difference it makes to each layer's activations — one vector per layer,
under a megabyte for a whole document (llama.cpp's own
`cvector-generator` builds vectors this way from paired prompts). Later,
apply that vector as a control vector with the document gone, and ask the
questions the document answers. How many of its facts survive? A capacity
curve: facts retained against document length, layers used, vectors
combined. *Status:* vectors are known to carry tasks and styles (task
vectors, function vectors, in-context vectors, 2023), and documents have
been turned into LoRA adapters ("parametric RAG", 2025); how much factual
content a control vector can carry, measured this way, I have not seen.
The likely answer is "not much" — and the exact number is worth knowing.
Useful either way: a cheap "keep this case in mind" primer for long
sessions. *Needs:* the Wheel; nothing else new. EXPERIMENTAL.
**The layer-0 baseline.** The same question can be asked at the input: a
phrase's or a passage's token embeddings averaged into a single input
vector (Input as vectors), with nothing trained. zip2zip (2025) merges
repeated phrases into single "hypertokens" and gets 15–40 % shorter
sequences, but it fine-tunes the model to read them (about 10 GPU-hours).
How much an untrained model can read from an averaged input vector is the
cheap baseline that every deeper injection, R6's included, has to beat.

### M22 · Evidence-weighted decoding
Run each step twice — with the documents and without them — and push the
choice toward what the documents change: the logits with the evidence, plus
a dial times (with − without). The model leans on the evidence over its own
priors, and the Waterfall shows, token by token, every place the two
disagreed — M17's contamination audit, live, inside one answer. *Status:*
a known method (Shi et al., "Trusting Your Evidence", 2023 — context-aware
decoding), rarely available in local tools; the per-token disagreement
view is Athanor's own. *Cost:* a second forward pass per token, over a
short context. *Needs:* Athanor's own sampling loop.

### M23 · Dataset X-ray
Before anyone fine-tunes, put the dataset through the base model: per
example and per token, what the base already knows (little to learn), what
looks like noise (surprise far above the rest), and the learnable middle.
After a fine-tune, the same pass on the checkpoint, and the difference per
source file: **which parts of the data did the teaching.** With the Dolphin
2.9.4 dataset (18 source files) and Llama 3.1 8B against the Dolphin
checkpoint, that question can be answered today. Output: a report, and a
selection file (JSONL) a trainer can use. *Status:* selecting training
tokens by the model's own loss is known (Rho-1, 2024); attributing a
finished fine-tune's learning to its sources this way is the new part.
*Needs:* Surprise and the ShareGPT reader (M18).

### M24 · Other ways to spell it
Give the model the same text spelled in different tokens: the canonical
spelling, the shortest (3.2's slack), character by character, split at
random, and numbers grouped from the right. Then measure, with Surprise and
the Waterfall, how much the model notices. Published work found that
instruction-tuned models keep up to 93 % of their performance on random
spellings and 91 % on character-level ones. Some tasks improve:
character-level spelling helps string manipulation by up to 14 %, and
digits grouped from the right help large-number arithmetic by up to 33 %
(Zheng et al., 2025).

Athanor's part has three pieces:
- the same measurement on Bill's own models and quants;
- the analyst's question: does a character-level spelling help a model copy
  an IP address, a callsign or a transliterated name exactly;
- with the Tap, where in the stack a broken spelling gets repaired.

This is also the test for two ideas from the tokenizer notes (below): a
"shortest spelling" to save context, and a scrambled spelling as a defence
against crafted inputs. *Needs:* logits (Phase 2). llama.cpp accepts any
token sequence, so there is no patch.

### From the tokenizer notes (2026-09-28)
Bill shared a Gemini conversation about tokenizers and asked what was useful
in it. Adopted:
- numbers, slack and variants (3.2);
- grammar pressure (3.10);
- three instruments: input as vectors, positions by hand, forking the
  cache;
- the causal check (M16);
- the side-by-side variant (M20 R1);
- the layer-0 baseline (M21);
- M24.

Not adopted, with the reason for each:
- **A proxy that rewrites jargon into known words.** This is prompt
  rewriting, and it belongs in the host, before the tokenizer.
- **"Gravity tokens"** (attention on chosen tokens multiplied by ten). At
  that strength it breaks the model. The careful form (PASTA, 2023) scales
  selected, profiled heads, and it needs an attention hook that llama.cpp
  does not expose. Revisit after S1.
- **Ghost tokens, the idling pulse, entropy smoothing and the "vent"
  stream.** Each depends on a mechanism that inference does not have. No
  loss is computed at inference. Nothing builds up between tokens that
  could be discharged. A deleted branch leaves nothing behind. Tokens fed
  to an idle model only fill its context. The real part of the idea,
  seeing what the model considered and did not say, is what the Waterfall
  and M19 already do.
- **Bytes where the text is hard, long tokens where it is easy.** The
  trained form of this is the Byte Latent Transformer (2024). On a
  pretrained model it is M24's character-level spelling, and it is measured
  there.

---

## 5. Order

**Spikes first — each a day or less, each answering one question:**
- **S1 the Tap** — capture `l_out-N` from Python on Windows / CUDA; cost per
  token. Gates M1–M3.
- **S2 the tools** — build `llama-perplexity`, `llama-bench`,
  `llama-tokenize`, `llama-quantize`, `llama-imatrix`, `llama-gguf-split`
  from the llama.cpp tree install.bat already compiles (b11093); install.bat
  build time +?. Gates the cross-checks and M5–M7.
- **S3 determinism** — are logits bit-identical run to run on the card, at a
  fixed batch? Gates M10, and says how to cache references.
- **S4 vocab-only speed** — on Windows, for Bill's 14–18 GB files.
- **S5 the writer** — native or pinned gguf-py (MIT); round-trip a real file
  byte-identical before trusting it with surgery.
- **S6 gradients, on the GGUF** — M19 needs how the output responds to a
  layer's activations. Bill, 2026-09-27: *"I want to be able to do it mainly
  on GGUF"*, with 64 GB of RAM behind the card. Two GGUF roads, both read in
  b11093:
  - **Road 1 — no patch, on the card.** The control vector
    (`llama_set_adapter_cvec`, in the pinned binding) adds a vector to one
    layer's output (`llama_adapter_cvec::apply_to` is a single `ggml_add`).
    Evaluate a context once, cache it, then evaluate only its final token
    with a small vector set on one layer: the nudge touches exactly one
    activation, and the change in the logits is a directional derivative —
    by finite differences, forward passes only, at decode speed. Random
    directions over many contexts estimate the average response. The catch:
    llama.cpp's CUDA kernels quantize activations to 8 bits (q8_1) before a
    quantized matmul, so the nudge must stay above that rounding or the run
    uses a non-quantizing kernel; S6 finds the step size.
  - **Road 2 — exact, on the CPU, in RAM.** ggml can differentiate through
    a quantized GGUF: the activation gradient of a matmul is `ggml_out_prod`,
    and b11093's CPU `out_prod` has a path for every quant type. llama.cpp's
    training entry points (`llama_opt_init` / `llama_opt_epoch`) exist, but
    make only F32 tensors trainable and never the control vector, so exact
    activation gradients need a small patch to llama.cpp — ATK already builds
    it from source, and the patch would be offered upstream so standalone
    users are not tied to a fork. Flash attention off (it has no backward
    pass; `llama_opt_init` turns it off itself). A 24B at Q8_0 (~25 GB) fits
    in 64 GB of RAM with room for the backward pass.
  - **The check:** on a small model, both roads must agree with an exact
    reference — PyTorch on the same small checkpoint, in the test suite
    only, never at run time. Athanor stays GGUF.
  - **The cost:** the lens is a calibration, computed once per model and
    cached, like an importance matrix; reading it afterwards costs no more
    than the logit lens. Road 1 on a 24B: hours rather than days by rough
    arithmetic — ESTIMATE, S6 measures. Gates M19.

**Phase 1 also lays the ground everything else stands on**: the public API,
the CLI with `--json`, the host port with `NullHost`, the capability probe,
`pyproject.toml`, the first version of `docs/INTEGRATING.md` with its examples
running as tests, and the no-`atk`-import test. A program that is not ATK can
use Athanor from the first release.

**Phase 1 — the file** (no model on the card): Notebook, Inspect, Tokenize,
Compare, Template. Useful the day it lands; risks nothing.
**Phase 1.1**: numbers, slack and variants (3.2). They need only the
vocabulary and the embedding rows, so no model has to run.

**Phase 2 — behaviour** (one model): Surprise first, because it is how every
later change is judged, and because the after-the-fact recorder is the same
machinery; then **the Waterfall** (the flagship — recorder, player,
branching), Next Token, Context, Speed, Structure, Override.

**Phase 3 — judging** (several configurations): Quantization, Arena, Report
Card, Embedders, Adapters.

**Phase 4 — writing files**: Override's baked fixes, tokenizer copy; then the
Mad Science Wing in the order its instruments arrive — M11, M12, M13, M15
and M17 need nothing new (logits, generation, ATK's extraction) and could
come as early as Phase 2; M8 needs the reader; M1–M3, M9 and M16 the Tap;
M4–M7 Surgery; M10 determinism; M14 Athanor's own sampling loop. M18's
behaviour and embedding curves need nothing new (Phase 2 on); its lenses the
Tap; M19 the Tap and S6. M22 and M23 need nothing new (M23 could come in
Phase 2 alongside Surprise); M21 the Wheel. M24 and M16's causal check
need logits only (Phase 2 on). M20's R0 comes as soon as S1
and the logit lens exist — it is the cheapest test of the idea Bill is most
excited about, so it goes first in the Wing; R1–R5 follow; R6 waits for M19.

## 6. ATK's side

ATK is Athanor's first host, and uses only what any other program could:
the public API, the widgets and the host port. Nothing below exists in
Athanor for ATK's sake.

- Sidebar: **"Model Lab"** (the engine is Athanor), glyph ⚗ (distinct from
  the other nineteen — `test_gui_fixes` checks), near the end of the rail,
  before Setup; the panel's header says EXPERIMENTAL.
- `atk/ui/subsystem.py`: `ATHANOR = Subsystem("athanor", "the Athanor
  engine (Model Lab)", "get_engines.bat /athanor",
  "github.com/photogbill/Athanor", ("atk.ui.lab_panel", "atk.core.lab_host"),
  "Every other workspace is unaffected. The Lab never modifies a model
  file.")`.
- `get_engines.bat /athanor` + install.bat, the same way as the Forensics
  Workshop. The repository, Bill's folder (`D:\Analyst_Toolkit\Athanor`) and
  the package share one name, so the folder `git clone` creates is the folder
  the finder looks for — the trap the Writing engine fell into cannot open
  here. The finder still lists `..\Athanor` and `vendor\Athanor`.
- `atk/ui/lab_panel.py` places `athanor.gui`'s widgets in ATK's rail and
  theme; it draws nothing of its own that another host would lack.
- Chat: a **[waterfall]** link on each reply opens it in the Lab (3.6,
  after the fact) — ATK passes the reply's `sent` messages and text to
  `athanor.api.replay`, the same call the guide documents.
- `atk/core/lab_host.py` fills the port: `borrow_gpu` (the AI-queue ticket,
  unload, restore — escalate's pattern), the VRAM plan, the model folders
  (`discover_models`, never a glob of `ATK\models`), `data_dir()` =
  `ATK\data\athanor\`, settings.
- **Findings flow into ATK only when the analyst applies them**: an Override
  preset into a model profile, a measured usable context into Chat's
  compression threshold, a speed surface into the layer plan, a template
  into Model Adjustments. The Lab proposes; Setup changes only on a click.

## 7. Decisions for Bill

1. ~~The GitHub name~~ — **Athanor** (2026-09-27); package `athanor`.
2. ~~The licence~~ — **MIT, open source** (2026-09-27). ATK stays all rights
   reserved; a permissive Athanor is what lets ATK host it and lets
   outside contributions flow back in.
3. ~~Where results live~~ — **local, always** (2026-09-27): the host's data
   folder (§2). Still open: the default folder for caches and new model
   files.
3a. The prompt-template library: Athanor needs its own (§2, `templates/`).
   ATK's `prompt_templates.py` is yours and all rights reserved; porting it
   means releasing that code under MIT. The plan assumes yes.
3b. Publishing to PyPI as well as GitHub — later, once the API reaches 1.0?
4. Whether install.bat builds the llama.cpp tools (S2) — more build time,
   and the reference every measurement is held to.
5. TokSuite's models and benchmark — welcome, if their licences pass the
   rule?
6. Which Mad Science experiment first — the Waterfall is now a tab and
   comes in Phase 2 regardless. Claude's pick, since retrieval by thought is
   the idea Bill is most excited about: **M20's R0, the lead-time study** —
   one question, cheap once S1 works, and it decides whether the whole
   programme is worth building. Then **M17 (the model's own dossier)**, the
   most immediately useful thing here for an analyst.

## 8. Reading

- TokSuite: Measuring the Impact of Tokenizer Choice on Language Model
  Behavior — arXiv 2512.20757.
- nostalgebraist, "interpreting GPT: the logit lens" (2020); Belrose et al.,
  "Eliciting Latent Predictions from Transformers with the Tuned Lens"
  (2023).
- Turner et al., "Activation Addition" (2023); Rimsky et al., "Steering Llama
  2 via Contrastive Activation Addition" (2024); Zou et al., "Representation
  Engineering" (2023).
- Marks & Tegmark, "The Geometry of Truth" (2023); Kadavath et al., "Language
  Models (Mostly) Know What They Know" (2022).
- Minixhofer et al., "WECHSEL" (2022) and "Zero-Shot Tokenizer Transfer"
  (2024); Dobler & de Melo, "FOCUS" (2023).
- Gromov et al., "The Unreasonable Ineffectiveness of the Deeper Layers"
  (2024).
- Wortsman et al., "Model soups" (2022); Frankle et al., "Linear Mode
  Connectivity and the Lottery Ticket Hypothesis" (2020).
- Land & Bartolo, "Fishing for Magikarp" (2024).
- Delétang et al., "Language Modeling Is Compression" (2023).
- Hans et al., "Spotting LLMs With Binoculars" (2024).
- Kirchenbauer et al., "A Watermark for Large Language Models" (2023).
- Liu et al., "Lost in the Middle" (2023) — for 3.8.
- Biderman et al., "Pythia: A Suite for Analyzing Large Language Models
  Across Training and Scaling" (2023) — arXiv 2304.01373; for M18.
- Springer et al., "Overtrained Language Models Are Harder to Fine-Tune"
  (2025) — arXiv 2503.19206; for M18.
- Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024);
  Dettmers et al., "QLoRA" (2023); Biderman et al., "LoRA Learns Less and
  Forgets Less" (2024) — for 3.12.
- Jiang et al., "Active Retrieval Augmented Generation" (FLARE, 2023); Su et
  al., "DRAGIN" (2024), arXiv 2403.10081; Yao et al., "SeaKR" (2024), arXiv
  2406.19215; "CtrlA" (2024); Su et al., "Parametric Retrieval Augmented
  Generation" (2025) — for M20 and M21.
- Hendel et al., "In-Context Learning Creates Task Vectors" (2023); Todd et
  al., "Function Vectors in Large Language Models" (2023); Liu et al.,
  "In-context Vectors" (2023) — for M21.
- Shi et al., "Trusting Your Evidence: Hallucinate Less with Context-aware
  Decoding" (2023) — for M22.
- Lin et al., "Rho-1: Not All Tokens Are What You Need" (2024) — for M23.
- Gurnee et al., "Verbalizable Representations Form a Global Workspace in
  Language Models" (Anthropic, 2026) —
  transformer-circuits.pub/2026/workspace; for M19.
- Zheng et al., "Broken Tokens? Your Language Model can Secretly Handle
  Non-Canonical Tokenizations" (2025), arXiv 2506.19004; Singh & Strouse,
  "Tokenization counts: the impact of tokenization on arithmetic in
  frontier LLMs" (2024), arXiv 2402.14903. These are for 3.2 and M24.
- Tam et al., "Let Me Speak Freely? A Study on the Impact of Format
  Restrictions on Performance of Large Language Models" (2024), arXiv
  2408.02442. For 3.10.
- "zip2zip: Inference-Time Adaptive Tokenization via Online Compression"
  (2025), arXiv 2506.01084. For M21.
- Ratner et al., "Parallel Context Windows for Large Language Models"
  (2023); Merth et al., "Superposition Prompting" (2024), arXiv 2404.06910;
  "Attention Entropy is a Key Factor: An Analysis of Parallel Context
  Encoding" (2024), arXiv 2412.16545. For M20 R1.
- Zhang et al., "Tell Your Model Where to Attend: Post-hoc Attention
  Steering for LLMs" (PASTA, 2023); Pagnoni et al., "Byte Latent
  Transformer" (2024); Jain et al., "Baseline Defenses for Adversarial
  Attacks Against Aligned Language Models" (2023). These were considered
  for the tokenizer notes.
