# Athanor — the plan

*Scoped 2026-09-27. **Phase 1 built 2026-09-28 (0.1.0)** — the engine,
Inspect, Tokenize, Compare, Template, the notebook, the CLI and the public
API. **The Waterfall built 2026-09-29 (0.2.0)**, ahead of the rest of
Phase 2 because Bill asked for it — the recorder, the `.athrec` format, the
player, and ATK's side: Chat's ⚗ record box and the Model Lab's Waterfall
tab. **The Tap (spike S1) built 2026-09-29 (0.3.0)** at Bill's word ("Go
ahead and start the tap") — proved exact on CPU, its Windows / CUDA cost
and exactness waiting on `athanor tap probe` on Bill's card — with its
first application, **M28's expert map**. **The Reader** (§4) — reading what
a model thinks — designed the same day at Bill's request and parked while
he works on other parts of ATK: *"But I promise, we will revisit it."* See
CHANGELOG.md. Next: §5, "Where to pick up".*
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
   **Writing into a model's state** (the Reader's T0, 2026-09-29) happens
   only in Athanor's own sandbox contexts on the loaded weights — never in
   a host's generation — to measure (a readout, a swap test, a knock-out),
   never along anything refusal-related, and nothing written is kept as a
   steering product. The Tap that runs inside a host's generation only
   ever reads.
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
- **The integrity ledger** (Phase 1.1). Bill, 2026-09-28: *"We can always
  keep a copy of the model in a separate folder just for Athanor, so that if
  anything is damaged we can learn from it and prevent it in the future."*
  Every model file Athanor opens is identified already (size, date, header
  hash); the ledger adds the full SHA-256, taken once, and checks the cheap
  identity on every later open. A file whose date or size moved is hashed
  again, and a changed hash is reported as what it is: this file is not the
  one you measured. Optionally, a published hash the analyst pastes in (a
  model page's SHA-256) is compared, offline. Bit rot, a half-finished
  copy, a re-upload under the same name — each is caught before it is
  measured, and each is a line in the log.

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

**Built, 2026-09-29 (0.2.0).** Bill, 2026-09-28: *"I like the idea of being
able to check a box in the chat interface for athanor record, so i can
review the response on the waterfall tab for it. It should show the actual
response at the top as we go through the waterfall over time, and the
potentials on the waterfall."* Then, 2026-09-29: *"Can you have what is
decided each time also present and highlighted?"* What exists:
- **Recording live**, from a `Llama` the host already has
  (`api.attach_recorder`), a reply carried on in parts as one recording, and
  from the command line (`athanor record`). In ATK: the **⚗ record** box
  beside Send, and an [open in the Model Lab] link on the reply.
- **The player**: the reply along the top (the token at the cursor lit, the
  rest dimmed or hidden, the words the model was unsure of underlined, hover
  for how sure); **taken**, a column on every row with the token actually
  written, whatever its rank (white frame: the favourite; amber and its rank:
  not); the candidates across, favourite first; **Aa read**, which writes
  each candidate in its cell so a row reads as the words weighed; step
  numbers and ◆ **moments of doubt** in a gutter, and a key to jump between
  them; the entropy beside it all; play at the recorded speed.

**What building it taught.**
- *Not the binding's custom-logits hook.* It works at the pinned commit
  (proved on a test-built model), but it cannot see which token the sampler
  took, only the logits, so the last token of every reply would have had to
  be guessed. Wrapping the instance's `sample` sees both: llama.cpp's own
  logits at the index the sampler read, and the token it picked. Tested:
  recorded and unrecorded replies are identical under greedy, top-k/p,
  min-p, penalties, logit bias, grammars, mirostat, a cache hit and a draft
  model.
- *A recording must be live to be exact.* Re-scoring a finished reply in one
  batch rounds differently, and re-tokenizing its TEXT need not give back
  the ids the model generated — models write non-canonical spellings. So the
  after-the-fact recording (below) is labelled a reconstruction.
- *Vision chats* advance the token count over image positions without
  keeping ids there; those recordings keep the reply and leave the prompt's
  ids out, saying why.

**Next for the Waterfall**, in the order they pay back:
- **Take this instead.** Right-click any candidate cell: the model is rewound
  to that step (`llama_state_seq_*`, or the cache fork of §4's instruments)
  and continues from the other word — a child recording joined to its
  parent. The branching below, made one click from the cell that prompts
  the question.
- **End pressure.** A trace of the probability on the end-of-turn tokens at
  every step: where the model wanted to stop, where a reply ran on past it,
  and where one stopped with the thought unfinished. ATK's cut-off replies
  (atk/core/replies.py, 2026-09-24) are exactly what it would have shown.
- **The sampler, replayed.** A recording holds the model's RAW distribution
  at every step, before the sampler touched it — so the sampler can be
  re-run on it without the model: "at temperature 0.2, how likely was it to
  take the same token here?", step by step, and the chance the whole reply
  would have come out the same. The educational version of choosing
  settings, and the practical one: see which steps a setting would change
  before spending a regeneration on it. ESTIMATE — repetition penalties and
  grammars depend on the path taken, and are marked where they apply.
- **Notes on steps.** The analyst marks a step and writes why; the note is an
  annotation in the recording, so a recording becomes something you can hand
  someone ("look at step 212").
- **A clip for a report.** A stretch of steps as an image, or as a
  self-contained HTML page with the candidates in it.
- **Claim confidence (in ATK).** ATK's grounding check already sorts a
  reply's specifics into sourced and unsourced; the recording says how sure
  the model was of each. Crossed, that is four kinds of claim, and the one
  that matters most is new: **unsourced and confident** — the model's priors
  speaking without hesitation, the claims most likely to be believed and
  least supported. A marker on each claim in the reply; M17's audit, done on
  every recorded reply for free.

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
- **After the fact** (above) is labelled a reconstruction: the reply's text
  re-tokenized, scored in one batch. Where the re-tokenized ids differ from
  what was generated (unknowable after the fact), the numbers are for a
  different spelling of the same text — M24 measures how much that matters.

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
- **The KV cache's precision.** llama.cpp can keep the cache at 8 or 4 bits
  (`type_k` / `type_v`), which on a 16 GB card is the cheapest way to more
  context. What it costs is measured the same way: KL divergence against an
  F16 cache, on Bill's material, beside how many more tokens it buys (ATK's
  VRAM plan). Long-context retrieval is where it hurts first, so 3.8's
  needle grid is re-run with it. *From:* KIVI, Liu et al. (2024), on how
  unevenly keys and values tolerate quantization.

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

- **The Tap (spike S1) — BUILT 2026-09-29 (0.3.0), `athanor/tap/`.**
  llama.cpp calls an evaluation callback for every tensor it computes
  (`cb_eval` in the context parameters, what `examples/eval-callback`
  prints with): it asks "do you want this one?", computes up to the wanted
  tensors, and hands each over. From Python: a ctypes callback (one
  dispatcher per context, idle until a Tap is set on it) and five ggml
  functions bound from the binding's own `ggml-base` (`ggml_get_name`,
  `ggml_nbytes`, `ggml_backend_tensor_get`, and `ggml_init` /
  `ggml_new_tensor_4d` for a self-test that pins the start of
  `struct ggml_tensor` before anything is read). The callback is fixed when
  a context is made, so `make_tappable` rebuilds a `Llama`'s context once
  (weights shared). **What building it taught:**
  - *Exact on CPU.* Tapped and untapped logits are bit-identical, idle and
    while copying every layer; the Tap's copy of `result_output` IS the
    logits; the last layer's output through the final norm and the output
    matrix (done in numpy) reproduces the logits to ~2e-6 — the logit lens
    stands. An early "difference" of 7e-6 turned out to be the binding
    re-using its prompt cache between runs (batched vs single-token
    kernels), not the Tap: the probe resets between runs for that reason.
  - *The callback must never return false after computing* — llama.cpp
    stops the graph there and the logits are garbage — and a Python
    exception in a ctypes callback returns 0. Every path is wrapped.
  - *Rows.* A tensor's token axis is axis 1 for almost everything and 2
    for the per-expert MoE tensors (`[1, n_used, tokens]`); the last layer
    holds only the output rows (llama.cpp drops the rest there). A
    forward pass nobody announced (a vision handler decoding by itself)
    gives each tensor's last row.
  - *Cost, on the tiny test models* (CPU, not representative): idle within
    noise to ~10 %; copying a few tensors adds a fixed cost per piece the
    graph is split into. On a real model the per-token compute dwarfs it —
    **to be measured on Bill's card with `athanor tap probe`**, which also
    answers the open question: does computing the graph in pieces on CUDA
    (no fusion across a piece boundary) change the last bits?
  - *Micro-batches.* A batch longer than the context's `n_ubatch` runs
    the whole graph once per micro-batch (consecutive `n_ubatch` tokens),
    inside one `llama_decode`. The graph's first node is asked about first,
    so the Tap marks each new graph by that name and stitches the rows
    back into batch order (review, 2026-09-29: before the fix, every layer
    but the last was silently dropped whenever `n_batch > n_ubatch`).
  - *Names repeat.* llama.cpp gives some names to several nodes in one
    graph (`Qcur-0` three times: before the reshape, after it, after
    RoPE). The Tap keeps the first and says so; the token axis of a
    tensor it does not know is read from its shape against the
    micro-batch's token and output counts, never guessed from the name.
  - *A context that cannot be rebuilt* (memory taken in the gap between
    freeing the old one and making the new) is replaced by a stand-in that
    raises in Python — before the fix, the next generation handed
    llama.cpp a freed context and the process died.
  - *Decided for ATK:* the Tap goes into the loaded model the first time a
    reply is recorded with the Lab's experts box ticked (inside the
    engine's lock, before the generation), and comes out when the box is
    unticked (`lab_host.release_tap`, on a worker through
    `LLMEngine.run_on_model`) or the model is reloaded. Each change
    rebuilds the context, so the next reply re-reads the conversation.
    While it is in, it costs one Python call per graph node per token even
    when idle, and a busy Python thread elsewhere (ATK's GUI) makes each
    call wait for the GIL: the probe's `tapped_idle` on the card says
    whether that matters.
  **Not yet:** attention maps (need flash attention off and per-head
  tensors), the residual stream in ATK's recordings (800 KB a token on a
  24B — the Tap can, ATK does not offer it until M30 has a view for it).
  **Measured on Bill's card (2026-10-02, the first run in ATK; the probe
  itself still to be run):** Qwen3-Coder-30B-A3B-Instruct Q4_K_M, llama.cpp
  with `token_embd` and all 144 expert tensors in RAM, attention, routers
  and norms on the RTX 3080 Ti; 291 tokens at 19.1 tokens/s. The experts
  preset (144 copies a token) cost **9.4 ms a token** — about 18 % — and
  the recorder 1.6 ms; no step flagged; router probabilities summed to 1 at
  every layer-step; the chosen 8 were the router's top-8 at all 13,968
  layer-steps (the probe's MoE check holds on CUDA). Note for the expert
  map: `ffn_moe_weights` is the router's score of the chosen experts BEFORE
  normalising (they summed to 0.13–0.78); Qwen3-MoE normalises them before
  mixing, which `share()` does for the display. Still open from the probe:
  exactness plain vs tapped on CUDA, and the idle cost.
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
**The engine BUILT 2026-10-02 (0.4.0), `athanor/lens/`; the picture next.**
`athanor record … --lens` / `athanor lens RECORDING`: every recorded
`l_out-N` through the file's own `output_norm` and `output.weight`
(Q4_K/Q5_K/Q6_K and the rest decoded natively, no new dependency; the
matrix kept as float32 under `<data>/lenses/<fingerprint>/` and
memory-mapped — 1.2 GB for the 30B-A3B's 152k × 2,048), as a full
distribution per layer per token: each layer's top-k, the favourite's and
the chosen token's rank and log-probability at every layer, the entropy,
in a derived `.athrec-lens` track beside the recording (its own meta
file; the recording is never changed). It checks itself at every step —
the last layer's reading against the recording's own logits — and is
MEASURED only when the favourite agrees at ≥ 98 % of steps; proved on
llama.cpp's tapped recordings of the test models (100 %, ~1e-6). A
post-pass on the CPU: ~0.25 s a token on two cores at the 30B's size,
so tens of milliseconds on Bill's fourteen. **Not yet:** the grid in the
player (layers × tokens, the depth trace, the column at the cursor), the
residual box in ATK's Lab (one setting beside the experts box;
`attach(tap=("experts", "residual"))`), the lens over the PROMPT's tokens
(`rows="all"`), per-layer attribution (the residual's deltas, attention
vs MLP), and the calibrated lens (M30's button).

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

### M25 · Does this doubt matter?
A moment of doubt is only interesting if the roads part. "Large" or "big"
is doubt about wording; "Tuesday" or "Thursday" is doubt about the world.
At a moment of doubt, branch: let the model continue a handful of times
from each of the leading candidates, embed the continuations (on the CPU,
ATK's embedders) and cluster them by meaning. One cluster: the doubt was
about phrasing. Several: the model did not know, and the claim that follows
is the one to check. The number is semantic entropy, computed at the one
step where it matters rather than over whole answers — so it costs a few
short continuations, not a hundred full ones. *Status:* semantic entropy is
established (Kuhn et al., 2023; Farquhar et al., Nature 2024; the cheaper
probes of Kossen et al., 2024); locating it at the Waterfall's own doubt
points is the new part. *Needs:* branching (3.6), an embedder.
EXPERIMENTAL.

### M26 · When did it decide?
A reasoning model thinks for thousands of tokens before a short answer.
Fork the cache at checkpoints through the thinking (every 50 tokens, say),
close the thought there, and read what it would answer at that moment: the
answer's distribution, plotted against the thinking. Three shapes are
possible, and each says something. It settled early and thought on (the
rest was checking — or padding). It settled late (the thinking did the
work). It changed its mind (the thought that changed it is at the step
where the answer moved — the Waterfall shows it). For ATK there is a
practical result: a thinking budget that stops when the answer has been
steady for long enough, instead of running out mid-thought (the cut-off
replies of 2026-09-24). *Status:* truncating a chain of thought to see
whether the answer depends on it is Lanham et al. (2023); stopping early
when the answer converges is recent work (DEER, 2025; "Answer Convergence
as a Signal for Early Stopping", 2025). Athanor's part is seeing it on
Bill's own models and questions, inside a recording, and testing whether a
convergence stop costs any accuracy on his material. *Needs:* forking the
cache; the Waterfall's thinking annotation. EXPERIMENTAL.

### M27 · Canary recordings
ATK builds llama-cpp-python from llama.cpp's master (2026-08-13), so the
code under every model changes without anyone deciding it should. A canary
is a handful of fixed prompts — Bill's kinds of work: a Pashto passage, an
extraction, a long document, a tool-free reasoning question — recorded once
per model and kept. After each update, the same prompts are recorded again
and compared step by step: the KL divergence at every token, and the first
step where the chosen token changed. Nothing changed is a green line. A
changed tokenizer, template or kernel shows as the step where the replies
part, with the recording to show it. *Status:* llama.cpp's own
`--kl-divergence-base` does the batch form for quantization; recorded,
per-token canaries across builds are Athanor's. *Needs:* S3 (whether logits
are bit-identical run to run on the card; the CPU path as the reference if
not), the Waterfall's comparison view. Cheap to run: seconds per prompt.

### Watching the inside — what the Tap opens
Bill, 2026-09-29: *"Is it possible to have something that can break out the
MoE models … highlighting the model inherent persona's being utilized? Also
can you think of other ways to monitor the inner thought process of a
model?"* Everything below reads the model's insides through the Tap (S1),
so S1 is the gate for all of it — and each one records into the Waterfall,
so what happened inside is scrubbed beside what was written.

### M28 · The expert map (mixture-of-experts models)
In a mixture-of-experts model (Mixtral and Dolphin-Mixtral, Qwen3's
30B-A3B, gpt-oss, DeepSeek) every layer's router picks a few experts for
every token. llama.cpp names what the router decided — `ffn_moe_topk-N`,
the experts chosen at layer N, and `ffn_moe_weights-N`, how much each
counted (b11093, llm_graph's MoE block) — so the Tap reads it for a few
kilobytes a token. Shown as a second waterfall beside the first: layers
down one axis, experts across, lit where they fired, for the token at the
cursor; and over a whole reply, which experts a stretch of text leaned on.
Then the questions worth asking: do the same experts carry Pashto and
English? Code and prose? Does a fine-tune (Dolphin-Mixtral against its
Mixtral base) re-route, or only re-weight? Does an expert that is never
used exist (a candidate for pruning, M5)?
**What to expect — honestly.** Experts are not personas. Mixtral's own
analysis found no clear assignment of experts by topic, only patterns
closer to syntax (Jiang et al., 2024); a 2026 study argues routing follows
the geometry of the hidden states rather than domains (Wang et al., 2026);
another finds experts more interpretable than dense neurons, as
fine-grained "task experts" (ICML 2026). The map is how to see which is
true of Bill's models, on his material. *Needs:* the Tap; an optional
second stream in the recording format (per step: layers × experts used,
ids and weights). EXPERIMENTAL.
**Built 2026-09-29 (0.3.0).** `attach(llm, tap="experts")` records
`ffn_moe_topk`, `ffn_moe_weights` and `ffn_moe_probs` for every layer into
`.athrec-tap`; `Recording.tap.experts()` reads it; the player's expert map
shows it three ways — at the cursor (the router's score for every expert in
dB, the experts used framed with their share), over the reply / its
thinking / its answer (how often each expert was used), and one layer over
time (a second waterfall, clickable). ATK: the Lab's "also record which
experts the model uses" box. **Next for it:** routing across a comparison
(the same prompt through Dolphin-Mixtral and Mixtral: re-routed, or only
re-weighted?); a "which tokens used this expert" list (click an expert,
see every token it carried — the quickest way to see what an expert is
*for*); and never-used experts across a battery (M5's pruning question).

### M29 · Trait monitors — the personas a model plays
The characters a model can play live in its activations as directions,
not in its experts: Anthropic's persona-vector work found linear directions
for traits (sycophancy, a readiness to make things up, and others) that can
be measured token by token, and that shift before the behaviour shows
(Chen et al., 2025). Athanor builds a monitor the way M2 builds a steering
direction — the mean difference of activations between contrasting
examples — but only READS it: each trait becomes a trace beside the
Waterfall, lighting up where the model starts agreeing because the user
seems to want it, hedging, turning formal, or reaching past its evidence.
For an analyst the useful ones are sycophancy (is it telling me what I
implied I wanted?) and confabulation (is it about to supply a specific it
was not given? — M3's question, as a live trace). How well each monitor
detects its trait is measured on held-out examples and shown with it.
*Boundary:* monitors for style, tone and honesty traits; never a refusal
direction (principle 9). Works on dense and MoE models alike. *Needs:* the
Tap. EXPERIMENTAL.

### M30 · Decision depth
With the logit lens (M1) at every step, each token has a depth: the first
layer at which it was already the model's answer and stayed so. Easy tokens
are decided early; the hard ones late, or only at the last layer. A trace
beside the Waterfall, and a finding across a reply: which kinds of words
needed the whole network (names, numbers, the turn in an argument), and
whether the moments of doubt are the deep ones. Cheap — it is M1's grid,
reduced to one number per token. *Needs:* the Tap, M1.
**The number BUILT 2026-10-02 (0.4.0), in the lens track.** Defined
exactly: `depth` is the first layer from which the recording's favourite
(its `ids[0]`) is the lens's top token at every layer up to the last; −1
when the last layer itself disagrees (flagged). Beside it `first_seen`
(the first layer at which the favourite is within the lens's k) and
`chosen_depth` (the same for the token the sampler took — −1 when the
sampler went off the favourite and no layer ever "decided" that token).
`athanor lens` prints the histogram of depths, how far each layer reads
(the share of tokens at which its answer is already the final favourite),
and the reply with each token's depth. **Not yet:** the trace under the
Waterfall; the live tier (depth read at record time from the final
candidates' rows alone, nothing stored — a few million multiply-adds a
token — once the probe says what the residual copy costs); the button
below.

**Calibrating the lens — one button** (designed 2026-09-29). Bill: *"Can we
automate the refinement process for the tuned lens? So that after you load
the cognitive core, you click a button to calibrate the lens?"* and *"I can
give up some time for accuracy."* The plain logit lens reads early layers
poorly (they are not yet in the output's terms); the tuned lens (Belrose
et al., 2023) fits a small translator per layer so each layer's reading
matches the model's own final distribution. The answer key is the model
itself, so calibration needs no labelled data, nothing downloaded and
nothing shipped. Local models only — an online core has no insides to
read.
- **The button**, in the Model Lab, for the model that is loaded:
  1. *Text through the model.* By default its own writing — answers to a
     fixed set of neutral prompts; optionally a folder of the analyst's
     documents, or the recorded chats. The lens is most faithful on text
     like its calibration text, so calibrating on one's own material, in
     one's own languages, is an advantage. The Tap takes every layer's
     output and the final distribution for each token (`rows="all"`).
  2. *A translator per layer*, trained so that layer's reading matches the
     final distribution (KL divergence), initialised to the identity.
  3. *A graded result.* 10% of the text held back; one chart — error by
     layer, plain lens against tuned lens, MEASURED — which is also the
     answer to "how far can the early layers be trusted?". The error falls
     as it trains; worth watching.
  4. *Kept under the model's fingerprint* (the header SHA-256 and size) in
     `<data_dir>/lenses/`, loaded automatically with the same file; a
     different file or quant is flagged uncalibrated (whether a
     calibration carries across quants of one model is itself worth
     measuring).
- **Modes** — the size is a setting, not a guess, and Thorough is the
  default:
  - *Quick* — a slim translator (low rank, a few million numbers a layer),
    scored against each token's few thousand likeliest candidates, ~20,000
    tokens: under an hour on the CPU (ESTIMATE).
  - *Thorough* — the paper's own: a full affine translator (d² + d, ~26 M
    numbers a layer at d = 5,120; ~1 billion over 40 layers), exact KL over
    the whole vocabulary (131,072), ~100,000 tokens (a bigger translator
    needs more text or it memorises). Two phases: the model reads the text
    on the card and each layer's outputs go to disk as float16 (~40 GB for
    100k tokens × 40 layers × 5,120, deleted afterwards); then the chat model
    is set aside (the host's `borrow_gpu`) and the card trains one layer at
    a time — about an hour or two (ESTIMATE). That needs a CUDA build of
    PyTorch importable by Athanor — an OPTIONAL dependency (BSD), checked
    by `capabilities`; without it, Thorough runs overnight on the CPU with
    the candidate-restricted score, and says so.
  - *Sweep* — calibrate at several sizes (rank 64, 256, 1,024, full) and
    plot the held-out error of each: where it stops falling is the size
    this model needs — the measured answer to "is a few million enough?".
- **Manners.** Phase 1 releases the engine between batches, so the host's
  chat still answers in the gaps; progress, time left and the falling error
  are shown; a stopped calibration keeps nothing half-made.
- **Needs:** the Tap (built); the output matrix dequantized — ~2.7 GB at
  float32 for a 131k × 5,120 matrix; Bill's quants carry `output.weight`
  as Q6_K, which Athanor decodes through gguf-py (llama.cpp's own, MIT,
  `pip install gguf`); the final norm's weights and epsilon from the file.
- **The same machinery trains the disposition lens** (the Reader's T2(a)):
  the target is the next K tokens instead of the next one.
- **Its limit:** a calibrated lens shows what can be read out of a layer,
  not that the model "thinks in" those words there. A better instrument,
  not a mind-reader.

### M31 · The language of thought
Multilingual models asked in one language and answering in another pass
through a third inside: on Llama 2, the logit lens shows French-to-Chinese
translation going through English in the middle layers (Wendler et al.,
2024). For ATK's Pashto and Dari work this is a direct question: when the
model translates Pashto, what language is it thinking in, layer by layer —
and do the words that come out wrong come from the layers where it was
thinking in English? The logit lens's tokens, sorted by script and
language, as a band of colour beside the Waterfall. *Needs:* the Tap, M1.

### M32 · Task drift — did a document start giving orders?
An analyst feeds a model documents nobody vetted, and a document can carry
instructions ("ignore the above and …"). Activations taken just before a
document is read and just after it differ in a recognisable way when the
text inside has pulled the model off its task — a linear probe on that
difference catches it, including injections the probe never saw
(Abdelnabi et al., 2024, "Get my drift?"; Microsoft's TaskTracker). For
ATK: a flag on the document, before the reply is trusted. Defensive,
read-only, and one of the most practical things on this list. *Needs:* the
Tap; a probe trained on examples Athanor can generate. EXPERIMENTAL.

### M33 · Hidden doubt
The Waterfall shows the doubt the model expressed — the probabilities of
what it wrote. A probe on its insides (M3) shows what it "knows" about
whether its claim is true. Where the two disagree is the interesting case:
a fluent, confident sentence the internals do not believe. Marked in the
reply like a moment of doubt, in a different colour. *From:* Kadavath et
al. (2022); Azaria & Mitchell, "The Internal State of an LLM Knows When
It's Lying" (2023). *Needs:* the Tap, M3.

### M34 · Features, and the circuits between them
The furthest reach. A sparse autoencoder turns a layer's activations into
thousands of features, many of them human-readable ("a date", "a military
unit", "the model is unsure"); a feature waterfall is the most detailed
picture of thought there is. Published dictionaries exist for some open
models (Gemma Scope for Gemma 2, Lieberum et al., 2024; Llama Scope for
Llama 3.1 8B, 2024 — each licence checked before use), and Athanor can
apply one to a GGUF of the same model, reporting how well it reconstructs
the quantized model's activations (the honest check that the dictionary
still fits). Circuit tracing — which features caused which, across layers
(Ameisen et al., Anthropic, 2025, with an open-source tracer) — is the
ceiling; it needs gradients (S6) and is recorded here as a direction, not a
promise. For Bill's own models no dictionary exists: training one on 64 GB
of RAM is possible for a layer at a time, and slow. *Needs:* the Tap; a
dictionary; S6 for circuits. EXPERIMENTAL.

### The Reader — reading what a model thinks (a programme)
Bill, 2026-09-29: *"If you were designing a tool to read what the model
thinks, what approaches would you consider. Can we get there, there must be
a way. Don't limit yourself to current research, but consider the vector of
research and anticipate the developments not yet demonstrated or
discovered, and lets try to do that."*

**The target, stated honestly.** No instrument will produce a transcript of
a thought: concepts share directions (superposition), many have no single
word, and some computation may have no words at all. What is reachable is
an **assessed account** — for any stretch of a reply, what the model was
representing and which of it drove what it wrote, each claim sourced to the
instruments that saw it and graded the way an analyst grades reporting:
*single source*, *corroborated* (two independent instruments agree),
*causally confirmed* (changing it in a sandbox changes the answer as
predicted). Two bearings make a fix; one is only a line.

**The rule that makes it honest:** every reading carries its grade, and the
view never shows a single-source reading as fact.

**Where the field is heading, and what the Reader bets on:**
- *From hand-built dictionaries to learned translators* — activations read
  out in plain language by a model trained for it (LatentQA, Pan et al.
  2024; Activation Oracles, Karvonen, Marks et al. 2025, ICML 2026 oral).
  The Reader collects the training data for such a translator as a
  by-product (verbalisations that passed a causal test are labelled pairs)
  and trains one when S6 gives gradients.
- *From token-by-token lenses to the workspace* — Gurnee et al. (Anthropic,
  2026) found a small working set, at most ~25 concepts, in the middle
  layers, carrying unspoken intermediates, plans and intentions (M19). The
  Reader's backbone is a desktop approximation of it (T2).
- *Reasoning leaving the visible text* — compressed or latent chains of
  thought. When models stop writing their reasoning, the inside is the only
  window left; the Reader reads vectors, so it does not depend on the model
  writing anything.
- *Causal evidence as the standard* — built in from the start (T7).
- *Time* — the residual stream as a signal over tokens, with slow and fast
  components, is little explored; T3 is the Reader's own bet.

Every part below is forward passes only unless it says otherwise, runs on a
GGUF through llama.cpp, and keeps to principle 9: writes happen only in
Athanor's own sandbox contexts on the loaded weights (never in a host's
replies), for measurement, never along anything refusal-related, and no
vector is kept as a steering product.

**T0 · The write path (spike S7).** Two ways to put a vector into the
stream at one layer and one position, in a sandbox context:
(a) *the one-token control vector* — decode up to the position normally;
set a control vector on layer ℓ equal to (target − the position's own
`l_out-ℓ`, measured by a dry run with the Tap); decode that single token;
clear it. Only the pinned binding's API (`llama_set_adapter_cvec`, a
`ggml_add` on each layer's output from layer 1) — the same mechanism as S6
road 1. (b) *the Tap writing* — in the evaluation callback, after the row is
computed and before the graph goes on, `ggml_backend_tensor_set` on that row:
any position, any batch, but it must be shown to be honoured on CUDA with
fusion and graph reuse. The spike proves one or both and measures the
error of each against a reference (a replaced state must reproduce the
logits the donor context gave).

**T1 · The Mirror — the model reads its own mind aloud.** Take a hidden
state from a recording (layer ℓ, token t); in a sandbox, a prompt with a
placeholder — identity (*"cat → cat; 1135 → 1135; hello → hello; ? →"*) or
descriptive (*"The thought [?] is about"*) — and put the state into the
placeholder at layer ℓ′ (T0); the model writes a few words about it. In the
player: click a token, *"What was it thinking here?"*, and read the answers
at a ladder of layers. A second or two a reading on the card (ESTIMATE).
*Limit:* the reader is the same model and can confabulate — which is why
T9 needs a second bearing. *From:* Patchscopes, Ghandeharioun et al.
(2024); SelfIE, Chen et al. (2024). The first visible payoff of the Reader.

**T2 · The workspace on a desktop.** The J-lens needs Jacobians averaged
over a thousand contexts — out of reach exactly for a 24B on one card
(M19 keeps the exact version, via S6). Three approximations, each
labelled: (a) *the disposition lens* — a per-layer translator, trained the
way the tuned lens is (the same Calibrate button, M30), but to predict the
tokens the model will write over the next K positions rather than the next
one: the regression cousin of the J-lens's "disposed to say later",
self-supervised from the model's own recordings; (b) finite-difference spot
checks (S6 road 1) along its strongest directions, to say how causal they
are; (c) the paper's own swap test — exchange a concept's component between
two prompts (T0) and see whether the answer follows it. Shown as a strip
beside the Waterfall: the few concepts held in mind at each token.

**T3 · The Demodulator.** An RF analyst's approach, not seen published in
this form (to be checked before anything is claimed). Treat each layer's
output as a multichannel signal over token time. Hypothesis: the slow part
— filtered over tens to hundreds of tokens — carries the *frame* (topic,
goal, stance, the persona being played, the plan), and the fast part the
current word and its grammar, the way a carrier carries its modulation.
Filter the recorded residuals into bands (moving averages, wavelets), read
each band through T1, T2 and T4, and show the carrier as a band beside the
Waterfall; change-point detection on it marks *"the frame shifted here"* —
from reporting to speculating, from summarising to arguing, from answering
to pleasing. A spectrogram of the leading components shows any rhythm. The
test that decides it: do the slow bands decode to the same concepts
across a reply, and do they shift where a reader would say the frame
changed? If not, it is a negative result and recorded as one. EXPERIMENTAL.

**T4 · The Atlas — the model's own concept map, built from Bill's
material.** Overnight: run a corpus (documents, recordings, the model's own
writing) through the model and stream one middle layer's outputs into
online clustering (thousands of centroids, by angle) — nothing stored but
the centroids. Label every centroid twice, independently: the local model
reads the twenty contexts nearest it and says what they share; the Mirror
verbalises the centroid itself. Where the two labels agree, the label is
trusted. Each token then shows its nearest concepts. Grows every night;
one atlas per model. The step up is a real sparse dictionary (M34) trained
from the same stream when there is data enough.

**T5 · The Compass — where the thought is heading.** For reasoning models.
Overnight, the model thinks through a few hundred varied questions; the
target is its own middle-layer representation of the answer it finally gave
(the mean over the answer's tokens — no outside embedder); a linear map
(ridge regression, closed form) from each position of the thinking to that
target. In the player, a needle: how closely the thought at each token
points at the answer it would give, and at alternatives (M15, M25). The
lock-in point is when the destination was fixed. **The faithfulness meter:**
lock-in long before the written reasoning "arrives" says the rest was
justification, not route — checked by M26's early answer (cut the
thinking at the lock-in and see whether the answer is already the same)
and by T7. *From:* Future Lens, Pal et al. (2023); Lanham et al. (2023);
"Reasoning Models Don't Always Say What They Think", Chen et al.
(Anthropic, 2025).

**T6 · Background subtraction.** The same prompt twice with one thing
changed (*"the source is reliable"* / *"unreliable"*); subtract the two
recordings layer by layer: where the difference enters, where it grows,
where it is squashed, and — read through T1/T2/T4 — what it means. The same
subtraction between a fine-tune and its base on the same text shows what
the fine-tune changed inside. Two recordings and a subtraction: the
cheapest instrument here.

**T7 · Fault injection — what actually drove the answer.** In a sandbox,
remove one concept's component (projected out at a layer and position, T0),
or resample one sentence of the reasoning and let the rest follow (Thought
Anchors, Bogdan et al., 2025), and measure what happens to the answer. This
is how a reading earns *causally confirmed*. Measurement only; principle 9.

**T8 · Introspection calibration — does this model know its own mind?**
After Anthropic's concept-injection experiments (Lindsey, 2025): put a known,
neutral concept into the stream (T0 — "the ocean", "shouting", "bread") and
ask the model whether it notices an injected thought and what it is about;
compare with trials where nothing was injected. The result is a score, per
model and per layer, for how far its self-reports can be trusted — and then
its self-reports (*"what were you thinking when you wrote this?"*) become
hypotheses for the other instruments to test.

**T9 · The assessed account — the product.** Fusion of all of the above
into one view per recording: the concepts the instruments agree on, graded;
where the frame shifted (T3); when the destination locked (T5); and where
the inside and the written reply diverge — M33's hidden doubt, generalised.
In words, the way an assessment is written: *"We assess with moderate
confidence that from token 120 the model represented the source as
unreliable (workspace and atlas agree; the swap test flips the conclusion)
while its text stayed neutral."*

**Order within it:** S7 → T1 (the first visible payoff) → T6 (cheap) → M30
and T2(a) (one Calibrate button) → T5 → T4 → T7 and T8 → T3 (needs the
decoders) → T9. Each ships on its own and says what it could not do.

**Implementation notes** (worked out 2026-09-29, for whoever picks it up):
- *T0(a) in detail.* `llama_set_adapter_cvec(ctx, data, len, n_embd,
  il_start, il_end)`: `data` is n_embd × n_layer floats starting from layer
  1 (there is no layer 0 slot: `llama_adapter_cvec::apply` skips it), so
  injection is possible at layers ≥ 1; set `il_start = il_end = ℓ`, only
  that layer's slice non-zero; `data = NULL` clears it. It is added to the
  layer's output (`build_cvec`, just before `l_out-ℓ` is named), for every
  token of the decode — hence the one-token decode. The spike must show
  that changing the vector between decodes is honoured under graph reuse
  (`LLAMA_GRAPH_REUSE_DISABLE` is the fallback) and measure the error of a
  full replacement against a donor context's logits.
- *Where a state to read comes from.* Either a recording made with the
  residual Tap at the layers wanted (float32: 20 KB per layer per token at
  d = 5,120; float16 halves it), or — cheaper, on demand — a replay: a
  sandbox context reads the recording's prompt ids and chosen ids up to
  the token, the prompt as one batch and the reply one token at a time,
  the shape it was generated in (re-scoring in one batch rounds
  differently: the Waterfall's own note), with the Tap on. Not possible for
  a vision prompt (its ids are not kept).
- *T1's layers.* Source layer ℓ and target layer ℓ′ are both sweeps;
  Patchscopes found early target layers decode best. The prompt set
  (identity, description, "is it about…" yes/no questions) is Athanor's own
  and versioned, so readings stay comparable.
- *T2(a)'s data* is the Calibrate button's: the same activations with a
  K-token-ahead target.
- *T3's data.* Every fourth layer's output over a reply — about 200 KB a
  token at float32 on a 24B; bands are computed after the fact from the
  recording, so the filters can change without recording again.
- *T4.* Mini-batch spherical k-means (K = 4,096 to start) on one layer
  about 60% of the way up; the labelling prompt shows the model the 20
  nearest contexts with the token marked; the Mirror's label is taken from
  the centroid alone; agreement is scored by the model itself as a yes/no
  and by overlap of the words.
- *T5.* A few hundred questions × ~1,000 thinking tokens ≈ 300,000
  positions; ridge regression over 5,120 dimensions is one 5,120 × 5,120
  solve (numpy, seconds); held-out questions for the grade.
- *T7.* Projecting a direction out at every position means stepping one
  token at a time with a new vector each step (T0a), or the Tap writing
  (T0b) — slow but mechanical; overnight for a long reply.
- *T8.* Concept vectors are M2's mean differences for neutral concepts at
  one layer; half the trials inject, half do not; the score per layer is
  the hit rate minus the false-alarm rate, with the names the model gives.
- *Storage* follows decision 7 (retention); the big items (residual
  recordings, calibration activations) are deleted when their product is
  made unless the analyst keeps them.

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
  token. Gates M1–M3. **Built 2026-09-29**, exact on CPU; Windows / CUDA:
  `athanor tap probe MODEL` on Bill's card is the last step of the spike.
- **S2 the tools** — build `llama-perplexity`, `llama-bench`,
  `llama-tokenize`, `llama-quantize`, `llama-imatrix`, `llama-gguf-split`
  from the llama.cpp tree install.bat already compiles (b11093); install.bat
  build time +?. Gates the cross-checks and M5–M7.
- **S3 determinism** — are logits bit-identical run to run on the card, at a
  fixed batch? Gates M10, and says how to cache references.
- **S4 vocab-only speed** — on Windows, for Bill's 14–18 GB files.
- **S5 the writer** — native or pinned gguf-py (MIT); round-trip a real file
  byte-identical before trusting it with surgery.
- **S7 the write path** — put a chosen vector into the residual stream at one
  layer and one position, in a sandbox context: the one-token control
  vector, and the Tap writing from its callback (T0). Is the replacement
  exact on CPU and on CUDA? Gates the Reader's T1, T2(c), T7 and T8.
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

**Phase 2 — behaviour** (one model): **the Waterfall's recorder and player
are built** (0.2.0, brought forward at Bill's request). Next in Phase 2,
in order: Surprise (how every later change is judged, and the same
machinery as the after-the-fact recording); the Waterfall's branching
("take this instead") and its comparison view; end pressure; then Next
Token, Context, Speed, Structure, Override. M27's canaries come with the
comparison view and S3; M25 and M26 as soon as branching and the cache
fork exist — both need only logits, no Tap.

**The Tap (S1) is the gate for M1, M3, M9, M16, M19, M20 and M28–M34** —
the whole of "watching the inside". **Built 2026-09-29 with M28's expert
map.** Next on it: the probe on Bill's card (the spike's last answer), then
M30 (the logit lens, reduced — the lens is already proved against the
logits by the probe; what remains is the output matrix dequantized in
chunks, and the view), then M20's R0. **2026-10-02: the Tap ran in ATK on
the 30B-A3B (numbers above); M1's engine and M30's number are built
(0.4.0, `athanor lens`) — the view remains.**

**The Reader** (2026-09-29, §4) builds on the Tap: S7 first, then the
Mirror (T1) as its first visible payoff; its order is in its own section.

**Where to pick up** (parked 2026-09-29 while Bill works on other parts of
ATK — *"I promise, we will revisit it"*; resumed 2026-10-02 with the Tap's
first run in ATK and the lens engine), in order:
1. **The probe on Bill's card** — the last step of S1, still to run. From
   the Athanor folder: `..\ATK\envs\atk_core\Scripts\python.exe -m athanor
   tap probe "D:\Analyst_Toolkit\Models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"`,
   and once with a dense one (Magistral). It answers: does computing the
   graph in pieces on CUDA change the last bits, what does the Tap cost
   idle and copying, does the lens reproduce the logits there. Bill pastes
   the output; the numbers go into the Tap section above (the ATK run's
   cost and MoE figures are already there).
2. **The first real lens** — `athanor record <model> --prompt … --lens`
   (or ATK's Lab once it has the residual box) on the 30B-A3B, then
   `athanor lens <recording> --step N`: the first look at decision depth
   on Bill's own model and question, and the agreement-by-layer curve that
   says how badly the plain lens reads the early layers there — the
   measured case for the Calibrate button.
3. **The picture (M9)** in the player: the grid, the depth trace, the
   column at the cursor; the residual box in ATK's Lab.
4. **S7 and the Mirror (T1)** — the write path proved, then "What was it
   thinking here?" in the player.
5. **The Calibrate button** (Thorough by default, the Sweep) — and T2(a)
   on the same machinery; per-layer attribution (what each layer ADDED,
   attention vs MLP) on the same track.
6. Then the Reader in its own order (T6, T5, T4, T7/T8, T3, T9).

**Next, concretely, outside the Reader** (2026-09-29): Phase 1.1 (numbers, slack, variants, the
integrity ledger) and the Phase 1 tabs' widgets in the Lab — Inspect,
Tokenize, Compare, Template are built as engines but not yet as pages; then
Surprise; then "take this instead".

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
- **The MoE perspective persona, pass by pass.** ATK's "MoE perspective"
  methodology (not to be confused with an MoE model) answers in several
  passes — perspectives, then a synthesis. Recorded, each pass is its own
  recording, grouped as one session, so the Waterfall shows where each
  perspective was sure and where the synthesis sided with one of them.
- **Confidence on extracted entities.** ATK's extraction asks the model for
  JSON; recorded, every value in it has the probabilities of the tokens that
  spelled it. A person, a place or a date on the network graph can then
  carry how sure the model was when it wrote it down — beside the existing
  "model-generated" label, and useful for sorting a large graph by what to
  check first. The recorder is the same; the new part is mapping a JSON
  value's characters back to its steps (the recording's spans already do).
- **Recordings as evidence.** ATK keeps a chain of custody (O.W.L.). A reply
  recorded for the Waterfall can be entered in it: the recording's SHA-256
  (already in its meta file), the model file's identity and the settings —
  so what a model said, and what it was choosing between when it said it,
  can later be shown not to have changed. On the analyst's click, not by
  default.
- **The Tap in ATK** (built 2026-09-29): the Waterfall page's box
  "⚗ also record which experts the model uses" (`settings["lab"]
  ["tap_experts"]`); `ReplyRecording(tap="experts")` installs the Tap
  inside the engine's lock on the first recorded reply (never on a dense
  model), unticking takes it out on a worker (`release_tap` →
  `LLMEngine.run_on_model`, which waits for the engine's lock), and the
  transcript's recording note says what the Tap did (`tap_note`).
- **Calibrate the lens** (M30, not built): a button on a Lab page for the
  loaded local model; a long job that yields the engine between batches;
  Thorough's training phase borrows the card through `borrow_gpu` (unload
  the chat model, train, restore). Greyed, with the reason, for an online
  core.
- **The Reader** (§4, not built): "What was it thinking here?" on a reply's
  word in the Waterfall (T1); the workspace strip and the carrier band
  beside it (T2, T3); the assessment (T9) as a panel a case can cite, with
  the recording's hash for the chain of custody.
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
7. **Recording retention.** A recording is about 2 KB per token plus its
   meta file (a 1,000-token reply: a few MB). Keep everything, or keep the
   newest N (or N days) and anything the analyst has noted or entered in
   O.W.L.? Until decided: everything is kept.
8. **The canary prompts** (M27) should be yours: which five prompts stand
   for the work you actually do?
9. **Calibration text** (M30): the model's own writing by default — and
   which of your folders (documents, recorded chats), in which languages,
   should a calibration read when you point it there?
10. **PyTorch as an optional dependency** — for Thorough calibration and,
    later, the Reader's trained parts on the card (BSD licence, so the rule
    is met). Only when present; nothing needs it. Yes?
11. **The Reader's write path** (T0) — sandbox-only, measurement-only,
    never refusal-related (principle 9, as amended 2026-09-29). Confirm when
    we resume.

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
- Jiang et al., "Mixtral of Experts" (2024), arXiv 2401.04088; Wang et al.,
  "The Myth of Expert Specialization in MoEs: Why Routing Reflects
  Geometry, Not Necessarily Domain Expertise" (2026), arXiv 2604.09780;
  "The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models
  at Expert Level" (ICML 2026), arXiv 2604.02178. For M28.
- Chen et al., "Persona Vectors: Monitoring and Controlling Character
  Traits in Language Models" (Anthropic, 2025), arXiv 2507.21509. For M29.
- Wendler et al., "Do Llamas Work in English? On the Latent Language of
  Multilingual Transformers" (2024). For M31.
- Abdelnabi et al., "Get my drift? Catching LLM Task Drift with Activation
  Deltas" (2024), arXiv 2406.00799. For M32.
- Azaria & Mitchell, "The Internal State of an LLM Knows When It's Lying"
  (2023). For M33.
- Lieberum et al., "Gemma Scope" (2024); He et al., "Llama Scope" (2024);
  Ameisen et al., "Circuit Tracing: Revealing Computational Graphs in
  Language Models" (Anthropic, 2025). For M34.
- For the Reader: Ghandeharioun et al., "Patchscopes" (2024), arXiv
  2401.06102; Chen et al., "SelfIE" (2024); Pan et al., "LatentQA" (2024),
  arXiv 2412.08686; Karvonen, Marks et al., "Activation Oracles" (2025),
  arXiv 2512.15674; Gurnee et al., "Verbalizable Representations Form a
  Global Workspace in Language Models" (Anthropic, 2026); Pal et al.,
  "Future Lens" (2023), arXiv 2311.04897; Lanham et al., "Measuring
  Faithfulness in Chain-of-Thought Reasoning" (2023); Chen et al.,
  "Reasoning Models Don't Always Say What They Think" (Anthropic, 2025),
  arXiv 2505.05410; Bogdan et al., "Thought Anchors" (2025), arXiv
  2506.19143; Lindsey, "Emergent Introspective Awareness in Large Language
  Models" (Anthropic, 2025).
- Kuhn, Gal & Farquhar, "Semantic Uncertainty" (2023); Farquhar et al.,
  "Detecting hallucinations in large language models using semantic
  entropy", Nature (2024); Kossen et al., "Semantic Entropy Probes" (2024),
  arXiv 2406.15927. For M25.
- Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning"
  (2023); "Dynamic Early Exit in Reasoning Models" (DEER,
  2025), arXiv 2504.15895; "Answer Convergence as a Signal for Early
  Stopping in Reasoning" (2025), arXiv 2506.02536. For M26.
- Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV
  Cache" (2024). For 3.13.
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
