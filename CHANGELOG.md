# Changelog

Until 1.0, anything in the public API, the command line or the result
formats may change. Every change is listed here.

## 0.4.0 — 2026-10-02 — the lens (M1, M30)

- **The lens** (`athanor.lens`, `api.run_lens`, `athanor lens RECORDING`):
  a recording made with the residual stream (`--tap residual`) read layer
  by layer — each layer's `l_out-N` through the model's own final norm and
  output matrix, as a full distribution over the vocabulary: what the model
  would say if it stopped at that layer (the logit lens, M1). Kept per step
  and layer in a derived track beside the recording (`.athrec-lens` +
  `.athrec-lens-meta`, `docs/formats/recording.md`): each layer's top-k,
  the eventual favourite's and the chosen token's rank and log-probability
  at every layer, the entropy; and per token the **decision depth** (M30) —
  the first layer from which the favourite is the lens's answer at every
  layer up to the top — and the first layer at which it appears among the
  k. `Recording.lens` reads it (`depth`, `fav_logprob` [tokens, layers],
  `column(step)`, `agreement_by_layer()`); `athanor recording` and the
  listing say when a recording has one.
- **It checks itself.** The last layer's reading is compared with the
  recording's own logits at every step; the result is MEASURED when the
  favourite agrees at ≥ 98 % of steps, EXPERIMENTAL otherwise, with the
  agreement and the log-probability errors recorded either way. Proved on
  llama.cpp's own tapped recordings of the test models (agreement 100 %,
  errors ~1e-6).
- **The unembedding** (`athanor.lens.unembed`, `api.build_unembedding`):
  `output_norm` (RMS, or LayerNorm with its bias), its epsilon,
  `output.weight` or the tied `token_embd.weight`, `output.bias` and
  Gemma's logit soft-cap, from the GGUF; the matrix dequantized to float32
  in row ranges and kept under `<data>/lenses/<header-sha256-prefix>-<size>/`,
  memory-mapped on use. A damaged cache is rebuilt, not trusted. The model
  file is needed once; a file that is not the recording's is refused unless
  named (`--model`), and then noted.
- **K-quants decoded natively** (`athanor.gguf.dequant`): Q4_K, Q5_K and
  Q6_K — the type every K-quant file keeps its output matrix in — plus
  Q5_0 and Q5_1, in numpy, following ggml's `dequantize_row_*` and tested
  against a transcription of its loops; bit-identical to gguf-py's.
  `dequantize_row_range` reads a run of rows in one piece.
  `can_dequantize(type)` says whether a type needs gguf-py.
- The lens meta carries the **text of every token the layers name** that the
  recording's own pieces lack (`pieces`, looked up through llama.cpp's
  tokenizer when the binding is present, else decoded from the file's
  token list — `athanor.lens.pieces`), so a column reads as words, not ids;
  `athanor lens --step` looks up anything still missing on the spot.
- `athanor record --lens [--lens-k N]`: record with the residual stream
  and run the lens on the recording straight after.
- Recordings: `stems()` knows the lens suffixes; `Recording.summary()` and
  `list_recordings` carry a `lens` entry.
- Measured on Bill's card (2026-10-02, ATK, Qwen3-Coder-30B-A3B Q4_K_M, the
  experts in RAM and attention on an RTX 3080 Ti): the Tap's experts preset
  cost 9.4 ms a token (144 copies) on a reply at 19.1 tokens/s; 291 steps,
  none flagged; the chosen experts were the router's top-8 at all 13,968
  layer-steps.
- **Where data goes.** Run from a source checkout (or an editable install
  of one), Athanor now keeps its data in `<checkout>/data` — beside its
  code, on the code's drive — instead of the platform's per-user folder,
  which on Windows put recordings under `C:\Users\…\AppData` unasked
  (2026-10-02). `ATHANOR_DATA` / `--data` still come first; a host's
  `data_dir` still replaces all of it; only a package installed into
  site-packages that was told nothing falls back to the per-user folder.
  `athanor data` prints the folder and which rule chose it;
  `host.data_dir_choice()` is the rule.
- `API_VERSION` 0.3.

## 0.3.0 — 2026-09-29 — the Tap (spike S1)

- **The Tap** (`athanor.tap`, `api.make_tappable`): a model's own tensors,
  copied out while llama.cpp computes them — each layer's output
  (`l_out-N`), the final norm, the logits, and a mixture-of-experts model's
  routing (`ffn_moe_topk-N`, `ffn_moe_weights-N`, `ffn_moe_probs-N`). It
  uses llama.cpp's evaluation callback (`cb_eval`) and five of ggml's
  functions, bound here (the binding binds none); the start of
  `struct ggml_tensor` is read directly and checked by a self-test at load,
  so a ggml that moved a field turns the Tap off with a reason instead of
  reading the wrong bytes. `make_tappable` rebuilds a `Llama`'s context once
  with the dispatcher in it (weights shared, the cache empty); `capture`,
  `watching` and `tensor_names` read it. A failure inside the Tap is kept
  and never reaches llama.cpp.
- **Proved on the test models (CPU)**: tapped and untapped logits
  bit-identical, idle and while copying; the Tap's copy of `result_output`
  IS llama.cpp's logits; the last layer's output, normed and multiplied by
  the output matrix in Athanor, reproduces them (the logit lens); each
  layer's chosen experts are the router's top-scoring ones, their weights
  the router's values.
- **`athanor tap probe MODEL`** (`api.tap_probe`): all of that, measured on
  the machine it runs on — exactness plain vs tapped, the lens, the expert
  check, and milliseconds per token plain, tapped-idle and copying each
  preset. `athanor tap names MODEL` lists what one forward pass computes.
- **The Tap in recordings**: `attach(llm, tap="experts")` (or `"residual"`,
  `"logits"`, patterns), `athanor record --tap …`. A third file,
  `.athrec-tap`, one fixed-size record per token, aligned with the
  Waterfall's steps and described by the meta file's `tap` block
  (`docs/formats/recording.md`). `Recording.tap` reads it;
  `.experts()` gives the routing (`ids`, `weights`, `probs`, `share()`,
  `usage()`, `grid()`). A model that was not made tappable is recorded
  without, and the recording says why.
- **The expert map** (M28) in the player: layers × experts at the cursor
  (the router's score in dB, the experts used framed with how much each
  counted), how often each was used over the reply, its thinking or its
  answer, and one layer over time — a second waterfall that scrolls with
  the cursor. `ExpertPanel` / `ExpertMap` on their own.
- **`tiny_model(n_experts=…)`** writes a mixture-of-experts test model.
- `capabilities`: `eval_callback` is now available when the Tap will run.
- ATK: the Model Lab's "also record which experts the model uses" box.

### Fixed before release, from an independent review

- A context that could not be rebuilt (both the tapped one and the plain
  fallback refused, e.g. the memory taken in the gap) left the `Llama`
  holding a freed context, and the next generation crashed the process. It
  now holds a stand-in that raises `TapUnavailable` ("reload the model").
- A batch longer than `n_ubatch` runs the graph once per micro-batch; the
  Tap kept only the tensors whose row count matched the whole batch, so
  every layer but the last was dropped and the recording's layout fixed
  without them. Each micro-batch is now recognised (by the graph's first
  node) and its rows stitched back into batch order.
- The token axis was guessed from a tensor's name, wrong for 3-D tensors
  such as `Qcur` after its reshape. It now comes from the shape against the
  micro-batch's token and output counts, with llama.cpp's known layouts as
  the tie-breaker; a name llama.cpp gives to several nodes keeps the first,
  and says so.
- The expert map threw on a step whose weights were unknown (NaN) when no
  router scores were recorded; such cells now draw, and say the share is
  unknown.
- An error inside `attach` after the Tap was attached could leave its
  session and decode wrapper on the model; the Tap is now set up inside the
  block that always cleans up.
- A Tap error mid-recording marked later steps "nothing captured"; they are
  now marked "stopped", with the error and the step it stopped at.
- The player read a recording's whole Tap file to list what it held; it
  now reads the meta file's layout, and the Tap file only for the experts.
- The probe's lens check could pair a layer's rows with the wrong step's
  logits when a row was missing.

## 0.2.0 — 2026-09-29 — the Waterfall

- **The recorder** (`athanor.waterfall`, `api.attach_recorder`): every token
  a `llama_cpp.Llama` samples, recorded with the distribution it was chosen
  from — the k most probable tokens (256 by default), the probability left
  over, the entropy, the token taken with its exact rank and probability,
  and the time. It reads llama.cpp's own logits after the sampler has
  picked, so a recorded reply is the same reply, token for token (tested
  against an unrecorded run with the same seed). A failure costs the
  recording, never the reply. One reply carried on in several parts is one
  recording (`recorder=`).
- **The format**, `.athrec-meta` + `.athrec-data`: SigMF-shaped, fixed-size
  records, readable with numpy alone (`docs/formats/recording.md`).
- **The player** (`athanor.gui`, PySide6, optional): the reply along the top
  (the token at the cursor lit, the rest dimmed or hidden, the words the
  model was unsure of underlined, hover for how sure); the waterfall (time
  down, candidates across, probability in dB, entropy beside it); **taken**
  — the token the model actually wrote, on every row, whatever its rank;
  **Aa read**, which writes each candidate in its cell; and the transport,
  with ◆ to jump between **moments of doubt** (`Recording.doubts()`).
  `WaterfallPanel` is the whole tab as one widget; `python -m athanor.gui`
  opens it in a window.
- **Command line**: `athanor record` (load, generate once, record) and
  `athanor recording` (summarise one).
- **`api.tiny_model(path)`**: a llama-architecture GGUF with random weights,
  written on the spot, that llama.cpp really runs — so a host can test its
  integration without downloading a model. Nothing ships with a model.
- ATK is the first host: Chat's **⚗ record** box, and the Model Lab's
  Waterfall tab.

## 0.1.0 — 2026-09-28 — Phase 1: the file

First release. Everything here reads files and tokenizers; no weights are
loaded.

- **GGUF reader and writer**, Athanor's own. The reader takes every key and
  every array, and every tensor's name, shape, type and place. The writer
  creates new files only, laid out as llama.cpp's writer does. A header read
  and written again is byte-identical (spike S5). Checked on llama.cpp's 19
  vocabulary files (GGUF v2 and v3), on files written by gguf-py, and on
  big-endian files.
- **Vocab**: llama.cpp's tokenizer through llama-cpp-python, vocab-only. Its
  load log is captured, not printed, using b11093's log levels. (The
  binding's own logger at ea3b56b still maps the old order, so with
  `verbose=False` it prints llama.cpp's warnings and hides its errors.)
- **Inspect**: anatomy, and tokenizer health checks against the header and
  against llama.cpp: pre-tokenizer, vocabulary vs embedding rows, name vs
  header, the model's own chat template (markers, the end of a reply, BOS),
  llama.cpp's own warnings, and projector pairing.
- **Tokenize**: the context ruler (per model, per script), worst-split
  analyst strings, and a fertility table.
- **Compare**: tokenizer, metadata and tensor differences; lineage from
  sampled embedding rows (F32, F16, BF16, Q8_0, Q4_0 and Q4_1 decoded
  natively, other types via gguf-py); and a vocabulary overlap matrix.
- **Template**: llama.cpp's 55 chat formats, ported from ATK and held byte
  for byte to llama.cpp's C++. A render is tokenized and classed token by
  token, and three mix-and-match checks run on it: markers that are not
  special tokens, a reply end llama.cpp will not stop on, and BOS twice. Two
  templates can be compared side by side.
- **Notebook**: append-only JSON Lines, with notes.
- **Host port** (`NullHost` by default), **capability probe**, **command
  line** with `--json` and stable exit codes, and the **public API**
  (`athanor.api`).
- **Labels**: MEASURED, DECLARED, ESTIMATE, EXPERIMENTAL. DECLARED is new
  since the plan: what a file says about itself, as distinct from anything
  measured.
- Docs: `docs/INTEGRATING.md` (every example runs in the tests),
  `docs/QUICKSTART.md`, `docs/formats/`.
- **The log** (`athanor.log`): every run, every vocabulary load with
  llama.cpp's complete log, breadcrumbs flushed to disk before native calls,
  and a crash log from `faulthandler`, all local (`docs/formats/log.md`).
  Bill: *"a comprehensive logging setup to ensure we learn the most
  possible."*

### Fixed before release, from an independent review

- The writer used the default alignment when `general.alignment` came in as
  a key, so it made files llama.cpp refuses. The key now sets the layout
  and is checked (UINT32, a power of two). ggml's own reader is now a test
  oracle for every file Athanor writes.
- `write_file` could overwrite a file that appeared during the write, or
  reuse someone else's `.part` file. The part file is now created
  exclusively, published by an operation that fails if the name exists, and
  removed on failure.
- Extra embedding rows were called harmless padding. llama.cpp refuses any
  row count but one per token, so any mismatch (embedding or output) is now
  a problem.
- Word counts dropped combining marks, splitting voweled Arabic and
  Devanagari words at every mark. Words are now built from Unicode
  categories.
- A token id outside the vocabulary made llama.cpp end the whole process.
  Ids are now checked first.
- Template checks raised false alarms. Llama 2's and Mistral v0.1's bracket
  markers are now judged by the format that uses them. The space before
  `</s>` is formatting, not the turn end. Templates that refuse a system
  message render without one. BOS is counted the way llama-cpp-python,
  llama.cpp's server and a naive runner each build the prompt.
- Loading a vocabulary replaced a host's llama.cpp logger with the binding's
  default. The previous logger is now put back.
- A token that is not UTF-8 made `--json` fail and lose the result. Such
  strings are now written as visible escapes, and a notebook failure never
  costs the printed result.
- stdin, conversation files, string files and template files are read as
  UTF-8 with any byte-order mark dropped. Short or malformed files are exit
  3; a damaged length fails fast instead of reading the file into memory.
- Lineage matches rows by token, so a fine-tune that added tokens is
  recognised as the same base.
- Embedding models are no longer held to chat checks. An explicit `default`
  pre-tokenizer is no longer reported as a missing one. A corpus cut short
  says so.

### Fixed before release, from a second review

- A closed `Vocab` could still call into llama.cpp after its model was freed.
  Every call now checks, and a closed vocabulary raises `RuntimeError`.
- Library templates were judged by the format `detect()` guessed, not by the
  name they were asked for, so Llama 2's bracket markers raised a false
  alarm again. The name now decides.
- The log kept the text given on the command line. `--text` values and
  notebook notes are now logged as a length and a hash, and the crash log
  has no command line.
- The writer's alignment can no longer be changed after keys are added.
- Any unpaired surrogate in a token or a path is written as a visible escape.
- Thai, Lao, Khmer, Myanmar and Tibetan write no spaces between words, so
  they are measured per character, not per "word".
- The BOS text is always given to the template; each runner's count decides
  whether it appears twice.
- If the final rename is refused, the writer copies to the name exclusively
  instead, so it still never overwrites.
