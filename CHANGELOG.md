# Changelog

Until 1.0, anything in the public API, the command line or the result
formats may change. Every change is listed here.

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
