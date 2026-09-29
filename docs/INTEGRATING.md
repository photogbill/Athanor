# Using Athanor in your own program

Athanor is built to be put inside other software. Nothing in it needs the
Analyst Toolkit (ATK) or any other host: ATK is simply its first user, and
uses only what is described here.

This guide covers Athanor **0.1** (Phase 1: the file tabs — Inspect,
Tokenize, Compare and Template). Every Python example in it runs in the test
suite (`tests/test_docs.py`), so the examples cannot drift from the code. In
them, `MODEL` and `MODEL_B` are paths to GGUF files and `DATA_DIR` is a
scratch folder.

**Contents**

1. [What you get](#1-what-you-get)
2. [Installing](#2-installing)
3. [The Python API](#3-the-python-api)
4. [The command line, with `--json`](#4-the-command-line-with---json)
5. [The file formats](#5-the-file-formats)
6. [Writing a host](#6-writing-a-host)
7. [Reading results: labels and findings](#7-reading-results-labels-and-findings)
8. [Threads, versions and the promise](#8-threads-versions-and-the-promise)
9. [When something is missing](#9-when-something-is-missing)
10. [The log](#10-the-log)

---

## 1. What you get

There are five ways in. Use as many or as few as you need:

| way in | for | status |
|---|---|---|
| the Python API, `athanor.api` | Python programs | **0.1** |
| the command line, `python -m athanor … --json` | programs in any language | **0.1** |
| open file formats (`docs/formats/`) | reading Athanor's output with no Athanor | **0.1** (notebook, results); recordings arrive with the Waterfall |
| the host port, `athanor.host` | an application lending its GPU, folders and data folder | **0.1** |
| Qt widgets, `athanor.gui` | PySide6 applications | next |

What Athanor does in 0.1, all without loading a model's weights:

* **Inspect**: a GGUF's anatomy (every figure labelled), its tokenizer as
  the header declares it and as llama.cpp actually loads it, and health
  checks — a missing pre-tokenizer, too few embedding rows, a name that
  claims a context the header doesn't, a chat template whose markers are not
  real special tokens in this vocabulary, a reply ending llama.cpp will not
  stop on, BOS twice.
* **Tokenize**: the context ruler. One text, several models, and how many
  words of *that* material a context holds on each. It also finds the
  analyst's strings (IPs, MGRS, names, hashes) that split worst.
* **Compare**: two files' tokenizers, metadata and tensors, and lineage
  (whether they are built on the same base, from their embedding rows). It
  also gives a vocabulary-overlap matrix across a library.
* **Template**: any of llama.cpp's 55 chat formats, a model's own, a file
  or Jinja text, rendered and seen through the model's tokenizer.

## 2. Installing

```text
pip install git+https://github.com/photogbill/Athanor
```

Or from a local clone, where your changes take effect immediately:

```text
pip install -e path/to/Athanor
```

Athanor needs **numpy** and **jinja2**. Both are installed automatically.

**llama.cpp: bring your own.** The tokenizer work goes through
`llama-cpp-python`, the Python binding for llama.cpp. Builds differ (CPU,
CUDA, Metal, a pinned commit), so Athanor does not install one. If you don't
have one yet:

```text
pip install llama-cpp-python        # a CPU build from PyPI
```

If you *do* have one (for example a CUDA build you compiled yourself), do
**not** install `athanor[llama]`. That extra pulls the PyPI build and would
replace yours. Athanor is written and tested against llama.cpp **b11093**
through llama-cpp-python commit `ea3b56b`. See what your build can do with:

```text
python -m athanor capabilities
```

**Optional:** `pip install gguf` (llama.cpp's own gguf-py, MIT) lets Compare
decode K-quant and i-quant embedding rows for its lineage check. Everything
else works without it.

**Offline** — Athanor never downloads anything at run time, and never sends
anything anywhere. Installing is the only connected step.

## 3. The Python API

Everything public is in `athanor.api`, and `api.__all__` is the list.
Results are plain dicts and lists that `json.dumps` can write (use
`api.dumps`, which also handles numpy values).

### Inspect a file

```python
from athanor import api

report = api.inspect(MODEL)
print(report["anatomy"]["architecture"])        # {'value': 'phi3', 'label': 'DECLARED'}
for f in report["findings"]:                    # problems first, then warnings…
    print(f["status"], f["check"], f["message"])
assert report["kind"] == "inspect"
```

`inspect(path, run_llama=False)` reads the header only and never touches
llama.cpp. The checks that need llama.cpp come back `skipped`, and say so.

Every key in the header, arrays shortened for display:

```python
from athanor import api

rows = api.metadata(MODEL, max_items=4)
tokens = next(r for r in rows if r["key"] == "tokenizer.ggml.tokens")
print(tokens["length"], tokens["preview"])
```

### How much of my material fits

```python
# needs: llama
from athanor import api

text = "The meeting moved to Thursday. Встреча перенесена на четверг."
result = api.tokenize([MODEL, MODEL_B], text, context=32768)
for m in result["models"]:
    ruler = m["ruler"]
    print(m["model"]["name"], m["tokens_per_word"], "tokens/word;",
          ruler["words"]["value"], "words fit", f"({ruler['words']['label']})")
```

`api.read_corpus(path)` turns a text file, or a folder of them, into one
string for this. `api.worst_splits(paths)` runs the built-in analyst strings
(`api.ANALYST_STRINGS`), or your own:

```python
# needs: llama
from athanor import api

r = api.worst_splits([MODEL], [("callsign", "KD2ABC"), ("grid", "18SUJ2337106519")])
for row in r["strings"]:
    print(row["string"], [m["n_tokens"] for m in row["models"]])
```

### Compare two files

```python
from athanor import api

diff = api.compare(MODEL, MODEL_B)
print(diff["tokenizer"]["identical"], diff["tokenizer"]["overlap"]["value"])
print(diff["lineage"].get("verdict"), "-", diff["lineage"].get("why", ""))
matrix = api.overlap_matrix([MODEL, MODEL_B])
print(matrix["identical_vocabulary"])
```

Lineage compares the same token's embedding row in both files (64 rows by
default, spread across the vocabulary). The cosine figures are MEASURED. The
verdict ("same base", "not the same weights") is an ESTIMATE, and the
thresholds behind it are stated in the result.

### Look at a chat template through a model's tokenizer

```python
# needs: llama
from athanor import api

r = api.template("chatml", model=MODEL)       # or "gguf" (the model's own), a file, Jinja text
print(r["render"])
for seg in r["segments"]:                     # the render as markers and text
    if seg["kind"] != "text":
        print(seg["kind"], repr(seg["text"]))
for f in r["findings"]:
    print(f["status"], f["message"])
```

How the render is made: the model's BOS text goes to the template only when
the model's tokenizer is configured to add a BOS. The render is then
tokenized with special-token parsing on and add-special off, as
llama-cpp-python's chat path does. `r["bos"]` counts the BOS tokens each
kind of runner would send: llama-cpp-python, llama.cpp's server (which strips
a duplicate), and a naive runner.

Without a model, a template still renders. You get the text and the marker
list, but no tokens:

```python
from athanor import api

r = api.template("llama3")
print(r["render"][:80])
print([m["marker"] for m in r["markers"]])

text = api.render_template(api.resolve_template("mistral-v7")["text"],
                           [{"role": "user", "content": "Hello"}])
print(text)
print(api.detect_template("{{ '<|im_start|>' }}"))   # 'chatml' — llama.cpp's own detector, ported
```

### The tokenizer itself

`Vocab` is llama.cpp's tokenizer for one file, loaded with `vocab_only`: no
weights and no GPU, typically a second or less even for a large model.

```python
# needs: llama
from athanor import api

with api.Vocab(MODEL) as v:
    ids = v.tokenize("Hello <|end|>", parse_special=True)
    print(v.type, v.n_tokens, ids, [v.piece(i) for i in ids])
    print(v.special_ids()["eos"], v.attr_names(ids[-1]))
    print(v.load_seconds, v.log_lines())       # what llama.cpp warned about while loading
```

llama.cpp's messages during a load are captured, not printed. That keeps
your console clean, and they are evidence: some problems are announced there
and nowhere else. `with v.capture() as log:` does the same around any call.

### Keep a record

```python
from athanor import api

nb = api.Notebook(DATA_DIR / "notebook.jsonl")
run = api.record(api.inspect(MODEL, run_llama=False), notebook=nb)
nb.note(run, "checked before the case")
print(nb.get(run)["question"], len(nb.runs()))
```

Leave out `notebook=` and the host's data folder is used (§6).

## 4. The command line, with `--json`

Every command prints a readable report. With `--json` it prints the same
result as JSON on stdout, and nothing else, so any language can run it and
parse the output.

| command | what |
|---|---|
| `athanor capabilities` | what the installed llama.cpp binding can do |
| `athanor inspect MODEL [--projector MMPROJ] [--no-llama] [--metadata] [--full-hash] [--strict]` | Inspect |
| `athanor tokenize MODEL… (--text T \| --file F) [--context N]` | the context ruler |
| `athanor splits MODEL… [--strings-file F]` | worst-split strings |
| `athanor compare A B [--rows N]` | Compare |
| `athanor overlap MODEL…` | vocabulary overlap matrix |
| `athanor template [SPEC] [--model M] [--vs SPEC2] [--conversation FILE] [--no-gen] [--list] [--strict]` | Template |
| `athanor roundtrip MODEL [--whole]` | does Athanor's writer reproduce this file? |
| `athanor notebook list\|show ID\|note ID TEXT\|where` | the record |
| `athanor version` | versions |

`athanor` is installed as a command, and `python -m athanor` works
everywhere. Every command also takes `--pretty` (indented JSON),
`--no-record` (don't write this run to the notebook) and `--data DIR`.

stdout and stderr are always UTF-8, even when Windows would default a pipe
to its ANSI code page. `tokenize` with neither `--text` nor `--file` reads
UTF-8 from stdin, and a byte-order mark is dropped there and in every file
Athanor reads. Text that starts with a dash goes as `--text="-…"`.

**Exit codes** are stable:

| code | meaning |
|---|---|
| 0 | ran; any findings are in the report |
| 1 | ran, `--strict` was given, and there were `problem` findings; or `roundtrip` found a file that does not round-trip |
| 2 | usage error |
| 3 | an input file is missing or is not a GGUF Athanor can read |
| 4 | llama.cpp (llama-cpp-python) is needed and not available (`tokenize`, `splits`). `inspect` and `template` still answer without it, with those checks marked `skipped` |
| 5 | anything else (the message says what; set `ATHANOR_DEBUG=1` for a traceback) |

From another language:

```csharp
// C#
var p = Process.Start(new ProcessStartInfo("python",
    $"-m athanor inspect \"{model}\" --json --no-record") { RedirectStandardOutput = true });
var report = JsonDocument.Parse(p.StandardOutput.ReadToEnd());
p.WaitForExit();   // p.ExitCode: 0 ran, 3 bad file, 4 no llama.cpp …
```

```javascript
// Node
const { execFileSync } = require("child_process");
const report = JSON.parse(execFileSync("python",
  ["-m", "athanor", "inspect", model, "--json", "--no-record"]).toString());
```

## 5. The file formats

What Athanor writes is specified, so another program can read it without
importing Athanor:

* [`formats/notebook.md`](formats/notebook.md): the notebook, JSON Lines,
  append-only.
* [`formats/results.md`](formats/results.md): the result of every command
  (what `--json` prints and what the notebook stores).
* Recordings (`.athrec-meta` / `.athrec-data`) arrive with the Waterfall in
  Phase 2, specified the same way.

## 6. Writing a host

A host is any object with five methods. Athanor calls them; it never
imports the host.

```python
import contextlib
from pathlib import Path
from athanor import api

class MyHost:
    def borrow_gpu(self, reason):          # held around anything that loads weights
        return contextlib.nullcontext()    # a real host: pause its own model, then restore it
    def vram_plan(self):                   # what you know about free VRAM, or None
        return None
    def model_dirs(self):                  # where your models live
        return [Path("models")]
    def data_dir(self):                    # where the notebook and results go
        return DATA_DIR / "myapp"
    def settings(self):                    # read-only settings Athanor may consult
        return {}

api.set_host(MyHost())
print(api.Notebook().path)                 # …/myapp/notebook/notebook.jsonl
api.set_host(api.NullHost())               # back to standalone
```

`set_host` checks the shape and names any missing method. `NullHost` is the
default. It uses `ATHANOR_DATA` (or the per-user data folder:
`%LOCALAPPDATA%\Athanor` on Windows, `~/.local/share/athanor` on Linux) and
`ATHANOR_MODELS` (folders, separated by `os.pathsep`).

In 0.1 nothing loads weights, so `borrow_gpu` is not yet called. It is in
the port now so hosts are written against its final shape. ATK's
`lab_host.py` will be the full-size example: its AI-queue ticket, unloading
its own model and restoring it afterwards.

## 7. Reading results: labels and findings

Every number says how it was made:

| label | meaning |
|---|---|
| `MEASURED` | observed by running something (llama.cpp tokenized it; a load was timed) |
| `DECLARED` | what the file says about itself, read from its header and not checked |
| `ESTIMATE` | arithmetic from other figures, not an observation |
| `EXPERIMENTAL` | research-grade: interesting, not evidence |

A labelled figure is `{"value": …, "label": …, "unit"?: …, "how"?: …}`.

A **finding** is one check's result:
`{"check", "status", "message", "label", "evidence"}`, where `status` is
`problem`, `warn`, `info`, `skipped` or `ok`. Reports list problems first.
`message` is a sentence meant for a person; `check` and `evidence` are for
programs.

## 8. Threads, versions and the promise

* A `Vocab` is one llama.cpp model handle. Use it from one thread at a time,
  and close it (or use `with`). Loads are serialised by a lock, because
  llama.cpp's log callback is process-wide.
* Every result records the versions that made it (`athanor`, the
  llama-cpp-python version, Python, the platform) when it goes into the
  notebook.
* **The promise.** Until 1.0, names in `athanor.api`, CLI options and result
  fields may change, and every change is listed in `CHANGELOG.md`. From 1.0,
  a name in `api.__all__`, a documented CLI option or exit code, or a field
  documented in `docs/formats/` is removed or changed only in a major
  version, after a minor version that warns. New fields may appear in any
  version, so ignore what you don't know.

## 9. When something is missing

* **`athanor capabilities`** is the first thing to run. It says which binding
  is installed, and for each feature whether it is available and, if not,
  why and what needs it.
* **LlamaUnavailable** (exit 4) means llama-cpp-python is not importable in
  this Python. The header-only parts still work: `inspect --no-llama`,
  `compare`, `overlap`, `template` without `--model`, and `roundtrip`.
* **VocabLoadError** means llama.cpp refused the file. The exception carries
  llama.cpp's own last lines, the only place the reason is written.
* **"decoding … rows needs gguf-py"**: `pip install gguf` for K-quant and
  i-quant lineage checks.
* **The process died.** Look in the log folder (§10). `crash-<date>.log`
  holds the stack of every thread at the moment of death. The last
  `breadcrumb` line in `athanor-<date>.jsonl` says what Athanor had just
  asked llama.cpp to do. A token id outside the vocabulary would kill the
  process inside llama.cpp, so `Vocab` checks every id first.

## 10. The log

Athanor keeps its own log, so a failure can be understood afterwards. It is
local only.

* `<data_dir>/logs/athanor-YYYY-MM-DD.jsonl` has one JSON object per line.
  It records every command-line run (arguments, versions, exit code,
  duration, and any exception with its traceback), and every vocabulary
  load with llama.cpp's complete load log at every level. It also records
  anything llama.cpp says inside a captured block, and a **breadcrumb**,
  flushed to disk, before each call into llama.cpp that could end the
  process.
* `<data_dir>/logs/crash-YYYY-MM-DD.log` is written by Python's
  `faulthandler` when something native kills the process. The command line
  turns this on. A host that wants it calls `api.log.enable_crash_log()`,
  unless it already uses faulthandler itself.

Turn the log off with `ATHANOR_LOG=0`, or from a host:

```python
from athanor import api

api.log.configure(directory=DATA_DIR / "athanor-logs")    # or enabled=False
api.log.event("host-note", what="started a batch", files=3)
print(api.log.read_events()[-1]["event"])
api.log.reset()                                             # back to the defaults
```

The format is [`formats/log.md`](formats/log.md). Athanor never writes to a
model file; it opens models read-only. Still, a folder of models kept just
for Athanor (point `ATHANOR_MODELS` at it) keeps experiments apart from the
copies your other tools use. Every result records the file it came from:
size, modification time, and a hash of its header.
