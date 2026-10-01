# Using Athanor in your own program

Athanor is built to be put inside other software. Nothing in it needs the
Analyst Toolkit (ATK) or any other host: ATK is simply its first user, and
uses only what is described here.

This guide covers Athanor **0.3**: the file tabs (Inspect, Tokenize,
Compare, Template), the Waterfall, which records a model's choices token
by token and plays them back, and the Tap, which copies a model's own
tensors out as it computes — which experts a mixture-of-experts model used,
each layer's output — beside every token. Every Python example in it runs in the test
suite (`tests/test_docs.py`), so the examples cannot drift from the code. In
them, `MODEL` and `MODEL_B` are paths to GGUF files, `RUNNABLE_MODEL` is a
GGUF with weights (a tiny one, in the tests), and `DATA_DIR` is a scratch
folder.

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
11. [The Waterfall player (Qt)](#11-the-waterfall-player-qt)

---

## 1. What you get

There are five ways in. Use as many or as few as you need:

| way in | for | status |
|---|---|---|
| the Python API, `athanor.api` | Python programs | **0.1** |
| the command line, `python -m athanor … --json` | programs in any language | **0.1** |
| open file formats (`docs/formats/`) | reading Athanor's output with no Athanor | **0.1** (notebook, results); **0.2** recordings |
| the host port, `athanor.host` | an application lending its GPU, folders and data folder | **0.1** |
| Qt widgets, `athanor.gui` | PySide6 applications | **0.2**: the Waterfall player; **0.3**: the expert map |

What Athanor does without loading a model's weights:

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

And with a model loaded:

* **The Waterfall**: every token a model writes, recorded with the
  distribution it was chosen from — the 256 most probable tokens, the
  entropy, where the chosen one ranked, and when. Recorded live from a
  `llama_cpp.Llama` your program already has, without changing a single
  token it generates, and played back in a Qt widget: the reply along the
  top, time down the screen, candidates across.
* **The Tap**: the model's own tensors, copied out while llama.cpp computes
  them — for a mixture-of-experts model, which experts every layer used for
  every token; for any model, each layer's output (what the logit lens
  reads). Recorded beside the Waterfall, bit-for-bit without changing the
  reply on the machines tested so far, and `athanor tap probe` checks yours.

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

### Record a generation: the Waterfall

If your program already has a `llama_cpp.Llama`, attach the recorder to it
around the call that generates. Nothing about the generation changes: the
recorder reads the logits llama.cpp sampled from, after the sampler has
picked, and the same seed gives the same reply, recorded or not.

```python
# needs: llama
import llama_cpp
from athanor import api

llm = llama_cpp.Llama(RUNNABLE_MODEL, n_ctx=512, verbose=False)
messages = [{"role": "user", "content": "Name a colour."}]
with api.attach_recorder(llm, meta={"settings": {"temperature": 0.7}}) as rec:
    out = llm.create_chat_completion(messages, max_tokens=16, temperature=0.7, seed=1)
reply = out["choices"][0]
path = rec.save(DATA_DIR / "recordings", finish_reason=reply["finish_reason"],
                host_reply=reply["message"]["content"])

r = api.read_recording(path)
print(r.n_steps, r.text)
first = r.chosen(0)                        # the token, its rank, p, entropy, time
for c in r.candidates(0, 5):               # what it was chosen from
    print(("▶" if c["chosen"] else " "), f"{c['p']:.1%}", repr(c["piece"]))
```

* **Streaming.** With `stream=True`, consume the stream *inside* the
  `with` block: the tokens are sampled as you iterate, and the recorder is
  only attached while the block is open.
* **Keep the model loaded until the `with` block ends.** On leaving, the
  recorder looks up the text of every token it saw, using the model's
  vocabulary. `save()` needs nothing from the model and can run anywhere.
* **Your own lock.** Attach inside whatever lock your program already holds
  around generation. The recorder wraps that one instance's `sample` for the
  length of the block and puts it back afterwards, even on an exception.
* **Cost.** Well under a millisecond per token on a 32k vocabulary, around
  0.7 ms on a 131k one, measured and written into each recording
  (`timing.overhead_ms_per_step`).
* **If the recorder fails**, the recording stops and says why (`error`);
  the generation carries on. `api.recorder_check(llm)` says in advance
  whether a binding and model can be recorded (`None` means yes).
* **Without a host model**, `api.record_once(path, messages=…)` loads the
  file (holding `borrow_gpu`), generates once, saves, and returns the
  summary. That is what `athanor record` runs.
* **One reply in several parts.** A reply your program carries on after it
  hit its length limit is still one reply: pass the first recorder back in,
  `api.attach_recorder(llm, recorder=rec)`, around each later part. The
  recording marks where each part began (a `segment` annotation).
* **Moments of doubt.** `r.doubts()` lists the steps where the sampler took
  something other than the model's favourite, or where even the token taken
  had less than half the probability — where to look first. The player
  jumps between them, and underlines them in the reply.

A host does not have to keep a recording's pieces together itself. ATK's
Chat, for example, makes one small session object per recorded reply, hands
the engine a callable that enters `attach_recorder` inside the engine's own
lock around every part, and saves on a worker thread when the reply ends —
one small class, `ReplyRecording` in ATK's `atk/core/lab_host.py`.

### Play a recording: the widgets

`athanor.gui` (PySide6; `pip install athanor[gui]`) is the player a person
looks at. `WaterfallPanel` is a folder's recordings beside the player — the
whole Waterfall tab, as one widget; `WaterfallPlayer` is the player alone.

```python
# needs: qt
from PySide6.QtWidgets import QApplication
from athanor.gui import WaterfallPanel

app = QApplication.instance() or QApplication([])
panel = WaterfallPanel(DATA_DIR / "recordings")    # the list beside the player
# your_layout.addWidget(panel), and when your program has made a recording:
#     panel.open(path)
```

What it shows, so your users can be told:

* **the reply along the top**, the token at the cursor lit, what is still to
  come dimmed (or hidden, to watch it written), the words the model was
  unsure of underlined; hover a word for how sure it was;
* **the waterfall**: time runs down, one row per token; the candidates run
  across, the model's favourite first, brightness their probability in dB
  (0 dB is certain, −10 dB is 10 %); the entropy of each step at the right;
* **"taken"**, the column at the left of every row: the token the model
  actually wrote, whatever its rank — framed white when it was the model's
  favourite, amber with its rank when the sampler took another;
* **Aa read** zooms in until each cell shows its candidate's text, so a row
  reads as the words the model weighed; press again for the heat-map
  overview;
* **the transport**: play at the recorded speed or faster, step, scrub, and
  ◆ to jump between moments of doubt (`[` and `]` from the keyboard).

`python -m athanor.gui [recording or folder]` opens the same thing in a
window of its own.

When a recording carries the Tap's `experts`, the player shows the
**expert map** beside the candidates: layers down, experts across,
brightness the router's score for each expert at the token under the
cursor (in dB, like the waterfall), the experts used framed in amber with
how much each counted. It can also show how often each expert was used over
the whole reply (or its thinking, or its answer), or one layer **over
time** — a second waterfall, tokens down, that layer's experts across,
scrolling with the cursor; click a row to move it. `ExpertPanel` is the
same thing on its own.

`api.list_recordings(folder)` lists a folder, newest first;
`api.recordings_folder()` is `<data_dir>/recordings`. The format is in
[`formats/recording.md`](formats/recording.md).

### Look inside: the Tap

llama.cpp computes a graph of named tensors for every token — `l_out-12`
is layer 12's output, `ffn_moe_topk-12` the experts layer 12's router
chose, `result_output` the logits. The Tap copies the ones you name while
llama.cpp computes them, from whichever device holds them. It never writes
into the model.

The callback it uses is fixed when a context is made, so a model is made
**tappable** once: `make_tappable` rebuilds the `Llama`'s context with the
Tap's dispatcher in it — same parameters, weights shared and untouched, an
empty cache (the next generation reads its prompt again). Do it right after
loading, while nothing is generating.

```python
# needs: llama
import llama_cpp
from athanor import tap

llm = llama_cpp.Llama(RUNNABLE_MODEL, n_ctx=512, verbose=False)
tap.make_tappable(llm)                     # once; tap.is_tappable(llm) is now True
with tap.capture(llm, ("l_out-*", "result_output")) as t:
    llm.create_completion("The river", max_tokens=4, temperature=0)
first = t.decodes[0]                       # one entry per forward pass
print(first["n_tokens"], sorted(first["tensors"])[:3])
logits = first["tensors"]["result_output"][0]   # [rows, values]: the prompt's last token
print(logits.shape)
```

* **What to name.** Glob patterns over llama.cpp's tensor names, or a
  preset: `"experts"` (`ffn_moe_topk-*`, `ffn_moe_weights-*`,
  `ffn_moe_probs-*`), `"residual"` (`l_out-*`), `"logits"`
  (`result_output`). `*` never crosses a space, so `l_out-*` does not take
  llama.cpp's derived `l_out-3 (view)`. `tap.tensor_names(llm)` (or
  `athanor tap names MODEL`) lists every tensor one forward pass computes.
* **Rows.** By default (`rows="outputs"`) each forward pass keeps the rows
  that produce logits — one per generated token. `rows="all"` keeps every
  token of the prompt too (large: a prompt's worth of each tensor). The
  last layer only ever has output rows: llama.cpp drops the rest there.
* **Memory.** `capture` keeps every forward pass, up to `limit_bytes`
  (1 GiB by default); past it the Tap stops and says so (`t.error`).
  `tap.watching(llm, tap.Tap(...))` keeps only the latest pass, which is
  what a recording needs.
* **Cost.** While a model is tappable, every forward pass makes one Python
  call per graph node (about 30 per layer) even when nothing is being
  copied; while copying, llama.cpp computes the graph in pieces so each
  wanted tensor can be read. `tap.make_plain(llm)` takes the Tap out again.
* **Failures** inside the Tap are kept (`t.error`) and never reach
  llama.cpp; the generation carries on.

With the Waterfall, name the Tap's streams when attaching, and every
recorded token carries the rows of the forward pass that produced it — for
a mixture-of-experts model, the **expert map**:

```python
# needs: llama
import llama_cpp
from athanor import api, tap

moe = api.tiny_model(DATA_DIR / "moe.gguf", n_experts=8, n_experts_used=2)
llm = llama_cpp.Llama(str(moe), n_ctx=512, verbose=False)
tap.make_tappable(llm)
with api.attach_recorder(llm, tap="experts") as rec:
    llm.create_completion("the radio signal", max_tokens=8, temperature=0)
r = api.read_recording(rec.save(DATA_DIR / "recordings"))
ex = r.tap.experts()                       # None for a model without experts
print(ex.ids.shape)                        # [tokens, layers, experts used per token]
print(ex.share()[0, 0])                    # how much each counted, first token, layer 0
print(ex.usage().shape)                    # [layers, experts]: how often each was used
```

* A model that is not tappable is recorded without the Tap, and the
  recording says why (`r.tap_info["unavailable"]`); so does a Tap that
  failed or stopped at its limit (`error`, `stopped_at`). The Tap's
  failures never cost the recording.
* `r.tap.stream("l_out-12")` is one stream, `[tokens, values]`;
  `r.tap.stacked("l_out")` stacks every layer's, `[tokens, layers, values]`.
* `ExpertRouting` (`r.tap.experts()`): `ids` (the experts used), `weights`
  (the router's value for each), `probs` (its value for every expert),
  `share()` (how much each used expert counted, summing to 1), `usage()`
  and `grid(step)`.
* Experts are not personas. The map shows which parts of the network the
  router used; what they are for is what it lets you ask.

**Does it work here, and what does it cost?** That depends on the build and
the card, so Athanor measures it rather than promising:

```python
# needs: llama
import llama_cpp
from athanor import api

llm = llama_cpp.Llama(RUNNABLE_MODEL, n_ctx=512, verbose=False)
report = api.tap_probe(llm, tokens=6)      # leaves llm's own context alone
print(report["verdict"])
print(report["exactness"]["idle_vs_plain_max_abs"], report["ms_per_token"])
```

The probe makes a plain context and a tapped one on the same loaded
weights, generates the same greedy continuation in each, and reports:
whether tapping changed any logit (0 means bit-identical), whether the
Tap's copy of `result_output` is exactly llama.cpp's logits, whether the
logit lens (the last layer's output through the final norm and the output
matrix, done by Athanor) reproduces them, milliseconds per token plain,
tapped-idle and copying each preset, and, for a mixture-of-experts model,
whether the experts used were the router's top-scoring ones. On a GPU,
computing the graph in pieces can stop the backend fusing some kernels,
which may change the last bits; the probe is how to find out.
`athanor tap probe MODEL` runs it from the command line.

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
| `athanor record MODEL (--prompt T [--system S] \| --messages FILE \| --raw --prompt T) [--max-tokens N] [--temperature T] [--top-k/--top-p/--min-p/--repeat-penalty] [--seed N] [--n-ctx N] [--gpu-layers N] [--k N] [--out DIR] [--tap experts\|residual\|logits\|PATTERN]…` | the Waterfall: generate once, recorded (with the Tap, if asked) |
| `athanor tap probe MODEL [--tokens N] [--n-ctx N] [--gpu-layers N]` | the Tap on this machine: does it change anything, does it read the truth, what does it cost |
| `athanor tap names MODEL` | every tensor one forward pass computes — what the Tap can read |
| `athanor recording PATH [--step N] [--top N] [--forks N] [--verify]` | read a recording: the text, where the favourite was not taken, one step's candidates |
| `athanor recordings [FOLDER]` | list recordings |
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
| 3 | an input file is missing, is not a GGUF Athanor can read, or llama.cpp refused to load it; or a recording is missing or damaged |
| 4 | llama.cpp (llama-cpp-python) is needed and not available (`tokenize`, `splits`, `record`, `tap`), or the Tap cannot run here. `inspect` and `template` still answer without it, with those checks marked `skipped` |
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
* [`formats/recording.md`](formats/recording.md): Waterfall recordings,
  `.athrec-meta` (JSON) beside `.athrec-data` (fixed-size binary records),
  and `.athrec-tap` when the Tap recorded too. Laid out like SigMF,
  readable with numpy alone.

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

`borrow_gpu` is held around anything that loads weights: in 0.2, only
`record_once` (and so `athanor record`). A host that records its own
already-loaded model with `attach_recorder` loads nothing, so nothing is
borrowed. ATK is the full-size example: it records the model its chat is
already running.

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
* **VocabLoadError** (and, when loading weights, **ModelLoadError**) means
  llama.cpp refused the file. The exception carries llama.cpp's own last
  lines, the only place the reason is written.
* **RecorderUnavailable** means this binding, or this object, cannot be
  recorded; the message says which part is missing. `athanor capabilities`
  lists it under `recording`.
* **TapUnavailable** (exit 4) means the Tap cannot run: the binding lacks
  the evaluation callback, ggml's library is not beside it, or ggml's
  tensor layout is not the one Athanor checks for (a newer ggml that moved
  a field turns the Tap off rather than reading the wrong bytes).
  `athanor capabilities` lists it under `eval_callback`.
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
