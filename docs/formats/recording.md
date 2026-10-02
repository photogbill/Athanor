# Recordings — `.athrec-meta` + `.athrec-data` (+ `.athrec-tap`, + `.athrec-lens`)

A Waterfall recording is what a model was choosing from at every token it
wrote. It is two files with the same stem, a third when the Tap recorded
the model's insides too, and two more when the lens has been run on it:

```text
20260928-221530-Magistral-Small-2509-Q4_K_M-3fa9c1.athrec-meta        JSON, UTF-8
20260928-221530-Magistral-Small-2509-Q4_K_M-3fa9c1.athrec-data        binary
20260928-221530-Magistral-Small-2509-Q4_K_M-3fa9c1.athrec-tap         binary, optional
20260928-221530-Magistral-Small-2509-Q4_K_M-3fa9c1.athrec-lens-meta   JSON, derived
20260928-221530-Magistral-Small-2509-Q4_K_M-3fa9c1.athrec-lens        binary, derived
```

The layout follows SigMF's (a JSON description beside a binary file, with
`global`, `captures` and `annotations`) so that anyone who reads RF captures
will recognise it. It is **not** a SigMF recording: the binary file holds one
fixed-size record per token, not samples.

Format version **1**. Needs nothing but a JSON parser and the ability to
read little-endian integers and floats. Athanor's reader,
`athanor/waterfall/fileformat.py`, is the reference implementation and uses
only numpy.

## What is recorded, and what is not

* The model's **raw** distribution: the logits llama.cpp computed for that
  position, read before any sampler touched them. Repetition penalties,
  grammars, logit biases, top-k/top-p/min-p and temperature are the
  sampler's; they are recorded as settings, never folded into the numbers.
  So a token forced by a logit bias shows a low rank: that is the truth
  about the model.
* The **k** most probable tokens (256 unless the recorder was told
  otherwise), best first, and the probability of everything else as one
  number, `tail`.
* The token the sampler **chose**, with its exact rank and log-probability
  in the full distribution, even when it is not among the k stored.
* **When**: seconds since the recording began. The first step's time is how
  long the prompt took.

Numbers are MEASURED: they are what llama.cpp computed during the
generation, not a reconstruction. (Re-scoring a finished reply in one batch
gives slightly different numbers, because llama.cpp's batched and
single-token kernels round differently. A recording is taken live for that
reason.)

## `.athrec-data`

`n_steps` records, back to back, each `32 + 8·k` bytes, little-endian, no
padding:

| offset | type | field | meaning |
|---|---|---|---|
| 0 | int32 | `chosen` | the token id the sampler picked |
| 4 | int32 | `rank` | its rank in the model's distribution: 0 is the model's favourite; −1 if unknown |
| 8 | float32 | `logprob` | its log-probability, in nats (NaN if unknown) |
| 12 | float32 | `entropy` | of the whole distribution, in nats |
| 16 | float32 | `tail` | probability outside the k stored candidates |
| 20 | uint32 | `flags` | 1: end of generation; 2: a control token; 4: the logits held NaN or ±inf |
| 24 | float64 | `t` | seconds since the recording began |
| 32 | int32[k] | `ids` | the k most probable tokens, best first; ties go to the lower id |
| 32+4k | float32[k] | `logprobs` | their log-probabilities, in nats |

Probability is `exp(logprob)`. For the Waterfall's colour scale in decibels,
`10·log10(p) = 10·logprob / ln 10`.

In numpy:

```text
dt = np.dtype([("chosen","<i4"),("rank","<i4"),("logprob","<f4"),("entropy","<f4"),
               ("tail","<f4"),("flags","<u4"),("t","<f8"),
               ("ids","<i4",(k,)),("logprobs","<f4",(k,))])
steps = np.fromfile("….athrec-data", dtype=dt)
```

## `.athrec-meta`

```text
{
  "global": {
    "athrec:version": 1,
    "athrec:k": 256,
    "athrec:step_bytes": 2080,              32 + 8·k
    "athrec:layout": "chosen:int32 …",      the table above, as one line
    "athrec:n_steps": 412,
    "athrec:n_vocab": 131072,
    "athrec:created": "2026-09-28T22:15:31Z",
    "athrec:started": "2026-09-28T22:15:30.412339Z",
    "athrec:recorder": "athanor 0.3.0",
    "athrec:versions": {…},                 athanor, llama-cpp-python, python, numpy, platform
    "athrec:label": "MEASURED",
    "athrec:how": "…",
    "athrec:model": {path, name, size, mtime, header_sha256},
    "athrec:settings": {…},                 the sampler's settings, as the host gave them
    "athrec:host": {"name": "ATK", …} | null,
    "athrec:error": null | "…",             why the recording stopped early, if it did
    "athrec:data_bytes": 856960,
    "athrec:data_sha256": "…"               of the .athrec-data file
  },
  "captures": [{"athrec:step_start": 0, "athrec:datetime": "…"}],
  "annotations": [],                        {athrec:step_start, athrec:label, …} — see below
  "prompt": {"n_tokens": 1843, "ids": [...] | null, "text": "…" | null, "note": null | "…"},
  "messages": [{role, content}, …] | null,  the conversation, if the host gave it
  "reply": {
    "text": "…",                            every chosen token's text, special tokens included
    "spans": [[start, end], …],             step i's characters in text, counted in Unicode
                                            code points (not bytes, not UTF-16 units — a
                                            JavaScript or C# reader converts); [s, s] for a
                                            token that ends inside a multi-byte character
    "finish_reason": "stop" | "length" | null,
    "n_steps": 412
  },
  "timing": {"prompt_seconds", "total_seconds", "tokens_per_second",
             "overhead_ms_per_step", "label": "MEASURED"},
  "pieces": {"<id>": "<text>", …},          the text of every token id that appears
  "tap": {…}                                only when the Tap was asked for — see below
}
```

* `pieces` holds the text of every id in the recording (candidates, chosen
  tokens and the prompt), so a recording can be read without the model.
  Bytes that are not valid UTF-8 on their own (half of a character) are
  written as `\xNN`.
* `prompt.text` is the prompt as the model saw it: the chat template's
  render, special tokens included. It is null, with `prompt.note` saying
  why, when the model reads images: llama-cpp-python does not keep token ids
  for the image positions, so they cannot be written honestly.
  `prompt.n_tokens` is still the prompt's length.
* **Annotations.** `thinking`: `athrec:step_start` and `athrec:step_count`,
  the stretch between the model's thinking markers (`athrec:markers`), and
  `athrec:unfinished: true` when the reply ended inside it. `segment`: where
  a later part of one reply began (a reply carried on after its length
  limit) — `athrec:step_start` and `athrec:prompt_tokens`, no step count.
* **Ranks and ties.** `rank` is the number of tokens strictly more probable
  than the one chosen — 0 means it was the favourite, or tied for it. `ids`
  orders tokens by probability and breaks exact ties by the lower id. So
  under an exact tie the chosen token can sit later in `ids` than its rank:
  find it in `ids` by its id, not by its rank.
* **Edge values.** A logit llama.cpp masked is −∞, so its `logprobs` entry
  is −∞ and its probability 0. When the logits held NaN or ±∞ the step has
  flag 4; `entropy` and `logprob` can then be NaN. When no logit at all was
  finite, `ids` is a placeholder (0 … k−1), every `logprobs` entry is −∞,
  `tail` is 1.0 and `rank` is −1.
* `reply.text` is built from the chosen tokens, so it includes the end
  token and anything a host's stop strings would have hidden. It is what the
  model produced, not what the host displayed.
* A meta file is only ever written after its data file is complete. Files
  are created exclusively: an existing recording is never overwritten.
* **Privacy.** A recording holds the whole conversation. It is written where
  the host says (ATK: inside its own folder) and nowhere else, and it is
  never sent anywhere.

## `.athrec-tap` — the Tap (optional)

When the recording was made with the Tap (`attach(llm, tap=…)`), the model's
own tensors were copied as llama.cpp computed them: for every step, the row
of each tensor that belongs to the token whose logits that step was
sampled from. They are in `.athrec-tap`, one fixed-size record per step —
exactly `reply.n_steps` records, aligned with `.athrec-data` — described by
the meta file's `tap` block:

```text
"tap": {
  "streams": ["ffn_moe_topk-*", …],         the patterns asked for
  "layout": [                               one entry per stream, in record order
    {"name": "ffn_moe_probs-0", "kind": "ffn_moe_probs", "layer": 0,
     "dtype": "<f4", "count": 8, …}, …],
  "record_bytes": 4 + Σ 4·count,
  "n_steps": 412,
  "flags": {"1": "…", "2": "…", "4": "…"},  what each flag bit means
  "rows": "outputs",
  "how": "…", "label": "MEASURED",
  "model": {"arch", "n_layer", "n_embd", "n_expert", "n_expert_used", "n_vocab"},
                                            from the GGUF's own metadata (DECLARED)
  "unavailable": null | "…",                why nothing was tapped, if nothing was
  "error": null | "…",                      why the Tap stopped, if it failed
  "stopped_at": null | 211,                 the first step it did not record
  "notes": ["…"],
  "overhead_ms_per_step": 0.4,
  "forward_passes": 413,
  "file": "<stem>.athrec-tap",
  "bytes": 1234567, "sha256": "…"           of the .athrec-tap file
}
```

Each record, little-endian, no padding: `flags` (uint32 — 1: a stream was
missing at this step; 2: nothing was captured at this step; 4: the Tap had
stopped), then one field per `layout` entry, in order: `count` values of
`dtype` (`<i4` or `<f4`, nothing else). A missing value is −1 (int32) or
NaN (float32).

```text
fields = [("flags", "<u4")] + [(s["name"], s["dtype"], (s["count"],)) for s in meta["tap"]["layout"]]
tap = np.fromfile("….athrec-tap", dtype=np.dtype(fields))
tap["ffn_moe_topk-3"]          # [steps, experts used]: the experts layer 3 chose
```

* **Names** are llama.cpp's (b11093): `l_out-N` a layer's output (the
  residual stream), `result_norm` the final norm's output, `result_output`
  the logits, `ffn_moe_topk-N` the experts layer N's router chose (int32),
  `ffn_moe_weights-N` the router's value for each of them before
  normalising, `ffn_moe_probs-N` the router's value for every expert (a
  softmax for Mixtral and Qwen, a sigmoid for DeepSeek V3, logits for
  gpt-oss). `kind` and `layer` are the name split at its last `-N`; `layer`
  is null for tensors outside the layers.
* **Rows.** A tensor's row for a step is the one of the forward pass whose
  logits that step was sampled from — the token it reads, not the token it
  chose. The first step's rows come from the prompt's last token.
* **When there is no `.athrec-tap`**: the `tap` block may still be there,
  with an empty `layout` and `unavailable` or `error` saying why (a model
  that was not made tappable, a binding without the callback).
* A reader that does not know the Tap ignores the `tap` key and the file,
  as with any unknown key. The meta file is written after the Tap file.

## `.athrec-lens` + `.athrec-lens-meta` — the lens (derived)

When a recording carries the residual stream (`--tap residual`: `l_out-N`
for every layer), `athanor lens` reads each layer's output through the
model's own final norm and output matrix and writes what each layer "would
say" at every step — the logit lens (M1), and from it each token's
**decision depth** (M30). These two files are **derived**: computed after
the fact from the Tap's rows and the model file, recomputable with another
`k` or a calibrated lens (`--replace`), and the recording's own files are
never changed by them. They have their own meta file so the recording's
stays exactly as it was written.

`.athrec-lens-meta`:

```text
{
  "format": "athlens", "version": 1,
  "kind": "logit",                          "tuned" once the calibrated lens exists
  "recording": "<stem>.athrec-meta",
  "k": 8,                                   candidates kept per layer
  "layers": [0, 1, …, 47],                  the l_out layers read, in record order
  "n_layer": 48,                            the model's layer count (from the Tap's facts)
  "n_steps": 291, "n_vocab": 151936,
  "record_bytes": 4052,                     20 + L·(8·k + 20)
  "layout": "flags:uint32 depth:int32 …",   the table below, as one line
  "flags": {"1": "…", "2": "…"},
  "depth_rule": "…",                        how depth, first_seen and chosen_depth are defined
  "how": "…",
  "label": "MEASURED" | "EXPERIMENTAL",     MEASURED only when the check below passes
  "check": {                                the last layer's lens against the recording's logits
    "made": true, "steps_compared": 291,
    "favourite_agrees": 291, "favourite_agreement": 1.0,
    "candidates_compared": 4656,
    "max_abs_logprob_error": 0.012, "mean_abs_logprob_error": 0.003,
    "agreement_floor": 0.98, "reproduces": true, "how": "…"
  },
  "by_layer": {"agreement": [L], "fav_logprob_mean": [L], "entropy_mean": [L]},
  "unembedding": {fingerprint, n_vocab, n_embd, output: {tensor, type, tied, softcap, bias},
                  norm: {kind, eps, tensor, bias}, cache, decode_seconds},
  "model": {…},                             the recording's model block
  "pieces": {"<id>": "<text>", …},         the text of every token the layers name that the
                                            recording's own pieces do not hold
  "pieces_from": "llama.cpp" | "the file's token list",
  "seconds": 14.2, "ms_per_step": 49,
  "created": "…", "athanor": "0.4.0", "versions": {…},
  "notes": ["…"],
  "file": "<stem>.athrec-lens", "bytes": …, "sha256": "…"
}
```

`.athrec-lens`: `n_steps` records, aligned with `.athrec-data`, each
little-endian, no padding, with `L = len(layers)`:

| offset | type | field | meaning |
|---|---|---|---|
| 0 | uint32 | `flags` | 1: no residual at this step (the Tap had nothing); 2: the last layer's lens does not name the recording's favourite here |
| 4 | int32 | `depth` | decision depth: the first layer (an INDEX into `layers`) from which the recording's favourite is the lens's top token at every layer up to the last; −1 when the last layer itself disagrees |
| 8 | int32 | `first_seen` | the first layer (index) at which the favourite is within the lens's k; −1 if never |
| 12 | int32 | `fav` | the recording's favourite at this step (its `ids[0]`) |
| 16 | int32 | `chosen_depth` | as `depth`, for the token the sampler took |
| 20 | int32[L][k] | `ids` | each layer's k likeliest tokens, best first (ties by the lower id) |
| 20+4Lk | float32[L][k] | `logprobs` | their log-probabilities at that layer, nats (a full softmax over the vocabulary at that layer) |
| 20+8Lk | int32[L] | `fav_rank` | the favourite's rank at each layer (0 = that layer's top token) |
| +4L | float32[L] | `fav_logprob` | its log-probability at each layer |
| +4L | int32[L] | `chosen_rank` | the chosen token's rank at each layer |
| +4L | float32[L] | `chosen_logprob` | |
| +4L | float32[L] | `entropy` | of the layer's whole distribution, nats |

A step with flag 1 holds −1 in every integer field and NaN in every float.

```text
L, k = len(meta["layers"]), meta["k"]
dt = np.dtype([("flags","<u4"),("depth","<i4"),("first_seen","<i4"),("fav","<i4"),
               ("chosen_depth","<i4"),("ids","<i4",(L,k)),("logprobs","<f4",(L,k)),
               ("fav_rank","<i4",(L,)),("fav_logprob","<f4",(L,)),("chosen_rank","<i4",(L,)),
               ("chosen_logprob","<f4",(L,)),("entropy","<f4",(L,))])
lens = np.fromfile("….athrec-lens", dtype=dt)
lens["fav_logprob"]            # [steps, layers]: the picture — time down, depth across
np.array(meta["layers"])[lens["depth"]]   # decision depth as layer numbers (where depth >= 0)
```

* **What the lens is.** Each `l_out-N` row (float32, as the Tap copied it)
  through the final norm — RMS for Llama-style models, LayerNorm with its
  bias for GPT-2-style ones — and the output matrix (`output.weight`, or
  `token_embd.weight` when tied) dequantized to float32 by Athanor, plus
  `output.bias` and Gemma's logit soft-cap when the file has them. The
  decoded matrix is kept under `<data>/lenses/<header-sha256-prefix>-<size>/`
  and reused.
* **The check.** At every step the last recorded layer's reading is compared
  with the recording's own logits: its top token must be the recording's
  favourite, and its log-probabilities over the recording's top candidates
  are compared. `label` is MEASURED when the favourite agrees at 98 % of
  steps or more, EXPERIMENTAL otherwise (a norm or architecture this lens
  does not know, or a recording without the last layer) — with the numbers
  either way. On a quantized model run on a GPU the log-probabilities differ
  a little from llama.cpp's (its kernels quantize the activations for the
  output matmul; Athanor's numpy does not): a small `max_abs_logprob_error`
  with full agreement is the expected picture.
* **Its limit.** A lens shows what can be read out of a layer in the
  output's own terms, not that the model "thinks in" those words there; the
  plain lens reads early layers poorly, which `by_layer.agreement` shows
  per layer.
* `depth`, `first_seen` and `chosen_depth` are indices into `layers`, not
  layer numbers, so a lens over a subset of layers (`--layers`) reads the
  same way. `athanor.lens.LensData` returns them as layer numbers.

## Reading

* Check `athrec:version`. A reader for version 1 refuses anything else.
* Check the data file's size: `n_steps × step_bytes`. Check
  `athrec:data_sha256` when you want to be sure. The same for the Tap file:
  `tap.n_steps × tap.record_bytes`, and `tap.sha256`; and for the lens file,
  `n_steps × record_bytes` and `sha256` from its own meta file.
* Unknown keys may appear in any version; ignore them.
