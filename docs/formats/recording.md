# Recordings — `.athrec-meta` + `.athrec-data` (+ `.athrec-tap`)

A Waterfall recording is what a model was choosing from at every token it
wrote. It is two files with the same stem, and a third when the Tap
recorded the model's insides too:

```text
20260928-221530-Magistral-Small-2509-Q4_K_M-3fa9c1.athrec-meta   JSON, UTF-8
20260928-221530-Magistral-Small-2509-Q4_K_M-3fa9c1.athrec-data   binary
20260928-221530-Magistral-Small-2509-Q4_K_M-3fa9c1.athrec-tap    binary, optional
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

## Reading

* Check `athrec:version`. A reader for version 1 refuses anything else.
* Check the data file's size: `n_steps × step_bytes`. Check
  `athrec:data_sha256` when you want to be sure. The same for the Tap file:
  `tap.n_steps × tap.record_bytes`, and `tap.sha256`.
* Unknown keys may appear in any version; ignore them.
