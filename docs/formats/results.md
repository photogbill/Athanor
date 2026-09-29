# Results — what every command returns

`--json` prints one result, and the notebook stores the same object. This
document lists the fields programs may rely on. Results can carry more; ignore
what you do not know. Until 1.0 these can change, with a line in
`CHANGELOG.md` for every change.

## Shared shapes

**Labelled figure**: `{"value": …, "label": "MEASURED"|"DECLARED"|"ESTIMATE"|"EXPERIMENTAL", "unit"?: string, "how"?: string}`

**Finding**: `{"check": string, "status": "problem"|"warn"|"info"|"skipped"|"ok", "message": string, "label": string, "evidence": object}`.
`check` ids are stable; `message` is for people.

**File identity**: see [notebook.md](notebook.md#file-identity).

## `inspect`

| field | meaning |
|---|---|
| `kind` | `"inspect"` |
| `file` | file identity |
| `gguf` | `version`, `endian`, `alignment`, `n_kv`, `n_tensors`, `header_bytes`, `data_offset`, `file_size` |
| `anatomy` | labelled figures: `architecture`, `name`, `basename`, `finetune`, `size_label`, `license`, `file_type`, `context_length`, `embedding_length`, `block_count`, `head_count`, `head_count_kv`, `expert_count`, `rope_freq_base`, `parameters`, `tensor_bytes`, `bits_per_weight`, `kv_cache_per_token`; plus `tensor_types` (list of `{type, tensors, bytes}`), `layers_with_tensors`, `tensors_per_layer`, `non_layer_tensors` |
| `tokenizer.declared` | from the header: `model`, `pre`, `n_tokens`, `n_merges` (figures), `has_scores`, `token_types` (counts by type), `special_ids` (`{role: {id, text}}`), `add_bos`, `add_eos`, `chat_templates` (`{name: {chars, detected_format}}`) |
| `tokenizer.measured` | from llama.cpp, or null when it was not run: `type`, `n_tokens`, `special_ids`, `end_of_generation` (first 64 `{id, text}`), `n_end_of_generation`, `add_bos`, `add_eos`, `load_seconds`, `llama_cpp_warnings` |
| `findings` | findings, problems first |
| `summary` | counts by status |
| `projector` | file identity of `--projector`, when given |

Check ids: `structure`, `header-round-trip`, `tensor-types`, `pre-tokenizer`,
`vocab-rows` (one per `token_embd.weight` / `output.weight`), `name-vs-header`,
`chat-template`, `embedder`, `llama-cpp` (`skipped` when llama-cpp-python is
not available, `problem` when llama.cpp refused the file), `llama-cpp-log`,
`vocab-size`, `eos`, `own-template/marker-is-special`, `own-template/turn-end`,
`own-template/bos-once`, `own-template/renders`, `own-template/tokenizer`,
`projector-width`, `projector-name`.

## `tokenize`

`{"kind": "tokenize", "text_chars", "context", "models": [...]}`. Each model:
`model` (file identity), `vocab_type`, `n_vocab`, `n_tokens` (figure),
`words`, `chars`, `bytes`, `tokens_per_word`, `chars_per_token`,
`bytes_per_token`, `by_script` (`{SCRIPT: {words, tokens, tokens_per_word}}`),
`tokens_per_second`, `tokens` (first N `{id, piece}`), `tokens_shown`, and
`ruler` when a context was given: `{context, words (figure), chars (figure)}`.
From the command line with `--file`, the result also has `input`:
`{path, files, files_read, bytes_read, bytes_total, truncated}`.

Words are runs of letters, combining marks, digits and joiners, so a voweled
Arabic word or a Devanagari word with its matras counts once. CJK
ideographs count one each.

## `worst-splits`

`{"kind": "worst-splits", "label", "note", "strings": [...]}`, worst first.
Each string: `category`, `string`, `chars`, `tokens_per_char_worst`, and
`models` (a list of `{model, n_tokens, pieces}`).

## `fertility`

`{"kind": "fertility", "label", "scripts": [...], "rows": [...]}`. Each row:
`model`, `vocab_type`, `n_vocab`, `tokens_per_word`, then one field per
script.

## `compare`

| field | meaning |
|---|---|
| `a`, `b` | file identities |
| `same_header` | the two headers are byte-identical (the data may still differ) |
| `tokenizer` | `model`, `pre`, `n_tokens` (`{a, b}` each), `identical`, `same_token_list`, `ids_that_differ`, `first_differences`, `only_in_a` / `only_in_b` (`{count, sample}`), `overlap` (figure), `special_ids_differ`, `arrays` (digest per tokenizer array) |
| `metadata` | `only_in_a`, `only_in_b`, `changed` (list of `{key, a, b}`) |
| `tensors` | counts, `only_in_a/b`, `shape_differs`, `type_differs` (lists and counts), `same_architecture` |
| `lineage` | `tensor`, `rows_compared`, `matched_by` (`"token string"`: the same token's row in each file, so a vocabulary that grew is still compared), `vocabulary_rows` (`{a, b}`, when they differ), `rows_byte_identical`, `cosine` `{mean, min, max}`, `relative_difference`, `verdict`, `verdict_label`, `how`; or `verdict` null or `"not the same base"` with `why` |

## `overlap`

`{"kind": "overlap", "models": [names], "overlap": [[share]], "identical_vocabulary": [[bool]], "note"}`

## `template`

| field | meaning |
|---|---|
| `template` | `{name, source, label}`; `source` is `library`, `gguf`, `file` or `text` |
| `detected_format` | llama.cpp's name for the format, or null |
| `model` | file identity, or null |
| `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, `bos_token_passed` | what was rendered, and how. `bos_token_passed` is the model's BOS text when its tokenizer adds one, and `""` otherwise. `messages` has no system message if the template refused one |
| `render`, `render_chars` | the text, or null when it would not render |
| `turn_end` | what the template writes after a reply |
| `markers` | per marker: `marker`; with a model also `ids`, `n_tokens`, `special`, `attrs`, `eog`, `always_special_style` |
| `n_tokens`, `tokens`, `segments` | with a model: every token `{id, piece, kind}` and the render as runs; `kind` is `bos`, `eos`, `eog`, `control`, `user_defined`, `byte` or `text` |
| `bos` | with a model: `{id, tokenizer_adds, expected, llama_cpp_python, llama_server, naive_runner, llama_cpp_said}`, the BOS count each kind of runner would send |
| `findings` | check ids `detected-format`, `renders`, `marker-is-special`, `turn-end`, `bos-once`, `bos-naive`, `tokenizer` |

## `template-compare`

`{"kind": "template-compare", "a": <template>, "b": <template>, "identical", "differences": [{op, a, b, a_at, b_at}], "token_difference"}`

## `roundtrip`

`{"kind": "roundtrip", "file", "header_identical", "first_difference", "header_bytes", "file_identical"?, "sha256"?}`
