# The notebook — format version 1

Athanor records every run in an append-only file of **JSON Lines**: one JSON
object per line, UTF-8, `\n` line endings. Nothing is ever rewritten. A note
about a run is a later line that names the run.

Default place: `<data_dir>/notebook/notebook.jsonl`, where `<data_dir>` is
the host's (`ATHANOR_DATA`, or `%LOCALAPPDATA%\Athanor` on Windows, or
`~/.local/share/athanor` on Linux, when running standalone).

## Lines

Every line has `type`, `format` (this document's version, `1`), `id` and
`time`.

* `id` is unique and sorts by time: `YYYYMMDDTHHMMSSZ-xxxxxx` (UTC, then six
  hex digits).
* `time` is ISO 8601 UTC: `2026-09-28T16:19:00Z`.

### `"type": "run"`

| field | type | meaning |
|---|---|---|
| `kind` | string | what ran: `inspect`, `tokenize`, `worst-splits`, `fertility`, `compare`, `overlap`, `template`, `template-compare` |
| `question` | string | the question the run answers, in words |
| `inputs` | list of objects | the files, each as a *file identity* (below) |
| `settings` | object | the options the run was given |
| `versions` | object | `athanor`, `llama_cpp_tag_tested`, `llama_cpp_python` (or null), `python`, `numpy`, `platform` |
| `result` | object | the result, exactly as `--json` prints it — see [results.md](results.md) |

### `"type": "note"`

| field | type | meaning |
|---|---|---|
| `run` | string | the `id` of the run it is about |
| `text` | string | the note |

## File identity

How a result names a file:

| field | meaning |
|---|---|
| `path` | absolute path when recorded |
| `name` | file name |
| `size` | bytes |
| `mtime` | modification time, ISO 8601 UTC |
| `header_sha256` | SHA-256 of the header (metadata, tokenizer and tensor layout). Cheap, and identifies the file's content description |
| `sha256` | SHA-256 of the whole file. Present only when asked for (`--full-hash`); slow on a large model |

## Reading it safely

* Read line by line and skip lines you do not understand (a `type` you do not
  know, or a `format` above the one you support).
* A last line cut off by a power failure is not valid JSON. Treat it as
  damaged and go on. Athanor's own reader reports it as `{"type": "damaged"}`.
* Never write into the middle of the file. To correct a run, add a note.
