# The log — format version 1

Athanor's own record of what happened, kept so that a failure can be
understood afterwards. It is local only and never sent anywhere. Turn it off
with `ATHANOR_LOG=0`.

## Files

All in `<data_dir>/logs/`:

* `athanor-YYYY-MM-DD.jsonl`: one JSON object per line, UTF-8, `\n` line
  endings, append-only, one file per UTC day.
* `crash-YYYY-MM-DD.log`: plain text, written by Python's `faulthandler`
  when something native (llama.cpp, a driver) kills the process. Each
  process that enables it first writes a header line
  (`=== <time> pid <pid> <argv>`), followed only by crash stacks, if any.

## Every line

| field | meaning |
|---|---|
| `time` | ISO 8601 UTC with milliseconds, e.g. `2026-09-28T21:30:05.123Z` |
| `format` | `1` |
| `pid` | process id |
| `event` | what happened (below) |

## Events

| event | fields |
|---|---|
| `cli-start` | `argv`, `cwd`, `versions` |
| `cli-end` | `exit_code`, `seconds` |
| `cli-input-error`, `cli-no-llama`, `cli-vocab-load-error`, `cli-usage-error`, `cli-error` | `error` (exception type), `message`, `traceback`, `exit_code` |
| `breadcrumb` | `what` (`vocab-load`, `tokenize`, `worst-splits`, `template-tokenize`, …) plus what it was about to do it to. Written and flushed to disk **before** the native call |
| `vocab-load` | `path`, `size`, `mtime`, `seconds`, `n_tokens`, `vocab_type`, `add_bos`, `add_eos`, `llama_cpp_log` (every line llama.cpp logged during the load, as `[level, text]`, debug included) |
| `vocab-load-failed` | `path`, `size`, `mtime`, `seconds`, `llama_cpp_log` |
| `llama-log` | `path`, `lines`: what llama.cpp said inside a captured block |
| `recorded` | `run`, `kind`, `notebook`: a result went into the notebook |
| `record-failed` | an exception while recording (the result was still printed) |

Hosts may add their own events with `athanor.log.event(name, **fields)`.

## Reading it

Read line by line, skip lines that do not parse (a line torn by a crash),
and ignore events and fields you do not know.

**After a crash**, the last `breadcrumb` for that `pid` is what Athanor had
just handed to llama.cpp. `crash-*.log` has the stack.
