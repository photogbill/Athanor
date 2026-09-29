# Test vocabularies

Three **vocabulary-only** GGUFs from llama.cpp's own test set
(`models/ggml-vocab-*.gguf`, llama.cpp `b11093`), used so the tests that
need llama.cpp's tokenizer run anywhere without a model download. They hold
a tokenizer and its metadata — **no weights**; they are not models.

| file | tokenizer | why it is here |
|---|---|---|
| `ggml-vocab-llama-spm.gguf` | SPM, 32,000 tokens | a clean SentencePiece vocabulary |
| `ggml-vocab-phi-3.gguf` | SPM, 32,064 tokens | carries a chat template (own-template checks) |
| `ggml-vocab-gpt-neox.gguf` | BPE, 50,432 tokens | has no pre-tokenizer — llama.cpp's "generation quality will be degraded" case |

SHA-256 (as copied):

```
16c3724582d59aa8bf84711894e833f916ee46a31d80e21312759c48bf8d0e69  ggml-vocab-llama-spm.gguf
967d7190d11c4842eab697079d98d56c2116e10eb617be355a2733bfc132e326  ggml-vocab-phi-3.gguf
ae593a7f9b8bb174ed4f5019e41530463e4dac7aa06e42dee8aa650d2bdac53d  ggml-vocab-gpt-neox.gguf
```

Licence: MIT, © The ggml authors — see `LICENSE-llama.cpp` beside them.

More of llama.cpp's vocabularies (Llama 3, Qwen 2, Gemma 4, Command-R …) are
used when `ATHANOR_TEST_VOCAB_DIR` points at a llama.cpp `models/` folder,
and a real model of yours when `ATHANOR_TEST_MODEL` names one.
