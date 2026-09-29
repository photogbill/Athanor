# Athanor in five minutes

**1. Install** (see [INTEGRATING.md §2](INTEGRATING.md#2-installing) about
llama.cpp):

```text
pip install git+https://github.com/photogbill/Athanor
python -m athanor capabilities
```

The second line says which llama.cpp you have and what it can do.

**2. Look at a model.** No weights are loaded and no GPU is used:

```text
python -m athanor inspect D:\Models\my-model.Q4_K_M.gguf
```

You get the anatomy (every figure says whether the file DECLARED it or
Athanor MEASURED it) and the health checks, problems first.

**3. See how much of your material fits:**

```text
python -m athanor tokenize modelA.gguf modelB.gguf --file my_documents\ --context 32768
```

**4. Check a chat template against a model**, including one the model did
not come with:

```text
python -m athanor template --list
python -m athanor template mistral-v7-tekken --model my-model.gguf
python -m athanor template gguf --vs chatml --model my-model.gguf
```

**5. Look back.** Every run is in the notebook:

```text
python -m athanor notebook list
python -m athanor notebook show <id>
python -m athanor notebook note <id> "looks fine for the Pashto set"
```

Add `--json` to any command for machine-readable output, and `--no-record`
to leave a run out of the notebook.
