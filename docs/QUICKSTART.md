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

**5. Watch a model choose its words.** This one loads the weights (onto the
GPU when llama-cpp-python was built for one), writes a reply, and records
every token with everything it was choosing between:

```text
python -m athanor record my-model.gguf --prompt "Describe the relay plan." --max-tokens 200
python -m athanor recording <the path it printed>
python -m athanor.gui
```

The last line opens the Waterfall (`pip install athanor[gui]` for PySide6):
the reply along the top, one row per token, the candidates across, the
token taken highlighted on every row. **Aa read** writes each candidate in
its cell; ◆ jumps to the next moment the model hesitated.

**6. Look inside.** For a mixture-of-experts model, record which experts
every layer used for every token, and see it beside the Waterfall:

```text
python -m athanor tap probe my-moe-model.gguf
python -m athanor record my-moe-model.gguf --prompt "Describe the relay plan." --tap experts
```

The probe comes first: it checks, on your card, that the Tap changes
nothing and reads llama.cpp's own numbers, and says what it costs. In the
player the **expert map** sits under the candidates — the router's score
for every expert at the cursor, the experts used framed; "one layer, over
time" turns it into a second waterfall.

**7. Look back.** Every run is in the notebook:

```text
python -m athanor notebook list
python -m athanor notebook show <id>
python -m athanor notebook note <id> "looks fine for the Pashto set"
```

Add `--json` to any command for machine-readable output, and `--no-record`
to leave a run out of the notebook.
