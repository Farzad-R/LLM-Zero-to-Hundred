---
name: test-pipeline
description: Run the RAG-GPT test suite and smoke-test the live app end-to-end. Use /test-pipeline to run all tests, or /test-pipeline <name|path> to run a single test.
---

# /test-pipeline

Full verification of the RAG-GPT pipeline: unit tests → live app smoke test → sources check.

`$ARGUMENTS` — optional. A pytest node ID (`tests/test_chatbot.py::test_name`), a file (`tests/test_chatbot.py`), or a keyword (`-k test_name`). If empty, the full suite runs.

---

## Step 1 — Run pytest

All pytest commands must run with `RAG-GPT/` as the working directory because `pyprojroot.here()` resolves paths relative to the working directory.

**If `$ARGUMENTS` is empty**, run the full suite:
```bash
cd RAG-GPT && python -m pytest tests/ -q --tb=short
```

**If `$ARGUMENTS` looks like a path or node ID** (contains `/` or `::`), run it directly:
```bash
cd RAG-GPT && python -m pytest $ARGUMENTS -v --tb=short
```

**If `$ARGUMENTS` is a plain name** (no `/` or `::`), treat it as a keyword filter:
```bash
cd RAG-GPT && python -m pytest tests/ -k "$ARGUMENTS" -v --tb=short
```

Capture exit code. If non-zero, report all failures with the `--tb=short` output and **stop here** — do not proceed to the smoke test.

---

## Step 2 — Start the app (only if Step 1 passed)

Check whether the two servers are already running:
```python
import socket
def port_open(p):
    with socket.socket() as s:
        return s.connect_ex(('127.0.0.1', p)) == 0
```

**If port 8000 is not open**, start the PDF file server in the background:
```bash
cd RAG-GPT && python src/serve.py &
```

**If port 7860 is not open**, start the Gradio app in the background and wait up to 20 seconds for it to become reachable:
```bash
cd RAG-GPT && python src/raggpt_app.py &
```
Poll `http://127.0.0.1:7860` every 2 seconds until it responds or 20 seconds elapse. If it never becomes reachable, report the startup failure and stop.

---

## Step 3 — Send a smoke-test query

Use `gradio_client` to call the `/respond` endpoint directly (no browser needed):

```python
import os, sys
sys.path.insert(0, 'RAG-GPT/src')
os.chdir('RAG-GPT')

from gradio_client import Client

client = Client('http://127.0.0.1:7860', verbose=False)
result = client.predict(
    chatbot=[],
    message="What is CLIP?",
    data_type="Preprocessed doc",
    temperature=0.0,
    api_name="/respond",
)

empty_str, history, retrieved_content = result
```

**Verify the response passes all three checks:**

| Check | Pass condition |
|---|---|
| LLM answered | `history[-1]['role'] == 'assistant'` and `len(history[-1]['content']) > 20` |
| Sources returned | `retrieved_content` is not None and `'# Retrieved content' in retrieved_content` |
| Not an error message | `'VectorDB does not exist'` not in `history[-1]['content']` and `'No file was uploaded'` not in `history[-1]['content']` |

If the "VectorDB does not exist" check fails, instruct the user to run `python RAG-GPT/src/upload_data_manually.py` to build the vector store, then re-run this skill.

---

## Step 4 — Report

Print a single summary block covering both stages:

```
══════════════════════════════════════
 /test-pipeline results
══════════════════════════════════════
pytest       : PASSED  (21/21)   ✓
  or
pytest       : FAILED  (3 failures — see above)

App smoke test : PASSED  ✓
  LLM answer   : "CLIP (Contrastive Language-Image Pre-training) is…"
  Sources      : 3 chunks retrieved from CLIP.pdf
  or
App smoke test : SKIPPED  (pytest failed)
  or
App smoke test : FAILED
  ✗ <which check failed and why>
══════════════════════════════════════
```

For the sources line, count the number of `# Retrieved content` headers in `retrieved_content` and name the source files found in the `Source:` lines. Keep the LLM answer preview to the first 120 characters.

If any stage failed, end with a concise action item telling the user exactly what to fix.
