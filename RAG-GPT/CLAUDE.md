# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sub-project: RAG-GPT

Document chat chatbot using OpenAI, LangChain, ChromaDB, and Gradio. Run from this directory.

```bash
pip install -r requirements.txt

# Terminal 1 — PDF file server (port 8000)
python src/serve.py

# Terminal 2 — Gradio UI (port 7860)
python src/raggpt_app.py
```

Pre-process documents into the vectorDB before first use:
```bash
python src/upload_data_manually.py
```

## Active Migration: Azure OpenAI → OpenAI

This sub-project is being migrated from Azure OpenAI (`openai==0.28.0`) to the standard OpenAI SDK.

**Current legacy pattern (being replaced):**
```python
import openai
openai.api_type = "azure"
openai.api_base = ...
openai.ChatCompletion.create(engine="gpt-35-turbo", ...)
```

**Target pattern:**
```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env
client.chat.completions.create(model="gpt-4o-mini", ...)
```

Key files to migrate: `src/utils/load_config.py` (Azure credential setup in `load_openai_cfg`), `src/utils/chatbot.py` (`openai.ChatCompletion.create`), `src/utils/summarizer.py`.

After migration, `OPENAI_API_TYPE`, `OPENAI_API_BASE`, and `OPENAI_API_VERSION` can be removed from `.env`. Only `OPENAI_API_KEY` is needed.

## Development Rules

**Type hints** — all functions must have type-annotated signatures.

**Dependencies** — update `requirements.txt` whenever a package is added, removed, or version-bumped.

**Tests first** — there are currently no tests. Before changing any pipeline logic, write pytest tests in `tests/` that verify the existing behavior, then make the change.

```bash
pytest tests/
pytest tests/test_chatbot.py::test_name   # single test
```

**Gradio UI** — keep the UI functional throughout all changes. The three-panel layout (chatbot, references, controls) must not regress. Test it manually after any change to `src/raggpt_app.py` or the utils it imports.

## Architecture

```
configs/app_config.yml   ← all tuneable settings (model, chunking, retrieval k, memory)
src/
  raggpt_app.py          ← Gradio UI entry point
  serve.py               ← HTTP server that serves PDFs for the reference viewer
  upload_data_manually.py← one-shot script to build the preprocessed vectorDB
  utils/
    load_config.py       ← LoadConfig singleton; reads app_config.yml at import time
    chatbot.py           ← ChatBot.respond() — retrieval + LLM completion
    summarizer.py        ← map-reduce summarization over large docs
    upload_file.py       ← handles runtime document uploads → custom vectorDB
    prepare_vectordb.py  ← shared chunking/embedding helpers
    ui_settings.py       ← Gradio component configuration
data/
  docs/                  ← pre-processed documents (served by serve.py)
  docs_2/                ← runtime-uploaded documents
  vectordb/              ← gitignored; generated locally
```

`LoadConfig` is instantiated once at module import (`APPCFG = LoadConfig()`). Config changes require a process restart.
