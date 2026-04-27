---
name: rag-gpt-pipeline
description: Load this skill when working on the RAG-GPT sub-project pipeline — document ingestion, embedding, retrieval, prompt construction, or response formatting. Also load when debugging ChromaDB, LangChain, or Gradio issues in RAG-GPT.
---

# RAG-GPT Pipeline Skill

## Architecture in one sentence

`LoadConfig` reads `configs/app_config.yml` once at import time → `PrepareVectorDB` chunks PDFs and writes ChromaDB → `ChatBot.respond()` retrieves chunks, builds a prompt, calls OpenAI, returns dict-format messages for Gradio 5.

---

## 1. Configuration — `LoadConfig`

**Single source of truth.** Every tuneable parameter lives in `configs/app_config.yml`. `LoadConfig` is instantiated once at module level (`APPCFG = LoadConfig()`) in every util that needs it. There is never more than one config object per process.

```python
APPCFG = LoadConfig()   # top of each utils/*.py file, outside any class
```

**Critical: `persist_directory` must be cast to `str()`.**  
ChromaDB's Rust backend concatenates path components internally and panics if the value is a `pathlib.Path`. Always:

```python
self.persist_directory: str = str(here(app_config["directories"]["persist_directory"]))
```

**Do not call `create_directory(persist_directory)` at startup.**  
If the persist directory is pre-created empty, the `chroma.sqlite3` existence check will never trigger correctly. The directory must only exist when ChromaDB has actually written data to it.

**Config changes require a process restart.** `APPCFG` is module-level; Python's import cache will serve the stale instance.

---

## 2. Document Processing — `PrepareVectorDB`

**Loader:** `PyPDFLoader` from `langchain_community.document_loaders`. Each page becomes one `Document(page_content=..., metadata={"source": path, "page": int, ...})`.

**Splitter:** `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", " ", ""]`. Current settings: `chunk_size=1500`, `chunk_overlap=500`. High overlap is intentional — PDFs contain dense technical text where context bleeds across chunk boundaries.

**Input flexibility:** `data_directory` can be either a `str` (directory path, scanned with `os.listdir`) or a `list` of file paths (for runtime uploads). `PrepareVectorDB.__load_all_documents` handles both:

```python
if isinstance(self.data_directory, list):
    # runtime upload path — each element is a file path
else:
    # pre-processed path — scan directory
```

**Persistence:** `Chroma.from_documents(documents, embedding, persist_directory=...)` writes the SQLite-backed store. The sentinel for "data exists" is the file `{persist_directory}/chroma.sqlite3`, not the directory.

```python
db_file = os.path.join(persist_directory, "chroma.sqlite3")
if not os.path.exists(db_file):
    prepare_and_save_vectordb()
```

**Two separate vectorDBs exist simultaneously:**
- `data/vectordb/processed/chroma/` — pre-processed docs, built once via `upload_data_manually.py`
- `data/vectordb/uploaded/chroma/` — runtime-uploaded docs, wiped on every app restart (`remove_directory` is called in `LoadConfig.__init__`)

---

## 3. Embedding Configuration

**Model:** `text-embedding-ada-002` via `OpenAIEmbeddings()` (reads `OPENAI_API_KEY` from env).

**Important:** `OpenAIEmbeddings()` is instantiated in two places: inside `LoadConfig.__init__` (as `self.embedding_model`) and again inside `PrepareVectorDB.__init__` (as `self.embedding`). These are independent instances but use the same underlying model and key.

**When querying an existing store**, pass the embedding model to `Chroma()`:
```python
vectordb = Chroma(
    persist_directory=APPCFG.persist_directory,
    embedding_function=APPCFG.embedding_model,   # ← must match what was used to build the store
)
```

**Do not change the embedding model after building the store.** The vectors on disk are tied to the model that produced them. Changing the model requires deleting `data/vectordb/` and rebuilding.

---

## 4. Retrieval Logic

**Similarity search:** `vectordb.similarity_search(message, k=APPCFG.k)` returns a list of `Document` objects ranked by cosine similarity. Default `k=3`.

**`k` is the only retrieval lever exposed in config.** There is no score threshold, MMR, or metadata filtering. Adding any of these requires changes only in `chatbot.py` around the `similarity_search` call.

**The raw query text (not the full prompt) is what gets embedded for retrieval.** The retrieval step happens before history and system role are added:

```python
docs = vectordb.similarity_search(message, k=APPCFG.k)   # message = raw user input
```

---

## 5. Prompt Construction — `chatbot.py`

The full prompt passed to the LLM is assembled in this order:

```
Chat history:
User: <prior question>
Assistant: <prior answer>
...

# Retrieved content 1:
<chunk text>
Source: filename.pdf | Page number: N | [View PDF](http://localhost:8000/filename.pdf)

# Retrieved content 2:
...

# User new question:
<current message>
```

**Memory window:** `number_of_q_a_pairs` (default: 2) controls how many past exchanges are included. The code slices `chatbot[-(number_of_q_a_pairs * 2):]` — multiply by 2 because each exchange is two messages (user + assistant).

**System role vs user prompt:** The LLM system role is passed as `{"role": "system", ...}` and instructs the model to answer only from retrieved content. The assembled prompt above is the `{"role": "user", ...}` message. Do not move retrieved content into the system message — it would bloat every call with context that changes per query.

**`clean_references(documents)` — always use object attributes, never parse `str(doc)`.**

```python
# Correct — reads Document object attributes directly
content = doc.page_content
source = os.path.basename(doc.metadata.get("source", "unknown"))
page = doc.metadata.get("page", "?")

# Wrong — LangChain 0.3.x changed str(Document) format; regex breaks silently
content, metadata = re.match(r"page_content=(.*?)( metadata=\{.*\})", str(doc)).groups()
```

---

## 6. LLM Call Pattern

All three call sites (`chatbot.py`, `summarizer.py`, `terminal_q_and_a.py`) use the same pattern via the shared `APPCFG.openai_client`:

```python
response = APPCFG.openai_client.chat.completions.create(
    model=APPCFG.llm_engine,          # "gpt-4o-mini" from app_config.yml
    messages=[
        {"role": "system", "content": APPCFG.llm_system_role},
        {"role": "user", "content": prompt},
    ],
    temperature=temperature,
)
return response.choices[0].message.content  # object-style access — NOT response["choices"][0]
```

**Never use `response["choices"][0]` (dict-style).** That was the openai 0.28 pattern. The 1.x SDK returns objects; dict-style access raises `TypeError`.

---

## 7. Gradio 5 Response Format

`ChatBot.respond()` returns a 3-tuple that maps directly to Gradio output components:

```python
return "", chatbot, retrieved_content
#        ↑          ↑                ↑
#   input_txt   gr.Chatbot      gr.Markdown (reference panel)
```

**Chat history must be a list of dicts.** Gradio 5 requires `type="messages"` on `gr.Chatbot` and `{"role": "user"/"assistant", "content": str}` entries. The old `(user, bot)` tuple format silently breaks rendering.

```python
# Append both messages before returning
chatbot.append({"role": "user", "content": message})
chatbot.append({"role": "assistant", "content": response.choices[0].message.content})
```

**`gr.update()` not a component instance in `.then()` callbacks:**

```python
# Correct
.then(lambda: gr.update(interactive=True), None, [input_txt])

# Wrong — Gradio 4 pattern, breaks in Gradio 5
.then(lambda: gr.Textbox(interactive=True), None, [input_txt])
```

**`gr.LikeData.value` is a dict in Gradio 5** when the chatbot uses `type="messages"`. Always guard:

```python
value = data.value
content = value.get("value", str(value)) if isinstance(value, dict) else str(value)
```

---

## 8. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Old ChromaDB 0.4.x data on disk | Rust panic at startup: `range start index N out of range` | Delete `data/vectordb/` and re-run `upload_data_manually.py` |
| `create_directory(persist_directory)` called at startup | Existence check always True; app thinks DB exists before any data is indexed | Never pre-create the persist dir; check for `chroma.sqlite3` file |
| `print(docs)` or `print(prompt)` in `chatbot.py` | `UnicodeEncodeError` on Windows cp1252 terminal with PDF ligature characters | Remove all debug prints that touch raw PDF text |
| Parsing `str(Document)` with regex | `AttributeError: 'NoneType' has no attribute 'groups'` on LangChain 0.3.x | Use `doc.page_content` and `doc.metadata` directly |
| `response["choices"][0]` dict access | `TypeError` at runtime | Use `response.choices[0].message.content` |
| Tuple `(user, bot)` chat history | Gradio 5 renders no messages, shows error | Use `{"role": ..., "content": ...}` dicts with `type="messages"` |
| `embedding_model` mismatch between build and query | Silent wrong results or ChromaDB dimension error | Embedding model must be identical at build time and query time |
| Changing `number_of_q_a_pairs` to an odd number | Memory slice cuts a user message without its response | Always use `number_of_q_a_pairs * 2` for the slice — one pair = two messages |

---

## 9. Adding a New Data Source

1. Place PDFs in `data/docs/` (or pass file paths as a list to `PrepareVectorDB`).
2. Delete `data/vectordb/processed/chroma/` if it exists.
3. Run `python src/upload_data_manually.py`.
4. The `chroma.sqlite3` sentinel will now exist and the app will load it on next startup.

Do not add non-PDF files without also updating `PrepareVectorDB.__load_all_documents` — it calls `PyPDFLoader` unconditionally.

---

## 10. Extending the Pipeline

**Add a retrieval filter (e.g. by source file):**  
`Chroma.similarity_search` accepts a `filter` kwarg: `vectordb.similarity_search(message, k=k, filter={"source": "data/docs/CLIP.pdf"})`. No other code changes needed.

**Add streaming:**  
Set `stream=True` in `openai_client.chat.completions.create(...)` and yield chunks. The Gradio event handler will need to become a generator — change `outputs` to use `.stream()` instead of a single return.

**Swap the embedding model:**  
Change `engine` in `embedding_model_config`, delete the vectorDB, rebuild. Both `LoadConfig.embedding_model` and `PrepareVectorDB.embedding` use `OpenAIEmbeddings()` which reads the model from the environment or defaults to `text-embedding-ada-002`.

**Add reranking:**  
Insert a reranking step between `similarity_search` and `clean_references` in `ChatBot._respond_inner`. The `docs` list can be reordered or trimmed before being formatted into the prompt.
