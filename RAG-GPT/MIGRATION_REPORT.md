# RAG-GPT Migration Report: Azure OpenAI → OpenAI + Gradio 4 → 5

Branch: `feat/openai-migration`

---

## Summary

RAG-GPT was originally built against Azure OpenAI (`openai==0.28.0`) and Gradio 4. This document records every breaking change encountered during the migration to the standard OpenAI SDK (`openai==1.81.0`) and Gradio 5, the root cause of each failure, and what fixed it.

---

## Part 1 — OpenAI SDK Migration

### What changed

The original code used the Azure-specific `openai` 0.28 API:

```python
import openai
openai.api_type = "azure"
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]
openai.api_key = os.environ["OPENAI_API_KEY"]

response = openai.ChatCompletion.create(
    engine="gpt-35-turbo",
    messages=[...],
)
content = response["choices"][0]["message"]["content"]  # dict-style access
```

The new code uses the standard OpenAI 1.x client:

```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env automatically

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
)
content = response.choices[0].message.content  # object-style access
```

### Files changed

| File | What changed |
|---|---|
| `src/utils/load_config.py` | Removed `load_openai_cfg()` and all Azure env vars. Added `self.openai_client = OpenAI()`. |
| `src/utils/chatbot.py` | `openai.ChatCompletion.create(engine=...)` → `APPCFG.openai_client.chat.completions.create(model=...)`. Response access changed from dict to object. |
| `src/utils/summarizer.py` | Same API call migration. Added `APPCFG = LoadConfig()` singleton instead of calling openai directly. |
| `src/terminal_q_and_a.py` | Same API call migration. |
| `configs/app_config.yml` | `engine: "gpt-35-turbo"` → `engine: "gpt-4o-mini"`. |
| `.env` | Removed `OPENAI_API_TYPE`, `OPENAI_API_BASE`, `OPENAI_API_VERSION`. Only `OPENAI_API_KEY` remains. |

---

## Part 2 — LangChain Migration (forced by OpenAI upgrade)

### Why this was required

LangChain 0.0.354 calls `openai.Embedding.create()` internally. This method was removed in `openai` 1.x. The import structure also changed entirely between 0.0.x and 0.3.x.

### Import changes

| Old import | New import |
|---|---|
| `from langchain.vectorstores import Chroma` | `from langchain_chroma import Chroma` |
| `from langchain.embeddings.openai import OpenAIEmbeddings` | `from langchain_openai import OpenAIEmbeddings` |
| `from langchain.document_loaders import PyPDFLoader` | `from langchain_community.document_loaders import PyPDFLoader` |

### Packages added to `requirements.txt`

```
langchain==0.3.25
langchain-core==0.3.60
langchain-community==0.3.24
langchain-openai==0.3.17
langchain-chroma==0.2.4
```

---

## Part 3 — ChromaDB Migration

### Package upgrade

`chromadb==0.4.22` → `chromadb==1.0.9` (required by `langchain-chroma==0.2.4`).

### Bug 1: Rust backend panic on old persisted data

**Symptom:** App crashed at startup with:
```
range start index 10 out of range for slice of length 9
```

**Root cause:** ChromaDB 1.0 rewrote its storage backend in Rust with a new SQLite schema. The existing `data/vectordb/` directory had been written by ChromaDB 0.4.22 and was binary-incompatible with the new backend.

**Fix:** Delete `data/vectordb/` and regenerate with `python src/upload_data_manually.py`. The old directory is gitignored so this only affects local state.

### Bug 2: Existence check always returning True

**Symptom:** App showed "VectorDB does not exist" even after building it, OR always found the DB even when it hadn't been built.

**Root cause:** `LoadConfig.__init__` called `self.create_directory(self.persist_directory)`, which pre-created the empty directory at startup. Since the existence check was `os.path.exists(directory)`, it always returned `True` even with no data inside.

**Fix:** Removed `create_directory` call from `__init__`. Changed the existence check in `chatbot.py`, `chatbot.py`, and `upload_data_manually.py` to check for the actual database file:

```python
# Before — wrong: checks only if directory exists (always True after create_directory)
if os.path.exists(APPCFG.persist_directory):

# After — correct: only True when ChromaDB has actually written data
db_file = os.path.join(APPCFG.persist_directory, "chroma.sqlite3")
if os.path.exists(db_file):
```

---

## Part 4 — Gradio Migration (4.x → 5.x)

### Bug 3: Chat history format — tuples vs dicts

**Symptom:** Gradio 5 UI showed an error immediately when the app loaded. No messages appeared in the chatbot panel.

**Root cause:** Gradio 4 used `(user_text, bot_text)` tuples for chat history. Gradio 5 requires `{"role": "user"/"assistant", "content": "..."}` dicts and the `gr.Chatbot` must be declared with `type="messages"`.

**Old code (`chatbot.py`, `upload_file.py`):**
```python
chatbot.append((message, response_text))
chatbot.append((message, "Error message"))
```

**New code:**
```python
chatbot.append({"role": "user", "content": message})
chatbot.append({"role": "assistant", "content": response_text})
```

**Also changed in `raggpt_app.py`:**
```python
# Before
chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False, height=500, ...)

# After
chatbot = gr.Chatbot([], elem_id="chatbot", type="messages", height=500, ...)
```

`bubble_full_width` was removed as it was deprecated in Gradio 5.

### Bug 4: Memory string extraction broken after chat format change

**Root cause:** The memory extraction code in `chatbot.py` built the chat history string using `str(chatbot[-N:])`, which serialised the old tuple format. With dict-format messages, the stringified output was unreadable garbage.

**Fix:** Iterate explicitly over the dict messages:

```python
recent = chatbot[-(APPCFG.number_of_q_a_pairs * 2):]
chat_history = "Chat history:\n" + "\n".join(
    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
    for m in recent
) + "\n\n"
```

### Bug 5: `.then()` returning a component instance instead of an update

**Symptom:** After a response, the input textbox stayed disabled.

**Root cause:** In Gradio 4, returning a component instance from an event handler was valid. In Gradio 5, event handlers that update component properties must return `gr.update(...)`.

**Old code (`raggpt_app.py`):**
```python
.then(lambda: gr.Textbox(interactive=True), None, [input_txt], queue=False)
```

**New code:**
```python
.then(lambda: gr.update(interactive=True), None, [input_txt], queue=False)
```

### Bug 6: `gr.LikeData.value` returns a dict in Gradio 5

**Root cause:** In Gradio 5, when the chatbot uses `type="messages"`, clicking like/dislike passes the full message dict as `data.value` instead of a plain string.

**Fix (`ui_settings.py`):**
```python
def feedback(data: gr.LikeData) -> None:
    value = data.value
    if isinstance(value, dict):
        content = value.get("value", str(value))
    else:
        content = str(value)
```

---

## Part 5 — Runtime Bugs (found during manual testing)

### Bug 7: UnicodeEncodeError when printing retrieved documents

**Symptom:** First question submitted in the UI returned an error. Nothing appeared in the chatbot panel. The server log showed:
```
UnicodeEncodeError: 'charmap' codec can't encode character 'ﬁ'
```

**Root cause:** A debug `print(docs)` statement in `chatbot.py` tried to print raw PDF text containing Unicode ligature characters (e.g. `ﬁ` = `ﬁ`) to a Windows terminal using the cp1252 codec. cp1252 cannot encode characters outside the Latin-1 range.

**Fix:** Removed the debug `print(docs)` and `print(prompt)` statements.

### Bug 8: `clean_references()` regex failed on LangChain 0.3.x Document format

**Symptom:** Second question (after the print was fixed) crashed with:
```
AttributeError: 'NoneType' object has no attribute 'groups'
```

**Root cause:** `clean_references()` parsed `str(document)` with a regex to extract `page_content` and `metadata`. The string format changed between LangChain versions:

```
# LangChain 0.0.x format (old regex target)
page_content=Some text metadata={'page': 0, 'source': '...'}

# LangChain 0.3.x format (new — regex didn't match)
page_content='Some text' metadata={'page': 0, 'source': '...', 'creationdate': '...', 'author': '...'}
```

The regex `r"page_content=(.*?)( metadata=\{.*\})"` never matched the new format (quoted content, extra metadata fields), so `.groups()` was called on `None`.

**Fix:** Stopped parsing `str(document)` entirely. Read from Document object attributes directly:

```python
# Before — fragile string parsing
content, metadata = re.match(r"page_content=(.*?)( metadata=\{.*\})", doc).groups()
metadata_dict = ast.literal_eval(metadata.split("=", 1)[1])

# After — reading object attributes directly
content = doc.page_content
source = os.path.basename(doc.metadata.get("source", "unknown"))
page = doc.metadata.get("page", "?")
```

This also removed the `ast` import and several brittle string-encoding workarounds that were compensating for the double-encoding artefacts introduced by `bytes(...).decode("unicode_escape")`.

### Bug 9: `upload_data_manually.py` crashed when vectordb directory didn't exist

**Symptom:** Running `python src/upload_data_manually.py` on a fresh clone raised `FileNotFoundError`.

**Root cause:** After removing `create_directory` from `LoadConfig.__init__`, the persist directory no longer existed at startup. The script called `os.listdir(persist_directory)` to check for existing data, which threw `FileNotFoundError` on a missing directory.

**Fix:** Same `chroma.sqlite3` sentinel check as in `chatbot.py`:

```python
db_file = os.path.join(CONFIG.persist_directory, "chroma.sqlite3")
if not os.path.exists(db_file):
    prepare_vectordb_instance.prepare_and_save_vectordb()
```

---

## Test Suite Added

No tests existed before this migration. 21 pytest tests were added across 5 files covering all migrated components. All tests mock the OpenAI client and ChromaDB — no live API calls or local files required.

```
tests/conftest.py              — shared fixtures (mock_openai_client, mock_vectordb)
tests/test_load_config.py      — LoadConfig attributes and OpenAI client instantiation
tests/test_chatbot.py          — ChatBot.respond() success, error, and model routing paths
tests/test_summarizer.py       — Summarizer.get_llm_response() and summarize_the_pdf()
tests/test_prepare_vectordb.py — PrepareVectorDB chunking and Chroma.from_documents()
tests/test_gradio_ui.py        — Gradio 5 dict-format message assertions across all paths
```

Run with:
```bash
cd RAG-GPT
pytest tests/ -v
```
