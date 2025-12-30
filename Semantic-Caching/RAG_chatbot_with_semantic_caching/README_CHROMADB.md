# TaskFlow RAG Chatbot with ChromaDB & Semantic Caching 🚀

A production-ready RAG chatbot demonstrating semantic caching with **persistent ChromaDB vectorstore**. This project shows how to build an efficient RAG system that combines document retrieval with intelligent caching.

## 🌟 Key Features

### 1. **Persistent ChromaDB VectorStore**
- Documents stored in ChromaDB with embeddings
- Survives restarts - no need to re-embed
- Efficient similarity search
- Based on real FAQ and documentation data

### 2. **Semantic Caching Layer**
- Instant responses for similar questions
- OpenAI embeddings for semantic matching
- User feedback loop for cache quality
- ~100ms response time for cache hits vs 3-5s for RAG

### 3. **Full LangGraph RAG Pipeline**
- Intelligent document retrieval
- Relevance grading
- Question rewriting
- Context-aware answer generation

### 4. **Interactive Streamlit UI**
- Visual cache hit/miss indicators
- Real-time statistics
- User feedback system
- Threshold tuning

## 📁 Project Structure

```
.
├── cached_rag_chatbot_chroma.py    # Main chatbot with ChromaDB
├── document_store_chroma.py        # ChromaDB vectorstore manager
├── semantic_cache.py               # Semantic cache implementation
├── prepare_data_chroma.py          # Data preparation script ⭐
├── streamlit_app.py                # Interactive UI
├── demo_chatbot_chroma.py          # Command-line demo
│
├── taskflow_faq.csv                # 30 FAQ entries (source data)
├── taskflow_docs.txt               # Main documentation
├── taskflow_cache_seed.csv         # Initial cache (8 pairs)
│
├── requirements.txt                # Dependencies
└── README_CHROMADB.md             # This file
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `chromadb` - Vector database
- `langchain-chroma` - LangChain ChromaDB integration
- `langchain-openai` - OpenAI embeddings and models
- `langgraph` - Orchestration framework
- `streamlit` - Interactive UI

### Step 2: Set API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Step 3: Prepare Data (Create ChromaDB)

**This is the crucial step** - it creates the persistent vectorstore:

```bash
python prepare_data_chroma.py
```

This script:
1. ✅ Loads 30 FAQ entries as documents
2. ✅ Loads main TaskFlow documentation
3. ✅ Creates ChromaDB vectorstore in `./chroma_db/`
4. ✅ Generates embeddings for all documents
5. ✅ Prepares semantic cache from seed data
6. ✅ Verifies retrieval is working

**Output:**
```
🗄️  STEP 1: Creating Document VectorStore (ChromaDB)
📥 Loading source data...
📚 Loading FAQ data as documents...
  ✓ Loaded 30 FAQ entries as documents
📖 Loading main documentation...
  ✓ Loaded main documentation (XX characters)

🔨 Creating ChromaDB vectorstore...
✅ ChromaDB vectorstore created successfully!
   Collection: taskflow_docs
   Total chunks: 150+
   Location: ./chroma_db

💾 STEP 2: Preparing Semantic Cache
✅ Semantic cache prepared with 8 pairs

🎉 DATA PREPARATION COMPLETE!
```

**Important:** The ChromaDB is now persisted in `./chroma_db/` and will be reused!

### Step 4: Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will:
- Load the existing ChromaDB (instant!)
- Load the semantic cache
- Be ready to answer questions

### Step 5: Try It Out!

**Questions that will hit cache (⚡ instant):**
- "How do I create a new project?"
- "What are the pricing plans?"
- "Can I integrate with Slack?"

**Questions that will use RAG (🔍 3-5s):**
- "How do I set up automations?"
- "What's the difference between workspaces and projects?"
- "Can I customize task fields?"

## 🏗️ Architecture

### Data Flow

```
User Question
     ↓
[Semantic Cache Check]
     ↓
Hit? ──YES─→ Return Cached Answer (100ms) ✨
     ↓ NO
[ChromaDB Similarity Search]
     ↓
[Retrieve Top-K Documents]
     ↓
[Grade Document Relevance]
     ↓
Relevant? ──NO─→ Rewrite Question → Retry
     ↓ YES
[Generate Answer with Context]
     ↓
[Show to User]
     ↓
[User Feedback]
     ↓
Approved? ──YES─→ Add to Cache 💾
```

### Components

1. **ChromaDB VectorStore** (`./chroma_db/`)
   - Persistent document embeddings
   - ~150+ chunks from FAQ + docs
   - OpenAI embeddings (1536 dimensions)

2. **Semantic Cache** (CSV files)
   - In-memory for fast lookup
   - OpenAI embeddings
   - Cosine distance matching

3. **LangGraph Pipeline**
   - Tool-based retrieval
   - Conditional routing
   - Multi-step reasoning

## 📊 Data Preparation Details

### What Gets Stored in ChromaDB?

The `prepare_data_chroma.py` script processes:

**1. FAQ Entries (30 items)**
Each FAQ becomes a document:
```
Q: How do I create a new project?

A: To create a new project, click the '+' button...
```

**2. Main Documentation (10,000+ words)**
Split into chunks:
- Chunk size: 500 tokens
- Chunk overlap: 50 tokens
- Result: ~120 chunks

**Total in ChromaDB:** ~150 document chunks

### ChromaDB Structure

```
./chroma_db/
├── chroma.sqlite3          # Metadata database
└── [embedding data]        # Vector embeddings
```

**Collection:** `taskflow_docs`
**Embeddings:** OpenAI `text-embedding-ada-002` (1536 dimensions)

### Re-running Data Prep

```bash
# Force recreate (deletes existing)
python prepare_data_chroma.py --force

# Custom directory
python prepare_data_chroma.py --persist-dir ./my_chroma_db
```

## 💡 How Semantic Caching Works

### Cache Lookup Process

1. **User asks:** "What's the cost?"
2. **Embed question** using OpenAI
3. **Calculate cosine distance** to all cached questions
4. **If distance < threshold (0.35):** Return cached answer
5. **Else:** Run full RAG pipeline

### Example Matches

| User Question | Cached Question | Distance | Match? |
|--------------|----------------|----------|--------|
| "What's the cost?" | "What are the pricing plans?" | 0.28 | ✅ Yes |
| "How to make a project?" | "How do I create a new project?" | 0.22 | ✅ Yes |
| "Tell me about Slack" | "Can I integrate with Slack?" | 0.31 | ✅ Yes |
| "What's the weather?" | "What are the pricing plans?" | 0.85 | ❌ No |

### Threshold Tuning

- **0.2** - Very strict (only nearly identical)
- **0.35** - **Recommended** (good balance)
- **0.5** - Loose (more matches, some false positives)

Adjust in Streamlit sidebar or code:
```python
chatbot = CachedRAGChatbot(cache_distance_threshold=0.35)
```

## 🎯 Usage Examples

### Command Line Demo

```bash
python demo_chatbot_chroma.py
```

### Programmatic Usage

```python
from cached_rag_chatbot_chroma import CachedRAGChatbot

# Initialize
chatbot = CachedRAGChatbot(
    cache_distance_threshold=0.35,
    chroma_persist_dir="./chroma_db",
    chroma_collection="taskflow_docs"
)

# Load existing ChromaDB
chatbot.load_existing_vectorstore()

# Load cache
chatbot.load_cache_from_file("taskflow_cache_seed.csv")

# Query
result = chatbot.query("How do I create a project?")
print(f"Answer: {result['answer']}")
print(f"From cache: {result['cache_hit']}")

# Add to cache
if not result['cache_hit']:
    chatbot.add_to_cache(question, result['answer'])
    chatbot.save_cache_to_file("updated_cache.csv")
```

## 📈 Performance Metrics

### Response Times

| Scenario | Time | Description |
|----------|------|-------------|
| **Cache Hit** | ~100ms | Semantic similarity search only |
| **Cache Miss - Simple** | 3-5s | ChromaDB retrieval + LLM generation |
| **Cache Miss - Complex** | 5-8s | Multiple retrievals + rewriting |
| **Subsequent Similar** | ~100ms | Now in cache! |

### Cache Hit Rate Example

After 20 queries in a session:
```
Total Queries: 20
Cache Hits: 12 (60%)
Cache Misses: 8 (40%)
Approved Additions: 5
```

60% of questions answered instantly!

## 🔧 Configuration

### ChromaDB Settings

```python
doc_store = DocumentVectorStore(
    persist_directory="./chroma_db",      # Where to store
    collection_name="taskflow_docs",       # Collection name
    chunk_size=500,                        # Tokens per chunk
    chunk_overlap=50                       # Chunk overlap
)
```

### Cache Settings

```python
cache = SemanticCache(
    distance_threshold=0.35  # Similarity threshold
)
```

### RAG Retrieval Settings

```python
retriever = doc_store.get_retriever(
    k=4,                          # Number of documents
    search_type="similarity"      # or "mmr"
)
```

## 🧪 Testing

### Verify ChromaDB

```python
from document_store_chroma import DocumentVectorStore

doc_store = DocumentVectorStore(persist_directory="./chroma_db")
doc_store.load_existing()

# Test search
results = doc_store.similarity_search("How do I create a project?", k=3)
for doc in results:
    print(doc.page_content[:100])
```

### Test Cache

```python
from semantic_cache import SemanticCache

cache = SemanticCache.load_from_file("taskflow_cache_seed.csv")

result = cache.check("What's the cost?")
if result.hit:
    print(f"Match: {result.best_match.prompt}")
    print(f"Answer: {result.best_match.response}")
```

## 📚 Understanding the Data

### FAQ Dataset (taskflow_faq.csv)

30 comprehensive Q&A pairs covering:
- Account management
- Project creation
- Task management
- Team collaboration
- Integrations
- Pricing
- Security
- Mobile apps

### Documentation (taskflow_docs.txt)

10,000+ word comprehensive guide covering:
- Getting started
- Core features
- Subscription plans
- Team collaboration
- Advanced features
- Mobile apps
- Security & privacy
- API documentation
- Best practices

### Cache Seed (taskflow_cache_seed.csv)

8 high-quality Q&A pairs for instant responses:
- Project creation
- Data export
- Pricing
- Password reset
- Slack integration
- Task assignment
- File uploads
- Account deletion

## 🚨 Troubleshooting

### "ChromaDB not found"

**Problem:** Vectorstore doesn't exist

**Solution:**
```bash
python prepare_data_chroma.py
```

### "Collection is empty"

**Problem:** ChromaDB exists but has no data

**Solution:**
```bash
python prepare_data_chroma.py --force
```

### Low cache hit rate

**Problem:** Threshold too strict

**Solution:** Increase threshold to 0.4 or 0.45

### Too many false cache hits

**Problem:** Threshold too loose

**Solution:** Decrease threshold to 0.25 or 0.3

### Slow queries

**Problem:** Too many documents retrieved

**Solution:** Reduce `k` parameter:
```python
retriever = doc_store.get_retriever(k=2)  # Instead of k=4
```

## 🎓 Learning Outcomes

This project demonstrates:

1. ✅ **Persistent Vector Databases** - ChromaDB setup and usage
2. ✅ **Semantic Caching** - Meaning-based response caching
3. ✅ **RAG Architecture** - Retrieval-augmented generation
4. ✅ **LangGraph Workflows** - Complex LLM orchestration
5. ✅ **User Feedback Loops** - Cache quality improvement
6. ✅ **Performance Optimization** - Balancing speed and accuracy

## 🔄 Updating the Knowledge Base

### Add New Documents

```python
from document_store_chroma import DocumentVectorStore
from langchain_core.documents import Document

doc_store = DocumentVectorStore(persist_directory="./chroma_db")
doc_store.load_existing()

new_docs = [
    Document(page_content="New content here...", metadata={"source": "new"})
]

doc_store.add_documents(new_docs)
```

### Add to Cache

```python
chatbot.add_to_cache(
    question="How do I do X?",
    answer="To do X, you..."
)
chatbot.save_cache_to_file("updated_cache.csv")
```

## 🌐 Deployment Considerations

For production:

1. **Use cloud-hosted ChromaDB** (Chroma Cloud)
2. **Implement rate limiting**
3. **Add authentication**
4. **Use production models** (gpt-4o instead of gpt-4o-mini)
5. **Monitor cache hit rates**
6. **Regular cache cleanup**
7. **Add logging and analytics**
8. **Implement feedback analytics**

## 📊 Comparison: With vs Without Caching

### Without Semantic Caching
```
Query 1: "How do I create a project?" → RAG (4.2s)
Query 2: "How to make a new project?" → RAG (4.5s)
Query 3: "What's the project creation process?" → RAG (4.1s)

Total time: 12.8s
API calls: 3
Cost: 3× API calls
```

### With Semantic Caching
```
Query 1: "How do I create a project?" → Cache hit (0.1s)
Query 2: "How to make a new project?" → Cache hit (0.1s)
Query 3: "What's the project creation process?" → Cache hit (0.1s)

Total time: 0.3s
API calls: 0
Cost: 0× API calls
Speedup: 43x faster! 🚀
```

## 📝 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

Built with:
- ChromaDB for persistent vector storage
- LangChain & LangGraph for RAG orchestration
- OpenAI for embeddings and language models
- Streamlit for the interactive UI

---

**Ready to build your own semantic cached RAG system?** 🚀

```bash
# Setup in 3 commands
pip install -r requirements.txt
python prepare_data_chroma.py
streamlit run streamlit_app.py
```

Happy building! 🎉
