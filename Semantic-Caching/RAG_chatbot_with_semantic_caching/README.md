# Semantic Cached RAG Chatbot 🚀

A production-ready RAG (Retrieval-Augmented Generation) chatbot with semantic caching using LangGraph and OpenAI embeddings. This system intelligently caches Q&A pairs and retrieves them based on semantic similarity, dramatically reducing API calls and improving response times.

## 🌟 Features

- **Semantic Caching**: Understands that "How do I get a refund?" means the same as "I want my money back"
- **Full RAG Pipeline**: Uses LangGraph for orchestrated document retrieval and answer generation
- **Auto-Cache Updates**: Automatically adds new Q&A pairs to the cache
- **Dual Vector Stores**: 
  - One for semantic cache (Q&A pairs)
  - One for document retrieval (RAG corpus)
- **OpenAI Integration**: Uses OpenAI embeddings and GPT-4 for high-quality results
- **Persistent Cache**: Save and load cache from CSV files
- **Interactive Mode**: Chat with the bot in real-time

## 📁 Project Structure

```
.
├── semantic_cache.py          # Semantic cache implementation with OpenAI embeddings
├── document_store.py           # Document vectorstore for RAG
├── cached_rag_chatbot.py       # Main chatbot with LangGraph orchestration
├── example_usage.py            # Example scripts and interactive mode
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up your OpenAI API key**:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or on Windows:
```cmd
set OPENAI_API_KEY=your-api-key-here
```

## 🚀 Quick Start

### Basic Usage

```python
from cached_rag_chatbot import CachedRAGChatbot

# Initialize chatbot
chatbot = CachedRAGChatbot(
    cache_distance_threshold=0.35,  # Lower = stricter matching
    model_name="gpt-4o",
    temperature=0
)

# Load initial Q&A pairs into cache
initial_cache = [
    ("What is machine learning?", 
     "Machine learning is a subset of AI that enables systems to learn from data."),
    ("How do neural networks work?", 
     "Neural networks are computing systems inspired by biological neural networks."),
]
chatbot.load_cache_pairs(initial_cache)

# Load documents for RAG
urls = [
    "https://example.com/article1",
    "https://example.com/article2",
]
chatbot.load_documents_from_urls(urls)

# Query the chatbot
result = chatbot.query("What is ML?", verbose=True)
print(result["answer"])
print(f"Cache hit: {result['cache_hit']}")
```

### Run Example Script

```bash
# Run the demo
python example_usage.py

# Run interactive mode
python example_usage.py interactive
```

## 📚 How It Works

### 1. Query Flow

```
User Question
     ↓
[Check Semantic Cache]
     ↓
Cache Hit? ─YES→ Return Cached Answer ✨
     ↓ NO
[Full RAG Pipeline]
     ↓
[Retrieve Documents]
     ↓
[Grade Relevance]
     ↓
Relevant? ─NO→ [Rewrite Question] → Retry
     ↓ YES
[Generate Answer]
     ↓
[Add to Cache] 💾
     ↓
Return Answer
```

### 2. Semantic Cache

The semantic cache uses **OpenAI embeddings** to understand question similarity:

- **Distance Threshold**: Controls matching strictness (default: 0.3)
  - Lower (e.g., 0.2): Stricter, only very similar questions match
  - Higher (e.g., 0.5): Looser, more variations match

**Example Matches**:
- "What is machine learning?" ↔️ "Can you explain ML?"
- "How do I get a refund?" ↔️ "I want my money back"
- "What's the weather like?" ↔️ "Tell me about the weather"

### 3. RAG Pipeline (LangGraph)

When cache misses, the system:
1. Uses LLM to decide: retrieve documents or answer directly
2. Retrieves relevant documents from vectorstore
3. Grades document relevance
4. Rewrites question if documents aren't relevant
5. Generates final answer using retrieved context
6. Adds Q&A to cache for future use

## 🎯 API Reference

### `CachedRAGChatbot`

#### Initialization
```python
chatbot = CachedRAGChatbot(
    cache_distance_threshold=0.35,  # Similarity threshold
    model_name="gpt-4o",            # LLM model
    temperature=0                    # LLM temperature
)
```

#### Loading Data
```python
# Load cache from Q&A pairs
chatbot.load_cache_pairs([("question", "answer"), ...])

# Load cache from CSV file
chatbot.load_cache_from_file("cache.csv")

# Load documents from URLs
chatbot.load_documents_from_urls(["https://...", ...])

# Load documents from directory
chatbot.load_documents_from_directory("./docs", glob_pattern="**/*.txt")
```

#### Querying
```python
# Single query with full details
result = chatbot.query("Your question here", verbose=True)
# Returns: {
#   "answer": str,
#   "cache_hit": bool,
#   "cache_info": dict or None
# }

# Streaming query (for interactive use)
for update in chatbot.query_stream("Your question"):
    # Handle different update types: cache_hit, cache_miss, node_update, complete
    pass
```

#### Cache Management
```python
# Save cache to file
chatbot.save_cache_to_file("cache_data.csv")

# Get all cached pairs
pairs = chatbot.cache.get_all_pairs()
```

## ⚙️ Configuration

### Cache Distance Threshold

Adjust the `cache_distance_threshold` parameter:

```python
# Strict matching (fewer false positives)
chatbot = CachedRAGChatbot(cache_distance_threshold=0.2)

# Moderate matching (balanced)
chatbot = CachedRAGChatbot(cache_distance_threshold=0.35)

# Loose matching (more matches, some false positives)
chatbot = CachedRAGChatbot(cache_distance_threshold=0.5)
```

### Document Chunking

Customize in `DocumentVectorStore`:

```python
from document_store import DocumentVectorStore

doc_store = DocumentVectorStore(
    chunk_size=500,    # Tokens per chunk
    chunk_overlap=50   # Overlap between chunks
)
```

## 💡 Use Cases

1. **Customer Support Chatbots**: Cache common questions, retrieve from docs for uncommon ones
2. **Technical Documentation**: Fast answers to repeated questions, detailed answers from docs
3. **Educational Assistants**: Common concept explanations cached, complex queries use RAG
4. **Product Q&A**: Product-specific questions cached, general queries use documentation

## 🔍 Example Scenarios

### Scenario 1: Cache Hit
```
User: "What is machine learning?"
System: ✅ CACHE HIT! (distance: 0.05)
Answer: "Machine learning is a subset of artificial intelligence..."
Response Time: ~100ms
```

### Scenario 2: Cache Miss → RAG
```
User: "What does the author say about reward hacking?"
System: ❌ CACHE MISS - Running RAG pipeline...
System: 📄 Retrieved 4 relevant documents
System: ✅ Generated answer from context
Answer: "According to the author, reward hacking occurs when..."
System: 💾 Added to cache
Response Time: ~3-5s
```

### Scenario 3: Semantic Match
```
User: "Tell me about ML"
System: ✅ CACHE HIT! (matched: "What is machine learning?", distance: 0.28)
Answer: "Machine learning is a subset of artificial intelligence..."
Response Time: ~100ms
```

## 🐛 Troubleshooting

**Issue**: `ValueError: Vectorstore not initialized`
- **Solution**: Make sure to call `load_documents_from_urls()` or `load_documents_from_directory()` before querying

**Issue**: Too many cache misses
- **Solution**: Increase `cache_distance_threshold` (e.g., from 0.3 to 0.4)

**Issue**: Too many false positive cache hits
- **Solution**: Decrease `cache_distance_threshold` (e.g., from 0.4 to 0.25)

**Issue**: OpenAI API errors
- **Solution**: Ensure `OPENAI_API_KEY` is set correctly in environment variables

## 📊 Performance Tips

1. **Batch load cache pairs** at startup rather than adding one-by-one
2. **Save cache periodically** to avoid losing accumulated Q&A pairs
3. **Monitor cache hit rate** to optimize the distance threshold
4. **Use appropriate chunk sizes** for your documents (smaller for FAQs, larger for articles)

## 🔐 Security Notes

- Never commit your `OPENAI_API_KEY` to version control
- Use environment variables for sensitive configuration
- Implement rate limiting for production deployments
- Sanitize user inputs before processing

## 📝 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- Based on LangChain and LangGraph frameworks
- Uses OpenAI's embedding and language models
- Inspired by semantic caching patterns in modern AI systems

---

**Happy caching!** 🎉 If you have questions or issues, feel free to open an issue or contribute to the project.
