"""
About Semantic Caching - Educational Page
"""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="About Semantic Caching",
                   page_icon="📚", layout="wide")

st.title("📚 About Semantic Caching")
st.markdown("### Understanding the concepts behind this dashboard")

st.markdown("---")

# What is semantic caching
st.markdown("## 🤔 What is Semantic Caching?")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Semantic caching** uses embeddings (vector representations) to understand the *meaning* 
    of queries, not just exact text matches.
    
    ### Traditional Cache vs Semantic Cache
    
    **Traditional Cache (Exact Match):**
    ```
    Query: "How do I get a refund?"
    Match: "How do I get a refund?" ✅
    Miss:  "I want my money back" ❌ (same meaning, different words)
    ```
    
    **Semantic Cache:**
    ```
    Query: "How do I get a refund?"
    Match: "How do I get a refund?" ✅
    Match: "I want my money back" ✅ (understands meaning!)
    Match: "refund process help" ✅
    ```
    """)

with col2:
    st.info("""
    **Key Insight**
    
    Semantic caching understands that:
    - "refund" ≈ "money back"
    - "password reset" ≈ "forgot password"
    - "order tracking" ≈ "where is my package"
    
    This dramatically increases cache hit rate!
    """)

st.markdown("---")

# How it works
st.markdown("## ⚙️ How Does It Work?")

tab1, tab2, tab3 = st.tabs(
    ["1️⃣ Embeddings", "2️⃣ Distance Calculation", "3️⃣ Threshold Matching"])

with tab1:
    st.markdown("""
    ### Converting Text to Vectors (Embeddings)
    
    Each question is converted into a high-dimensional vector (e.g., 384 or 768 dimensions) 
    that captures its semantic meaning.
    
    **Example:**
    ```python
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    q1 = "How do I get a refund?"
    q2 = "I want my money back"
    
    embedding1 = model.encode(q1)  # [0.23, -0.45, 0.12, ...]
    embedding2 = model.encode(q2)  # [0.21, -0.48, 0.09, ...]
    ```
    
    **Key Properties:**
    - Similar meanings → Similar vectors
    - Different meanings → Different vectors
    - Language-agnostic (with multilingual models)
    """)

with tab2:
    st.markdown("""
    ### Measuring Similarity (Distance Metrics)
    
    We calculate how "close" two vectors are using **cosine distance**:
    
    ```python
    # Cosine similarity (0 to 1, higher = more similar)
    similarity = dot(v1, v2) / (norm(v1) * norm(v2))
    
    # Cosine distance (0 to 1, lower = more similar)
    distance = 1 - similarity
    ```
    
    **Distance Values:**
    - `0.0` - Identical meaning (exact same query)
    - `0.1-0.3` - Very similar (paraphrases, synonyms)
    - `0.4-0.6` - Somewhat related
    - `0.7+` - Different topics
    
    **Example:**
    ```python
    query = "How do I get a refund?"
    
    FAQ entries with distances:
    - "What is the refund policy?" → 0.14 ✅ (very close)
    - "Can I reset my password?" → 0.78 ❌ (different topic)
    - "I want my money back" → 0.18 ✅ (paraphrase)
    ```
    """)

with tab3:
    st.markdown("""
    ### Threshold-Based Matching
    
    The **threshold** determines when we consider a match "close enough":
    
    ```python
    def check_cache(query, threshold=0.35):
        query_embedding = encode(query)
        
        # Find closest FAQ entry
        distances = [distance(query_embedding, faq_emb) 
                    for faq_emb in cache_embeddings]
        
        best_match_distance = min(distances)
        
        if best_match_distance <= threshold:
            return cached_answer  # Cache HIT
        else:
            return None  # Cache MISS → Call LLM
    ```
    
    **Threshold Trade-offs:**
    
    | Threshold | Behavior | Use Case |
    |-----------|----------|----------|
    | 0.10-0.20 | Strict matching | High-stakes (medical, legal) |
    | 0.25-0.35 | Balanced | Most applications |
    | 0.40-0.50 | Lenient | Maximize cache hits |
    
    **Lower threshold** = Higher precision, lower recall
    **Higher threshold** = Lower precision, higher recall
    """)

st.markdown("---")

# Caching strategies
st.markdown("## 🎯 Caching Strategies")

st.markdown("This dashboard compares three strategies:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🎯 Exact Match
    
    **How it works:**
    - Case-insensitive string comparison
    - Must match character-by-character
    
    **Pros:**
    - ⚡ Fastest (no computation)
    - 🎯 100% precision
    
    **Cons:**
    - ❌ 0% recall on paraphrases
    - ❌ Typo-sensitive
    
    **Best for:**
    - Limited, fixed queries
    - API endpoints with exact formats
    """)

with col2:
    st.markdown("""
    ### 🔤 Fuzzy Match
    
    **How it works:**
    - Levenshtein distance
    - Tolerates typos and variations
    
    **Pros:**
    - 🔤 Handles typos
    - ⚡ Faster than semantic
    
    **Cons:**
    - ❌ Still misses paraphrases
    - ⚠️ Can match unrelated strings
    
    **Best for:**
    - User input with typos
    - Mobile keyboards
    """)

with col3:
    st.markdown("""
    ### 🧠 Semantic Cache
    
    **How it works:**
    - Vector embeddings
    - Cosine distance matching
    
    **Pros:**
    - ✅ Understands meaning
    - 🎯 High recall
    - 🌐 Multilingual support
    
    **Cons:**
    - 🐢 Slower (embedding computation)
    - 💾 Higher memory usage
    
    **Best for:**
    - Natural language queries
    - Production LLM apps
    """)

st.markdown("---")

# Reranking
st.markdown("## 🎯 Reranking Strategies")

st.markdown("""
**Problem:** Semantic cache can have false positives (return wrong cached answer).

**Solution:** Add a reranking step to validate matches before returning them.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Two-Stage Retrieval
    
    ```
    Stage 1: Semantic Cache (Recall)
    ↓
    Retrieve top-K candidates (e.g., K=5)
    ↓
    Stage 2: Reranker (Precision)
    ↓
    Validate and reorder candidates
    ↓
    Return best match (if validated)
    ```
    
    This approach:
    - Maximizes recall (find potential matches)
    - Ensures precision (filter false positives)
    """)

with col2:
    st.markdown("""
    ### Reranking Methods
    
    1. **Simple Keyword** - Fast rule-based matching
       - Boost matches with keyword overlap
       - No external models needed
       
    2. **Cross-Encoder** - Neural semantic validation
       - Better than simple keywords
       - Balanced speed/accuracy
       
    3. **LLM** - Highest accuracy with reasoning
       - GPT-based validation
       - Provides explanations
       - Slowest but most accurate
    """)

st.markdown("---")

# Key metrics
st.markdown("## 📊 Key Metrics")

st.markdown("This dashboard tracks several important metrics:")

metrics_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Hit Rate'],
    'Definition': [
        'Of all cache hits, how many were correct?',
        'Of all queries that should hit, how many did?',
        'Harmonic mean of precision and recall',
        'Overall percentage of correct predictions',
        'Percentage of queries served from cache'
    ],
    'Goal': [
        'High (avoid false positives)',
        'High (avoid false negatives)',
        'High (balanced performance)',
        'High (overall correctness)',
        'Moderate (40-60% is typical)'
    ]
})

st.dataframe(metrics_df, hide_index=True)

st.markdown("---")

# Use cases
st.markdown("## 💼 Common Use Cases")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Customer Support", "E-commerce", "Healthcare", "Enterprise"])

with tab1:
    st.markdown("""
    ### 📞 Customer Support Chatbots
    
    **Scenario:** 24/7 support bot answering common questions
    
    **Requirements:**
    - Handle paraphrases of same question
    - Fast response time
    - Consistent answers
    
    **Recommended Strategy:**
    - Semantic cache (threshold: 0.30-0.35)
    - Cross-encoder reranker
    - Expected hit rate: 50-70%
    
    **Benefits:**
    - Reduce support tickets by 60%
    - Instant responses 24/7
    - Consistent quality
    """)

with tab2:
    st.markdown("""
    ### 🛒 E-commerce FAQ
    
    **Scenario:** Product questions, shipping, returns
    
    **Requirements:**
    - High accuracy (wrong info hurts sales)
    - Moderate speed
    - Handle product variations
    
    **Recommended Strategy:**
    - Semantic cache (threshold: 0.25)
    - LLM reranker for validation
    - Expected hit rate: 40-50%
    
    **Benefits:**
    - Reduce LLM costs by 50%+
    - Maintain high accuracy
    - Scale during peak seasons
    """)

with tab3:
    st.markdown("""
    ### 🏥 Healthcare Information
    
    **Scenario:** Medical FAQ, symptom checker
    
    **Requirements:**
    - Highest accuracy (critical domain)
    - False positives unacceptable
    - Regulatory compliance
    
    **Recommended Strategy:**
    - Semantic cache (threshold: 0.15-0.20)
    - LLM reranker with reasoning
    - Conservative hit rate: 20-30%
    
    **Benefits:**
    - Ensure accuracy in critical domain
    - Audit trail with LLM reasoning
    - Reduce API costs while maintaining safety
    """)

with tab4:
    st.markdown("""
    ### 🏢 Enterprise Knowledge Base
    
    **Scenario:** Internal documentation, policies, procedures
    
    **Requirements:**
    - Handle domain-specific terms
    - Multiple languages
    - Version control
    
    **Recommended Strategy:**
    - Semantic cache (threshold: 0.30)
    - Cross-encoder reranker
    - Expected hit rate: 60-80%
    
    **Benefits:**
    - Quick access to company knowledge
    - Reduce repetitive questions
    - Support remote workforce
    """)

st.markdown("---")

# Dashboard workflow
st.markdown("## 🔄 Dashboard Workflow")

st.markdown("""
Follow this workflow to find your optimal strategy:

1. **📊 Data Upload**
   - Upload FAQ entries (your cache contents)
   - Upload test queries (with ground truth labels)
   - Validate data format

2. **🔍 Cache Comparison**
   - Compare Exact, Fuzzy, and Semantic strategies
   - Analyze confusion matrices
   - Review performance metrics

3. **⚙️ Threshold Tuning**
   - Experiment with different thresholds
   - Find optimal precision-recall balance
   - Visualize distance distributions

4. **🎯 Reranker Analysis**
   - Add reranking for false positive reduction
   - Compare Simple, Cross-Encoder, and LLM methods
   - Evaluate cost-benefit trade-offs

5. **📈 Get Recommendations**
   - Receive strategy suggestions based on your data
   - Export configuration for production
   - Implement in your application
""")

st.markdown("---")

# Technical concepts
with st.expander("🔬 Advanced: Technical Deep Dive"):
    st.markdown("""
    ### Embedding Models
    
    Common models used for semantic caching:
    - **all-MiniLM-L6-v2** (384 dim) - Fast, good quality, most popular
    - **all-mpnet-base-v2** (768 dim) - Higher quality, slower
    - **paraphrase-MiniLM-L6-v2** (384 dim) - Optimized for paraphrases
    
    ### Distance Metrics
    
    **Cosine Distance:**
    ```python
    cosine_similarity = dot(v1, v2) / (||v1|| * ||v2||)
    cosine_distance = 1 - cosine_similarity
    ```
    
    **Why cosine?**
    - Angle-based (direction matters more than magnitude)
    - Normalized (0 to 1 range)
    - Efficient to compute
    
    ### Confusion Matrix
    
    ```
                    Predicted
                 HIT      MISS
    Actual  HIT   TP       FN
            MISS  FP       TN
    ```
    
    - **TP (True Positive):** Correctly returned cached answer
    - **TN (True Negative):** Correctly identified cache miss
    - **FP (False Positive):** Returned wrong cached answer ⚠️
    - **FN (False Negative):** Missed a valid cache hit
    
    ### Optimization Goals
    
    - **Minimize FP:** False positives give wrong answers (bad!)
    - **Minimize FN:** False negatives waste LLM calls (costly)
    - **Maximize TP:** More cache hits (faster, cheaper)
    - **Threshold tuning:** Balance FP vs FN based on use case
    """)

st.markdown("---")

# Next steps
st.markdown("## 🚀 Ready to Start?")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    ### ✅ You now understand:
    - What semantic caching is
    - How it works (embeddings + distance)
    - Different strategies available
    - Key metrics to track
    - When to use each approach
    """)

with col2:
    st.info("""
    ### 👉 Next Steps:
    1. Prepare your data files
    2. Go to **Data Upload** page
    3. Run the analysis
    4. Find your optimal strategy
    5. Deploy in production!
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Have questions? Check the landing page for quick start guide</p>
</div>
""", unsafe_allow_html=True)
