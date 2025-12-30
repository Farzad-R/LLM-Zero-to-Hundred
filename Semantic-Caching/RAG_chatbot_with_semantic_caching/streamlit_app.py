"""
TaskFlow RAG Chatbot with Semantic Caching - Streamlit UI

This interactive demo shows how semantic caching improves response times
and allows users to contribute to the cache through feedback.
"""

import streamlit as st
import os
from datetime import datetime
import traceback
from cached_rag_chatbot_chroma import CachedRAGChatbot
from dotenv import load_dotenv
from pyprojroot import here


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(
    page_title="TaskFlow Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark mode support
st.markdown("""
<style>
    /* Cache Hit Box - Green theme */
    .cache-hit {
        background-color: rgba(40, 167, 69, 0.15);
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        color: inherit; /* Inherit text color from Streamlit theme */
    }
    
    .cache-hit h4 {
        color: #28a745 !important;
        margin-top: 0;
        font-size: 1.1em;
    }
    
    .cache-hit p {
        margin: 5px 0;
        color: inherit;
    }
    
    .cache-hit strong {
        color: inherit;
        font-weight: 600;
    }
    
    /* Cache Miss Box - Yellow/Orange theme */
    .cache-miss {
        background-color: rgba(255, 193, 7, 0.15);
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        color: inherit;
    }
    
    .cache-miss h4 {
        color: #ff9800 !important;
        margin-top: 0;
        font-size: 1.1em;
    }
    
    .cache-miss p {
        margin: 5px 0;
        color: inherit;
    }
    
    /* Dark mode specific adjustments */
    @media (prefers-color-scheme: dark) {
        .cache-hit {
            background-color: rgba(40, 167, 69, 0.2);
            border-left-color: #4caf50;
        }
        
        .cache-hit h4 {
            color: #4caf50 !important;
        }
        
        .cache-miss {
            background-color: rgba(255, 193, 7, 0.2);
            border-left-color: #ffb300;
        }
        
        .cache-miss h4 {
            color: #ffb300 !important;
        }
    }
    
    /* For Streamlit's dark theme detection */
    [data-theme="dark"] .cache-hit {
        background-color: rgba(40, 167, 69, 0.2);
        border-left-color: #4caf50;
    }
    
    [data-theme="dark"] .cache-hit h4 {
        color: #4caf50 !important;
    }
    
    [data-theme="dark"] .cache-miss {
        background-color: rgba(255, 193, 7, 0.2);
        border-left-color: #ffb300;
    }
    
    [data-theme="dark"] .cache-miss h4 {
        color: #ffb300 !important;
    }
    
    /* Stats and feedback boxes */
    .stats-box {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: inherit;
    }
    
    .feedback-section {
        background-color: rgba(33, 150, 243, 0.1);
        border: 1px solid rgba(33, 150, 243, 0.3);
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
    st.session_state.initialized = False

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'last_response' not in st.session_state:
    st.session_state.last_response = None

if 'cache_stats' not in st.session_state:
    st.session_state.cache_stats = {
        'total_queries': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'approved_additions': 0
    }


def initialize_chatbot():
    """Initialize the chatbot with ChromaDB vectorstore and cache."""
    try:
        with st.spinner("🚀 Initializing TaskFlow Assistant..."):
            # Initialize chatbot
            chatbot = CachedRAGChatbot(
                cache_distance_threshold=0.1,
                model_name="gpt-4o-mini",  # Using mini for cost efficiency
                temperature=0,
                chroma_persist_dir=str(here("data/chroma_db")),
                chroma_collection="taskflow_docs"
            )

            progress_bar = st.progress(0)

            # Load existing ChromaDB vectorstore
            st.write("📄 Loading ChromaDB vectorstore...")
            if chatbot.load_existing_vectorstore():
                stats = chatbot.get_vectorstore_stats()
                st.write(f"   ✓ Loaded collection: {stats['collection_name']}")
                st.write(f"   ✓ Documents: {stats['document_count']}")
            else:
                st.error("❌ ChromaDB vectorstore not found!")
                st.info("Please run: python prepare_data_chroma.py")
                return None

            progress_bar.progress(75)

            # Load initial cache seed
            st.write("💾 Loading semantic cache...")
            chatbot.load_cache_from_file(here("data/taskflow_cache_seed.csv"))
            progress_bar.progress(100)

            st.session_state.chatbot = chatbot
            st.session_state.initialized = True
            st.success("✅ TaskFlow Assistant is ready!")

    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        st.info("Make sure OPENAI_API_KEY is set and ChromaDB is prepared")
        st.code(traceback.format_exc())


def save_to_cache(question: str, answer: str):
    """Save a Q&A pair to the cache after user approval."""
    st.session_state.chatbot.add_to_cache(question, answer)
    st.session_state.chatbot.save_cache_to_file("taskflow_cache_approved.csv")
    st.session_state.cache_stats['approved_additions'] += 1


def format_response_box(result: dict, question: str):
    """Format the response with appropriate styling."""
    if result['cache_hit']:
        cache_info = result['cache_info']
        return f"""
        <div class="cache-hit">
            <h4>✨ Answer from Cache (Instant Response!)</h4>
            <p><strong>Matched Question:</strong> {cache_info['matched_question']}</p>
            <p><strong>Similarity Score:</strong> {cache_info['similarity']:.2%}</p>
            <p><strong>Distance:</strong> {cache_info['distance']:.4f}</p>
        </div>
        """
    else:
        return f"""
        <div class="cache-miss">
            <h4>🔍 Answer from RAG Pipeline (Full Search)</h4>
            <p>This answer was generated by searching the documentation and will be cached for future similar questions.</p>
        </div>
        """


# Sidebar
with st.sidebar:
    st.title("🤖 TaskFlow Assistant")
    st.markdown("---")

    # API Key check
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not found!")
        st.info("Please set your OpenAI API key as an environment variable.")
        api_key = st.text_input("Or enter it here:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.success("✅ API Key loaded")

    st.markdown("---")

    # Initialize button
    if not st.session_state.initialized:
        if st.button("🚀 Initialize Assistant", use_container_width=True):
            initialize_chatbot()
    else:
        st.success("✅ Assistant Ready")

        # Stats
        st.markdown("### 📊 Session Statistics")
        stats = st.session_state.cache_stats

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", stats['total_queries'])
            st.metric("Cache Hits", stats['cache_hits'])
        with col2:
            hit_rate = (stats['cache_hits'] / stats['total_queries']
                        * 100) if stats['total_queries'] > 0 else 0
            st.metric("Hit Rate", f"{hit_rate:.1f}%")
            st.metric("Approved", stats['approved_additions'])

        st.markdown("---")

        # Cache management
        st.markdown("### 💾 Cache Management")

        if st.session_state.chatbot:
            cache_size = len(st.session_state.chatbot.cache.get_all_pairs())
            st.info(f"Current cache size: **{cache_size}** Q&A pairs")

            if st.button("📥 Download Cache", use_container_width=True):
                st.session_state.chatbot.save_cache_to_file(
                    "taskflow_cache_export.csv")
                st.success("Cache exported to taskflow_cache_export.csv")

            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.conversation_history = []
                st.rerun()

        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Settings")

        threshold = st.slider(
            "Cache Similarity Threshold",
            min_value=0.0,
            max_value=0.6,
            value=0.1,
            step=0.05,
            help="Lower = stricter matching, Higher = looser matching"
        )

        if st.session_state.chatbot:
            st.session_state.chatbot.cache.distance_threshold = threshold

    st.markdown("---")
    st.markdown("### 💡 About")
    st.info("""
    This demo shows semantic caching in action:
    - ✨ Cache hits return instantly
    - 🔍 Cache misses use full RAG
    - 👍 Your feedback improves the cache
    """)


# Main content
st.title("💬 TaskFlow Assistant")
st.markdown(
    "Ask me anything about TaskFlow! I'll check my cache first, then search the documentation if needed.")

if not st.session_state.initialized:
    st.warning("👈 Please initialize the assistant using the sidebar")

    # Show example questions
    st.markdown("### 🎯 Example Questions to Try:")
    examples = [
        "How do I create a new project?",
        "What are the pricing plans?",
        "Can I integrate with Slack?",
        "How do I set up automations?",
        "What's the difference between workspaces and projects?",
        "How do I track time on tasks?",
    ]

    cols = st.columns(2)
    for idx, example in enumerate(examples):
        with cols[idx % 2]:
            st.markdown(f"- {example}")

else:
    # Chat interface
    st.markdown("---")

    # Display conversation history
    for idx, entry in enumerate(st.session_state.conversation_history):
        # User message
        with st.chat_message("user"):
            st.markdown(entry['question'])

        # Assistant message
        with st.chat_message("assistant"):
            st.markdown(entry['answer'])

            # Show cache status
            st.markdown(entry['cache_status'], unsafe_allow_html=True)

            # Show timestamp
            st.caption(f"⏱️ Response time: {entry['response_time']}")

    # Query input
    query = st.chat_input("Ask a question about TaskFlow...")

    if query:
        # Add user message to chat
        with st.chat_message("user"):
            st.markdown(query)

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                start_time = datetime.now()

                try:
                    result = st.session_state.chatbot.query(
                        query, verbose=False)

                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds()

                    # Update stats
                    st.session_state.cache_stats['total_queries'] += 1
                    if result['cache_hit']:
                        st.session_state.cache_stats['cache_hits'] += 1
                    else:
                        st.session_state.cache_stats['cache_misses'] += 1

                    # Display answer
                    st.markdown(result['answer'])

                    # Display cache status
                    cache_status = format_response_box(result, query)
                    st.markdown(cache_status, unsafe_allow_html=True)

                    # Show response time
                    st.caption(f"⏱️ Response time: {response_time:.2f}s")

                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'question': query,
                        'answer': result['answer'],
                        'cache_hit': result['cache_hit'],
                        'cache_status': cache_status,
                        'response_time': f"{response_time:.2f}s",
                        'cache_info': result.get('cache_info')
                    })

                    # Store for feedback
                    st.session_state.last_response = {
                        'question': query,
                        'answer': result['answer'],
                        'cache_hit': result['cache_hit']
                    }

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)

        st.rerun()

    # Feedback section for last response (only if cache miss)
    if st.session_state.last_response and not st.session_state.last_response['cache_hit']:
        st.markdown("---")
        st.markdown("### 💭 Was this answer helpful?")

        col1, col2, col3 = st.columns([1, 1, 3])

        with col1:
            if st.button("👍 Yes, add to cache", key="helpful"):
                save_to_cache(
                    st.session_state.last_response['question'],
                    st.session_state.last_response['answer']
                )
                st.success(
                    "✅ Added to cache! Similar questions will now be answered instantly.")
                st.session_state.last_response = None
                st.rerun()

        with col2:
            if st.button("👎 No, don't cache", key="not_helpful"):
                st.info("👌 Noted. This answer won't be cached.")
                st.session_state.last_response = None
                st.rerun()

        with col3:
            st.caption(
                "Your feedback helps improve response quality for everyone!")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 Powered by Semantic Caching + RAG | Built with LangGraph & OpenAI</p>
    <p style='font-size: 0.9em;'>This is a demo showcasing semantic caching benefits in RAG systems</p>
</div>
""", unsafe_allow_html=True)
