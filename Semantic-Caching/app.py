"""
Semantic Cache Analyzer - Landing Page
"""
import streamlit as st


# Page config
st.set_page_config(
    page_title="Semantic Cache Analyzer",
    # page_icon="🚀",
    layout="wide"
)

# Header
st.title("🔷 Semantic Cache Analyzer")
st.markdown("### Find the optimal caching strategy for your LLM application")

st.markdown("---")

# Introduction
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ## What is this tool?
    
    This dashboard helps you:
    - 📊 **Analyze** different caching strategies
    - ⚙️ **Optimize** threshold settings
    - 🎯 **Compare** reranking methods
    - 💡 **Get recommendations** for your use case
    
    ### Why semantic caching?
    
    - 💰 **Reduce costs**
    - ⚡ **Improve speed**
    - 🎯 **Maintain consistency** in responses
    """)

with col2:
    st.markdown("""
    ## How it works
    
    1. **Upload your data** - FAQ entries and test queries
    2. **Run analysis** - Compare different strategies
    3. **Tune parameters** - Find optimal settings
    4. **Deploy** - Use the best strategy in production
    
    ### Quick Start
    
    👉 Check out the **About Semantic Caching** page to learn more
    
    📊 Then head to **Data Upload** to get started
    """)

st.markdown("---")

# Feature cards
st.markdown("## 🎯 Features")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### 📊 Data Analysis
    Upload and validate your FAQ and test datasets
    """)

with col2:
    st.markdown("""
    ### 🔍 Strategy Comparison
    Compare Exact, Fuzzy, and Semantic caching
    """)

with col3:
    st.markdown("""
    ### ⚙️ Optimization
    Fine-tune thresholds and parameters
    """)

with col4:
    st.markdown("""
    ### 🎯 Reranker
    Imporove cache hits with reranking methods
    """)

# st.markdown("---")

# # Getting started section
# st.markdown("## 🚀 Getting Started")

# with st.expander("📖 New to semantic caching?"):
#     st.markdown("""
#     **Start here:** Go to the **About Semantic Caching** page in the sidebar to learn:
#     - What semantic caching is
#     - How it differs from traditional caching
#     - When to use different strategies
#     - Key concepts and terminology
#     """)

# with st.expander("💡 Ready to analyze your data?"):
#     st.markdown("""
#     **Next steps:**
#     1. Prepare your FAQ data (CSV with id, question, answer columns)
#     2. Prepare test queries (CSV with question, answer, src_question_id, cache_hit columns)
#     3. Go to the **Data Upload** page
#     4. Follow the analysis workflow
#     """)

# st.markdown("---")

# # Footer
# st.markdown("""
# <div style='text-align: center; color: gray; padding: 20px;'>
#     <p>Built with Streamlit | Version 1.0</p>
# </div>
# """, unsafe_allow_html=True)
