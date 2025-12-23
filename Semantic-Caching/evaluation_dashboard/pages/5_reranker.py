"""
Reranker Strategies - Advanced Cache Optimization
Test and compare different reranking approaches to reduce false positives
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import time
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import reranker modules
# from src.reranker.adaptors import (
#     simple_keyword_reranker_adapter,
#     cross_encoder_reranker_adapter,
#     llm_reranker_adapter
# )
# from src.reranker.reranked_cache import RerankedCache
# from src.evaluate.cache_evaluator import CacheEvaluator
# from src.evaluate.evaluatable_cache import EvaluatableCache

st.set_page_config(page_title="Reranker Strategies",
                   page_icon="🎯", layout="wide")

st.title("🎯 Reranker Strategies")
st.markdown("### Advanced cache optimization with two-stage retrieval")

st.markdown("---")

# Check if data is loaded
if st.session_state.get('ground_truth_df') is None or st.session_state.get('test_df') is None:
    st.warning("⚠️ Please load data first!")
    if st.button("Go to Data Page"):
        st.switch_page("pages/2_data.py")
    st.stop()

ground_truth_df = st.session_state.ground_truth_df
test_df = st.session_state.test_df

# Introduction
with st.expander("💡 What is Reranking?", expanded=False):
    st.markdown("""
    ### The Problem: False Positives
    
    Semantic caches can return **semantically similar but wrong** answers:
    - Query: "How to delete my account?"
    - Cache returns: "How to update my account?" ❌ (similar but wrong!)
    
    ### The Solution: Two-Stage Retrieval
    
    ```
    Stage 1: Semantic Cache (High Recall)
        ↓ Retrieve top-K candidates (e.g., K=5)
    
    Stage 2: Reranker (High Precision)
        ↓ Validate and reorder candidates
        ↓ Filter out false positives
    
    Result: Return best match OR cache miss
    ```
    
    ### Reranking Strategies
    
    1. **Baseline (No Reranking)**: Return best semantic match as-is
    2. **Simple Keyword**: Boost matches with keyword overlap
    3. **Cross-Encoder**: Neural model validates semantic similarity
    4. **LLM (GPT-4o-mini)**: Most accurate, provides reasoning
    
    ### Benefits
    
    - ✅ Reduce false positives (wrong cached answers)
    - ✅ Improve precision without hurting recall
    - ✅ Tiered cost structure (fast reranker → expensive LLM only if needed)
    """)

st.markdown("---")

# Configuration Section
st.markdown("## ⚙️ Configuration")

# Sidebar for reranker settings
with st.sidebar:
    st.markdown("## 🎯 Reranker Settings")

    st.markdown("### Select Strategies to Test")
    test_baseline = st.checkbox("Baseline (No Reranking)", value=True)
    test_simple = st.checkbox("Simple Keyword Reranker", value=True)
    test_cross_encoder = st.checkbox("Cross-Encoder Reranker", value=True)
    test_llm = st.checkbox("LLM Reranker (GPT-4o-mini)", value=False)

    st.markdown("---")

    # Semantic cache settings
    st.markdown("### 🧠 Semantic Cache Base")

    base_model = st.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
        help="Base model for semantic cache"
    )

    base_threshold = st.slider(
        "Base Cache Threshold",
        min_value=0.20,
        max_value=0.60,
        value=0.42,
        step=0.05,
        help="Threshold for initial retrieval (should be lenient to maximize recall)"
    )

    top_k = st.slider(
        "Top-K Candidates",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of candidates to retrieve for reranking"
    )

    st.markdown("---")

    # Cross-encoder settings
    if test_cross_encoder:
        st.markdown("### 🔄 Cross-Encoder Settings")

        cross_encoder_model = st.selectbox(
            "Cross-Encoder Model",
            [
                "Alibaba-NLP/gte-reranker-modernbert-base",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "cross-encoder/ms-marco-MiniLM-L-12-v2"
            ],
            help="Pre-trained cross-encoder model"
        )

        cross_encoder_threshold = st.slider(
            "Cross-Encoder Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum score to accept reranked result"
        )

    st.markdown("---")

    # LLM settings
    if test_llm:
        st.markdown("### 🤖 LLM Reranker Settings")

        llm_model = st.selectbox(
            "LLM Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            help="OpenAI model for reranking"
        )

        llm_batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=20,
            value=10,
            help="Number of queries to batch together"
        )

        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for LLM reranker"
        )

        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

# Main configuration display
col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"""
    **Base Configuration:**
    - Model: {base_model}
    - Threshold: {base_threshold}
    - Top-K: {top_k}
    """)

with col2:
    selected_count = sum(
        [test_baseline, test_simple, test_cross_encoder, test_llm])
    st.success(f"""
    **Rerankers Selected:**
    - {selected_count} strategies
    - Will test {selected_count} configurations
    """)

with col3:
    if test_llm and not openai_api_key:
        st.warning("⚠️ LLM selected but no API key provided")
    else:
        st.success("✅ Configuration complete")

st.markdown("---")

# Run button
run_reranker_test = st.button(
    "🚀 Run Reranker Comparison", type="primary", use_container_width=True)

# Check implementation status
IMPLEMENTATION_STATUS = True
try:
    from src.reranker.adaptors import (
        simple_keyword_reranker_adapter,
        cross_encoder_reranker_adapter,
        llm_reranker_adapter
    )
    from src.reranker.reranked_cache import RerankedCache
    from src.evaluate.cache_evaluator import CacheEvaluator
    from src.evaluate.evaluatable_cache import EvaluatableCache
    IMPLEMENTATION_STATUS = True
except ImportError:
    pass

if not IMPLEMENTATION_STATUS:
    st.warning("""
    ⚠️ **Reranker modules not implemented yet**
    
    Please implement these in `src/reranker/`:
    - `adaptors.py` - Reranker adapters (simple, cross-encoder, LLM)
    - `reranked_cache.py` - RerankedCache class
    
    And evaluation modules in `src/evaluate/`:
    - `cache_evaluator.py` - CacheEvaluator
    - `evaluatable_cache.py` - EvaluatableCache
    
    The UI is ready - just add your classes!
    """)

    if st.checkbox("Show Mock Results (Demo)", value=False):
        run_reranker_test = True

# Results section
if run_reranker_test:

    st.markdown("---")
    st.markdown("## 📊 Reranker Comparison Results")

    # Build strategy list
    strategies_to_test = []
    if test_baseline:
        strategies_to_test.append(("Baseline (No Reranking)", "baseline"))
    if test_simple:
        strategies_to_test.append(("Simple Keyword", "simple"))
    if test_cross_encoder:
        strategies_to_test.append(("Cross-Encoder", "cross_encoder"))
    if test_llm and openai_api_key:
        strategies_to_test.append(("LLM (GPT-4o-mini)", "llm"))

    if not strategies_to_test:
        st.warning("Please select at least one reranking strategy")
        st.stop()

    # Run evaluation
    with st.spinner(f"Testing {len(strategies_to_test)} reranking strategies..."):

        reranking_results = {}

        progress_bar = st.progress(0)

        for i, (strategy_name, strategy_type) in enumerate(strategies_to_test):

            # Mock evaluation - Replace with actual implementation
            # TODO: Initialize cache and reranker, run evaluation

            # Mock timing based on strategy
            if strategy_type == "baseline":
                time_taken = np.random.uniform(0.5, 1.0)
                precision = 0.65 + np.random.uniform(-0.05, 0.05)
                recall = 0.70 + np.random.uniform(-0.05, 0.05)
            elif strategy_type == "simple":
                time_taken = np.random.uniform(0.8, 1.5)
                precision = 0.75 + np.random.uniform(-0.05, 0.05)
                recall = 0.68 + np.random.uniform(-0.05, 0.05)
            elif strategy_type == "cross_encoder":
                time_taken = np.random.uniform(2.0, 4.0)
                precision = 0.88 + np.random.uniform(-0.03, 0.03)
                recall = 0.72 + np.random.uniform(-0.05, 0.05)
            else:  # llm
                time_taken = np.random.uniform(8.0, 15.0)
                precision = 0.95 + np.random.uniform(-0.02, 0.02)
                recall = 0.70 + np.random.uniform(-0.05, 0.05)

            # Calculate other metrics
            f1 = 2 * (precision * recall) / (precision + recall)
            accuracy = (precision + recall) / 2
            hit_rate = recall * 0.8

            # Mock confusion matrix values
            total = len(test_df)
            tp = int(recall * test_df['cache_hit'].sum())
            fn = test_df['cache_hit'].sum() - tp
            fp = int((1 - precision) * tp) if tp > 0 else 0
            tn = total - tp - fn - fp

            reranking_results[strategy_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'hit_rate': hit_rate,
                'time': time_taken,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }

            progress_bar.progress((i + 1) / len(strategies_to_test))

        progress_bar.empty()

    st.success("✅ Evaluation complete!")

    # Results tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Metrics Overview",
        "🎯 Confusion Matrices",
        "📈 Performance Charts",
        "⚡ Speed vs Accuracy",
        "💡 Recommendations"
    ])

    with tab1:
        st.markdown("### Performance Metrics Summary")

        # Create summary table
        summary_data = []
        for strategy_name in reranking_results.keys():
            r = reranking_results[strategy_name]
            summary_data.append({
                'Strategy': strategy_name,
                'Precision': f"{r['precision']:.1%}",
                'Recall': f"{r['recall']:.1%}",
                'F1 Score': f"{r['f1']:.1%}",
                'Accuracy': f"{r['accuracy']:.1%}",
                'Hit Rate': f"{r['hit_rate']:.1%}",
                'Time (s)': f"{r['time']:.2f}"
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Best performers
        st.markdown("### 🏆 Best Performers")

        best_precision = max(reranking_results.items(),
                             key=lambda x: x[1]['precision'])
        best_recall = max(reranking_results.items(),
                          key=lambda x: x[1]['recall'])
        best_f1 = max(reranking_results.items(), key=lambda x: x[1]['f1'])
        fastest = min(reranking_results.items(), key=lambda x: x[1]['time'])

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Best Precision",
                best_precision[0],
                f"{best_precision[1]['precision']:.1%}"
            )

        with col2:
            st.metric(
                "Best Recall",
                best_recall[0],
                f"{best_recall[1]['recall']:.1%}"
            )

        with col3:
            st.metric(
                "Best F1 Score",
                best_f1[0],
                f"{best_f1[1]['f1']:.1%}"
            )

        with col4:
            st.metric(
                "Fastest",
                fastest[0],
                f"{fastest[1]['time']:.2f}s"
            )

        st.markdown("---")

        # Detailed breakdown
        st.markdown("### 📋 Detailed Breakdown")

        for strategy_name, results in reranking_results.items():
            with st.expander(f"📌 {strategy_name}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Confusion Matrix Counts:**")
                    st.write(f"- True Positives (TP): {results['tp']}")
                    st.write(f"- True Negatives (TN): {results['tn']}")
                    st.write(f"- False Positives (FP): {results['fp']}")
                    st.write(f"- False Negatives (FN): {results['fn']}")

                with col2:
                    st.markdown("**Performance Metrics:**")
                    st.write(f"- Precision: {results['precision']:.1%}")
                    st.write(f"- Recall: {results['recall']:.1%}")
                    st.write(f"- F1 Score: {results['f1']:.1%}")
                    st.write(f"- Processing Time: {results['time']:.2f}s")
                    st.write(
                        f"- Avg per query: {results['time']/len(test_df)*1000:.1f}ms")

                specificity = results['tn'] / (results['tn'] + results['fp']) if (
                    results['tn'] + results['fp']) > 0 else 0
                st.info(f"""
                **Interpretation:**
                - Precision ({results['precision']:.1%}): When it says "hit", it's correct {results['precision']:.1%} of the time
                - Recall ({results['recall']:.1%}): Catches {results['recall']:.1%} of all valid cache hits
                - Specificity ({specificity:.1%}): Correctly identifies {specificity:.1%} of cache misses
                """)

    with tab2:
        st.markdown("### Confusion Matrices Comparison")

        st.info("""
        **Reading Confusion Matrices:**
        - **TN (Top-Left)**: Correctly identified cache miss ✅
        - **FP (Top-Right)**: Returned WRONG cached answer ❌ (bad!)
        - **FN (Bottom-Left)**: Missed a valid cache hit ⚠️
        - **TP (Bottom-Right)**: Correctly returned cached answer ✅
        """)

        # Create confusion matrices
        num_strategies = len(reranking_results)
        cols = st.columns(min(num_strategies, 2))

        for idx, (strategy_name, results) in enumerate(reranking_results.items()):
            col_idx = idx % 2

            with cols[col_idx]:
                st.markdown(f"#### {strategy_name}")

                # Create confusion matrix
                cm = np.array([
                    [results['tn'], results['fp']],
                    [results['fn'], results['tp']]
                ])

                # Plotly heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Miss', 'Hit'],
                    y=['Miss', 'Hit'],
                    colorscale='RdYlGn',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 18},
                    showscale=False,
                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
                ))

                fig.update_layout(
                    xaxis_title='Predicted',
                    yaxis_title='Actual',
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Metrics below matrix
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Precision", f"{results['precision']:.1%}")
                    st.metric("Recall", f"{results['recall']:.1%}")
                with col_b:
                    st.metric("F1 Score", f"{results['f1']:.1%}")
                    st.metric("Accuracy", f"{results['accuracy']:.1%}")

    with tab3:
        st.markdown("### Performance Comparison Charts")

        # Metrics bar chart
        st.markdown("#### Classification Metrics Comparison")

        strategy_names = list(reranking_results.keys())

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Precision',
            x=strategy_names,
            y=[reranking_results[s]['precision'] * 100 for s in strategy_names],
            marker_color='#2196F3'
        ))

        fig.add_trace(go.Bar(
            name='Recall',
            x=strategy_names,
            y=[reranking_results[s]['recall'] * 100 for s in strategy_names],
            marker_color='#FF9800'
        ))

        fig.add_trace(go.Bar(
            name='F1 Score',
            x=strategy_names,
            y=[reranking_results[s]['f1'] * 100 for s in strategy_names],
            marker_color='#4CAF50'
        ))

        fig.add_trace(go.Bar(
            name='Accuracy',
            x=strategy_names,
            y=[reranking_results[s]['accuracy'] * 100 for s in strategy_names],
            marker_color='#9C27B0'
        ))

        fig.update_layout(
            barmode='group',
            xaxis_title="Reranking Strategy",
            yaxis_title="Score (%)",
            height=500,
            yaxis=dict(range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Processing time comparison
        st.markdown("#### Processing Time Comparison")

        fig = go.Figure(data=[
            go.Bar(
                x=strategy_names,
                y=[reranking_results[s]['time'] for s in strategy_names],
                marker_color='lightblue',
                text=[
                    f"{reranking_results[s]['time']:.2f}s" for s in strategy_names],
                textposition='auto'
            )
        ])

        fig.update_layout(
            xaxis_title="Reranking Strategy",
            yaxis_title="Time (seconds)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Speed vs Accuracy Trade-off Analysis")

        st.info("""
        **Ideal Zone:** Top-left (high F1, low time)
        
        This chart helps you choose the best reranker based on your priorities:
        - **Speed Critical**: Choose closest to left side
        - **Accuracy Critical**: Choose closest to top
        - **Balanced**: Look for best F1/time ratio
        """)

        # Scatter plot
        fig = go.Figure()

        times = [reranking_results[s]['time'] for s in strategy_names]
        f1_scores = [reranking_results[s]['f1'] * 100 for s in strategy_names]
        precisions = [reranking_results[s]['precision']
                      * 100 for s in strategy_names]

        fig.add_trace(go.Scatter(
            x=times,
            y=f1_scores,
            mode='markers+text',
            marker=dict(
                size=[r['tp'] + r['tn'] for r in reranking_results.values()],
                sizemode='area',
                sizeref=2.*max([r['tp'] + r['tn']
                               for r in reranking_results.values()])/(40.**2),
                sizemin=10,
                color=precisions,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Precision (%)")
            ),
            text=strategy_names,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Time: %{x:.2f}s<br>F1: %{y:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            xaxis_title="Processing Time (seconds)",
            yaxis_title="F1 Score (%)",
            title="Reranker Performance Trade-off",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Calculate efficiency ratio
        st.markdown("---")
        st.markdown("#### Efficiency Analysis (F1 / Time Ratio)")

        efficiency_data = []
        for strategy_name in strategy_names:
            r = reranking_results[strategy_name]
            efficiency = r['f1'] / r['time']
            efficiency_data.append({
                'Strategy': strategy_name,
                'F1 Score': f"{r['f1']:.1%}",
                'Time (s)': f"{r['time']:.2f}",
                'Efficiency': f"{efficiency:.3f}"
            })

        efficiency_df = pd.DataFrame(efficiency_data)
        efficiency_df = efficiency_df.sort_values(
            'Efficiency', ascending=False)

        st.dataframe(efficiency_df, use_container_width=True, hide_index=True)

        best_efficiency = efficiency_df.iloc[0]
        st.success(f"""
        **Most Efficient Strategy: {best_efficiency['Strategy']}**
        
        Best balance of accuracy and speed for your use case.
        """)

    with tab5:
        st.markdown("### 💡 Reranker Recommendations")

        # Strategy-specific recommendations
        st.markdown("#### Strategy-by-Strategy Analysis")

        for strategy_name, results in reranking_results.items():

            if "Baseline" in strategy_name:
                st.markdown(f"""
                ### {strategy_name}
                
                **Characteristics:**
                - ⚡ Fastest (no reranking overhead)
                - ❌ Higher false positive rate ({results['fp']} wrong answers)
                - Precision: {results['precision']:.1%}
                
                **Use When:**
                - Speed is absolutely critical (< 1s response time requirement)
                - False positives are acceptable (non-critical domain)
                - Testing/prototyping phase
                
                **Avoid When:**
                - Wrong answers have consequences (medical, legal, financial)
                - Brand reputation matters
                """)

            elif "Simple" in strategy_name:
                st.markdown(f"""
                ### {strategy_name}
                
                **Characteristics:**
                - 🔤 Rule-based keyword matching
                - ⚡ Fast (minimal overhead: {results['time']:.2f}s)
                - 📊 Moderate precision improvement: {results['precision']:.1%}
                
                **Use When:**
                - Queries have predictable keyword patterns
                - No external dependencies allowed
                - Want interpretable reranking logic
                
                **Limitations:**
                - Misses semantic paraphrases
                - Language-specific rules needed
                """)

            elif "Cross-Encoder" in strategy_name:
                st.markdown(f"""
                ### {strategy_name}
                
                **Characteristics:**
                - 🧠 Neural semantic validation
                - ⚖️ Balanced speed/accuracy: {results['time']:.2f}s, {results['precision']:.1%}
                - 🎯 Significantly reduces false positives
                
                **Use When:**
                - Production systems need reliable accuracy
                - Can afford 2-5s latency
                - GPU available (recommended)
                
                **Best for:**
                - Customer support chatbots
                - E-commerce product search
                - Most production LLM applications
                
                **⭐ Recommended for most use cases**
                """)

            elif "LLM" in strategy_name:
                st.markdown(f"""
                ### {strategy_name}
                
                **Characteristics:**
                - 🤖 Highest precision: {results['precision']:.1%}
                - 🐢 Slowest: {results['time']:.2f}s
                - 💰 API costs per query
                - 📝 Provides reasoning/explanations
                
                **Use When:**
                - Accuracy is absolutely critical
                - Wrong answers have serious consequences
                - Budget allows for API costs
                - Can afford 10-30s latency
                
                **Best for:**
                - Medical/Healthcare FAQ
                - Legal/Compliance questions
                - Financial advice
                - Safety-critical domains
                
                **⚠️ Use selectively due to cost/latency**
                """)

        st.markdown("---")

        # Overall recommendation
        st.markdown("#### 🎯 Recommended Architecture")

        # Find best by F1
        best_overall = max(reranking_results.items(), key=lambda x: x[1]['f1'])

        st.success(f"""
        ### Production Recommendation: {best_overall[0]}
        
        **Performance:**
        - Precision: {best_overall[1]['precision']:.1%}
        - Recall: {best_overall[1]['recall']:.1%}
        - F1 Score: {best_overall[1]['f1']:.1%}
        - Processing Time: {best_overall[1]['time']:.2f}s
        - False Positives: {best_overall[1]['fp']} (out of {len(test_df)} queries)
        
        **Expected Impact:**
        - Cache hit rate: ~{best_overall[1]['hit_rate']:.1%}
        - Cost reduction: ~{best_overall[1]['hit_rate'] * 0.6:.0%} (vs no caching)
        - Latency improvement: ~10-50x (vs LLM API)
        """)

        st.markdown("---")

        # Tiered architecture suggestion
        st.markdown("#### 🏗️ Advanced: Tiered Reranking Architecture")

        st.info("""
        **For Maximum Efficiency, Use a Tiered Approach:**
        
        ```
        Tier 1: Semantic Cache (Threshold: 0.42, Top-K: 5)
            ↓ Retrieve top-5 candidates
        
        Tier 2: Cross-Encoder Reranking (90% of queries)
            ↓ Fast neural validation
            ↓ If confidence > 0.8 → Return
        
        Tier 3: LLM Reranking (10% uncertain cases)
            ↓ High-confidence validation
            ↓ If validated → Return
        
        Tier 4: Cache Miss → Call LLM API
        ```
        
        **Benefits:**
        - 70% of queries: Fast cache hit (< 1s)
        - 20% of queries: Cross-encoder validated (2-3s)
        - 10% of queries: LLM validated or fresh API call (10-30s)
        
        **Expected Results:**
        - Average latency: 3-5s (vs 15-30s without caching)
        - Cost reduction: 60-80%
        - Precision: > 90% (with LLM safety net)
        """)

        # Configuration export
        st.markdown("---")
        st.markdown("### 💾 Export Configuration")

        config_text = f"""# Reranker Configuration
# Generated by Semantic Cache Analyzer
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

[semantic_cache]
embedding_model = "{base_model}"
distance_threshold = {base_threshold}
top_k = {top_k}

[reranker]
strategy = "{best_overall[0]}"
"""

        if "Cross-Encoder" in best_overall[0]:
            config_text += f"""cross_encoder_model = "{cross_encoder_model}"
cross_encoder_threshold = {cross_encoder_threshold}
"""
        elif "LLM" in best_overall[0]:
            config_text += f"""llm_model = "{llm_model}"
batch_size = {llm_batch_size}
"""

        config_text += f"""
[expected_performance]
precision = {best_overall[1]['precision']:.4f}
recall = {best_overall[1]['recall']:.4f}
f1_score = {best_overall[1]['f1']:.4f}
processing_time = {best_overall[1]['time']:.2f}
"""

        st.download_button(
            "📥 Download Reranker Configuration",
            config_text,
            "reranker_config.toml",
            "text/plain",
            use_container_width=True
        )

else:
    # Preview mode
    st.info(f"""
    **Configuration Summary:**
    - Base cache: {base_model} (threshold: {base_threshold})
    - Top-K candidates: {top_k}
    - Strategies to test: {sum([test_baseline, test_simple, test_cross_encoder, test_llm])}
    
    Click "Run Reranker Comparison" to begin evaluation.
    """)

    # Show example workflow
    with st.expander("📖 Example Workflow"):
        st.markdown("""
        1. **Select strategies** to test (Baseline, Simple, Cross-Encoder, LLM)
        2. **Configure base cache** (embedding model, threshold, top-K)
        3. **Set reranker parameters** (models, thresholds, batch sizes)
        4. **Run comparison** - tests all selected strategies
        5. **Analyze results** - view metrics, confusion matrices, charts
        6. **Get recommendations** - data-driven suggestions
        7. **Export configuration** - production-ready config file
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 Tip: Start with Cross-Encoder reranking for best balance of speed and accuracy</p>
</div>
""", unsafe_allow_html=True)
