"""
Threshold Optimization & Model Comparison
Find the optimal configuration for production deployment
"""
from sentence_transformers import SentenceTransformer
from cachelab.evaluate.evaluation_result import EvaluationResult
from cachelab.evaluate.evaluatable_cache import EvaluatableCache
from cachelab.evaluate.cache_evaluator import CacheEvaluator
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Optimization", page_icon="⚙️", layout="wide")

st.title("⚙️ Threshold Optimization & Model Comparison")
st.markdown("### Find the optimal configuration for production deployment")

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
st.info("""
**Goal:** Find the optimal threshold and embedding model for your production semantic cache.

This page helps you:
- 🎯 Systematically test multiple thresholds
- 📊 Compare embedding models (sentence-transformers + OpenAI)
- 📈 Visualize precision-recall trade-offs
- 💡 Get data-driven recommendations for deployment
""")

st.markdown("---")

# Configuration Section
st.markdown("## 🔧 Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📐 Threshold Range")

    threshold_min = st.number_input(
        "Minimum Threshold",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05,
        help="Start of threshold range to test"
    )

    threshold_max = st.number_input(
        "Maximum Threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="End of threshold range to test"
    )

    threshold_step = st.number_input(
        "Step Size",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Increment between threshold values"
    )

    # Generate threshold list
    thresholds = list(
        np.arange(threshold_min, threshold_max + threshold_step, threshold_step))
    st.info(
        f"Will test **{len(thresholds)} thresholds**: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")

with col2:
    st.markdown("### 🤖 Models to Compare")

    st.markdown("**Sentence Transformers (Open Source):**")
    test_minilm = st.checkbox("all-MiniLM-L6-v2 (Fast, 384d)", value=True)
    test_mpnet = st.checkbox("all-mpnet-base-v2 (Balanced, 768d)", value=True)
    test_paraphrase = st.checkbox(
        "paraphrase-MiniLM-L6-v2 (Paraphrase-optimized, 384d)", value=False)
    test_multilingual = st.checkbox(
        "paraphrase-multilingual-MiniLM-L12-v2 (Multilingual, 384d)", value=False)

    st.markdown("**OpenAI Embeddings (Requires API Key):**")
    test_ada = st.checkbox("text-embedding-ada-002 (1536d)", value=False)
    test_small = st.checkbox("text-embedding-3-small (512-1536d)", value=False)
    test_large = st.checkbox("text-embedding-3-large (256-3072d)", value=False)

    # Check if OpenAI models selected but no API key
    openai_models_selected = any([test_ada, test_small, test_large])
    if openai_models_selected:
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for OpenAI embedding models"
        )
        if not openai_api_key:
            st.warning("⚠️ OpenAI API key required for selected models")

# Build model list
selected_models = []
if test_minilm:
    selected_models.append(
        ("all-MiniLM-L6-v2", "MiniLM-L6 (Fast)", "sentence-transformers"))
if test_mpnet:
    selected_models.append(
        ("all-mpnet-base-v2", "MPNet (Balanced)", "sentence-transformers"))
if test_paraphrase:
    selected_models.append(
        ("paraphrase-MiniLM-L6-v2", "Paraphrase-MiniLM", "sentence-transformers"))
if test_multilingual:
    selected_models.append(("paraphrase-multilingual-MiniLM-L12-v2",
                           "Multilingual-MiniLM", "sentence-transformers"))
if test_ada and openai_models_selected and openai_api_key:
    selected_models.append(("text-embedding-ada-002", "Ada-002", "openai"))
if test_small and openai_models_selected and openai_api_key:
    selected_models.append(
        ("text-embedding-3-small", "Embedding-3-Small", "openai"))
if test_large and openai_models_selected and openai_api_key:
    selected_models.append(
        ("text-embedding-3-large", "Embedding-3-Large", "openai"))

st.info(f"**{len(selected_models)} models** selected for comparison")

st.markdown("---")

# Run optimization
run_optimization = st.button(
    "🚀 Run Optimization", type="primary", width='stretch')

# All modules are implemented
if run_optimization:

    st.markdown("---")
    st.markdown("## 📊 Optimization Results")

    # Initialize results storage
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = {}

    # Tab structure for results
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Threshold Analysis",
        "🤖 Model Comparison",
        "🎯 Optimal Configuration",
        "📋 Detailed Metrics"
    ])

    with tab1:
        st.markdown("### Threshold Sweep Results")

        # Select model for threshold analysis
        if len(selected_models) > 0:
            model_for_threshold = st.selectbox(
                "Select model for threshold analysis",
                range(len(selected_models)),
                format_func=lambda i: selected_models[i][1]
            )

            selected_model_name, selected_model_display, selected_model_type = selected_models[
                model_for_threshold]

            with st.spinner(f"Initializing {selected_model_display}..."):
                # Load embedding model
                encoder = SentenceTransformer(selected_model_name)

                # Create evaluatable cache
                # Initial threshold, will override
                cache = EvaluatableCache(encoder, distance_threshold=0.3)
                qa_pairs = list(
                    zip(ground_truth_df["question"], ground_truth_df["answer"]))
                cache.add_many(qa_pairs)

                # Create evaluator
                evaluator = CacheEvaluator(ground_truth_df, test_df)

            with st.spinner(f"Testing {len(thresholds)} thresholds on {selected_model_display}..."):

                # Real threshold evaluation
                threshold_results = []

                progress_bar = st.progress(0)
                for i, threshold in enumerate(thresholds):
                    # Real evaluation with threshold override
                    def check_fn(q, t=threshold): return cache.check(
                        q, threshold_override=t)
                    result = evaluator.evaluate(check_fn)

                    threshold_results.append({
                        'threshold': threshold,
                        'precision': result.precision,
                        'recall': result.recall,
                        'f1': result.f1_score,
                        'accuracy': result.accuracy,
                        'hit_rate': result.hit_rate
                    })

                    progress_bar.progress((i + 1) / len(thresholds))

                progress_bar.empty()

                # Store results
                st.session_state.optimization_results[selected_model_display] = threshold_results

                # Find optimal thresholds
                best_by_f1 = max(threshold_results, key=lambda x: x['f1'])
                best_by_precision = max(
                    threshold_results, key=lambda x: x['precision'])
                best_by_recall = max(
                    threshold_results, key=lambda x: x['recall'])

                # Display optimal thresholds
                st.success("✅ Threshold sweep complete!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Best F1 Score",
                        f"{best_by_f1['f1']:.2%}",
                        f"Threshold: {best_by_f1['threshold']:.2f}"
                    )

                with col2:
                    st.metric(
                        "Best Precision",
                        f"{best_by_precision['precision']:.2%}",
                        f"Threshold: {best_by_precision['threshold']:.2f}"
                    )

                with col3:
                    st.metric(
                        "Best Recall",
                        f"{best_by_recall['recall']:.2%}",
                        f"Threshold: {best_by_recall['threshold']:.2f}"
                    )

                st.markdown("---")

                # Visualization 1: Metrics vs Threshold
                st.markdown("#### Metrics vs Threshold")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=[r['threshold'] for r in threshold_results],
                    y=[r['precision'] for r in threshold_results],
                    mode='lines+markers',
                    name='Precision',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))

                fig.add_trace(go.Scatter(
                    x=[r['threshold'] for r in threshold_results],
                    y=[r['recall'] for r in threshold_results],
                    mode='lines+markers',
                    name='Recall',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))

                fig.add_trace(go.Scatter(
                    x=[r['threshold'] for r in threshold_results],
                    y=[r['f1'] for r in threshold_results],
                    mode='lines+markers',
                    name='F1 Score',
                    line=dict(color='green', width=3),
                    marker=dict(size=8)
                ))

                # Add vertical line for best F1
                fig.add_vline(
                    x=best_by_f1['threshold'],
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Best F1: {best_by_f1['threshold']:.2f}",
                    annotation_position="top"
                )

                fig.update_layout(
                    title=f"Threshold Analysis - {selected_model_display}",
                    xaxis_title="Distance Threshold",
                    yaxis_title="Score",
                    hovermode='x unified',
                    height=500,
                    yaxis=dict(range=[0, 1.05])
                )

                st.plotly_chart(fig, width='stretch')

                st.markdown("---")

                # Visualization 2: Precision-Recall Curve
                st.markdown("#### Precision-Recall Curve")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=[r['recall'] for r in threshold_results],
                    y=[r['precision'] for r in threshold_results],
                    mode='lines+markers+text',
                    text=[f"{r['threshold']:.2f}" for r in threshold_results],
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=[r['f1'] for r in threshold_results],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="F1 Score")
                    ),
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Threshold: %{text}</b><br>Recall: %{x:.2%}<br>Precision: %{y:.2%}<extra></extra>'
                ))

                fig.update_layout(
                    title="Precision-Recall Trade-off",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    height=500,
                    xaxis=dict(range=[0, 1.05]),
                    yaxis=dict(range=[0, 1.05])
                )

                st.plotly_chart(fig, width='stretch')

                st.markdown("---")

                # Visualization 3: Hit Rate & Accuracy
                st.markdown("#### Hit Rate & Accuracy vs Threshold")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=[r['threshold'] for r in threshold_results],
                    y=[r['hit_rate'] for r in threshold_results],
                    mode='lines+markers',
                    name='Hit Rate',
                    line=dict(color='magenta', width=3),
                    marker=dict(size=8)
                ))

                fig.add_trace(go.Scatter(
                    x=[r['threshold'] for r in threshold_results],
                    y=[r['accuracy'] for r in threshold_results],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='cyan', width=3),
                    marker=dict(size=8)
                ))

                fig.update_layout(
                    title="Hit Rate & Accuracy Analysis",
                    xaxis_title="Distance Threshold",
                    yaxis_title="Rate",
                    hovermode='x unified',
                    height=400,
                    yaxis=dict(range=[0, 1.05])
                )

                st.plotly_chart(fig, width='stretch')

                # Threshold recommendation
                st.markdown("---")
                st.markdown("#### 💡 Threshold Recommendations")

                col1, col2 = st.columns(2)

                with col1:
                    st.info(f"""
                    **For Maximum F1 Score (Balanced):**
                    - Threshold: `{best_by_f1['threshold']:.2f}`
                    - Precision: {best_by_f1['precision']:.2%}
                    - Recall: {best_by_f1['recall']:.2%}
                    - F1: {best_by_f1['f1']:.2%}
                    
                    **Use when:** You want balanced performance
                    """)

                with col2:
                    st.warning(f"""
                    **For Maximum Precision (Minimize False Positives):**
                    - Threshold: `{best_by_precision['threshold']:.2f}`
                    - Precision: {best_by_precision['precision']:.2%}
                    - Recall: {best_by_precision['recall']:.2%}
                    - F1: {best_by_precision['f1']:.2%}
                    
                    **Use when:** Wrong answers are very costly
                    """)

                # Results table
                st.markdown("---")
                st.markdown("#### 📋 Complete Results Table")

                results_df = pd.DataFrame(threshold_results)
                st.dataframe(
                    results_df.style.format({
                        'threshold': '{:.2f}',
                        'precision': '{:.2%}',
                        'recall': '{:.2%}',
                        'f1': '{:.2%}',
                        'accuracy': '{:.2%}',
                        'hit_rate': '{:.2%}'
                    }).background_gradient(subset=['f1'], cmap='RdYlGn'),
                    width='stretch',
                    height=400
                )
        else:
            st.warning("Please select at least one model to test")

    with tab2:
        st.markdown("### Model Performance Comparison")

        if len(selected_models) > 1:

            st.info(
                f"Comparing {len(selected_models)} models at threshold = 0.30")

            with st.spinner("Comparing models..."):

                # Real model comparison
                model_comparison_results = []

                # Create evaluator once (will be used for all models)
                evaluator = CacheEvaluator(ground_truth_df, test_df)
                qa_pairs = list(
                    zip(ground_truth_df["question"], ground_truth_df["answer"]))

                progress_bar = st.progress(0)

                for i, (model_name, model_display, model_type) in enumerate(selected_models):

                    start = time.time()

                    # Load model and create cache
                    model = SentenceTransformer(model_name)
                    model_cache = EvaluatableCache(
                        model, distance_threshold=0.3)
                    model_cache.add_many(qa_pairs)

                    # Evaluate
                    result = evaluator.evaluate(model_cache.check)
                    elapsed = time.time() - start

                    model_comparison_results.append({
                        'model': model_display,
                        'model_name': model_name,
                        'model_type': model_type,
                        'f1': result.f1_score,
                        'precision': result.precision,
                        'recall': result.recall,
                        'time': elapsed
                    })

                    progress_bar.progress((i + 1) / len(selected_models))

                progress_bar.empty()

                st.success("✅ Model comparison complete!")

                # Best model
                best_model = max(model_comparison_results,
                                 key=lambda x: x['f1'])
                fastest_model = min(model_comparison_results,
                                    key=lambda x: x['time'])

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "🏆 Best F1 Score",
                        best_model['model'],
                        f"{best_model['f1']:.2%}"
                    )

                with col2:
                    st.metric(
                        "⚡ Fastest Model",
                        fastest_model['model'],
                        f"{fastest_model['time']:.2f}s"
                    )

                st.markdown("---")

                # Model comparison table
                st.markdown("#### 📊 Model Performance Table")

                comparison_df = pd.DataFrame(model_comparison_results)
                display_df = comparison_df[[
                    'model', 'f1', 'precision', 'recall', 'time']].copy()
                display_df.columns = ['Model', 'F1 Score',
                                      'Precision', 'Recall', 'Time (s)']

                st.dataframe(
                    display_df.style.format({
                        'F1 Score': '{:.2%}',
                        'Precision': '{:.2%}',
                        'Recall': '{:.2%}',
                        'Time (s)': '{:.2f}'
                    }).background_gradient(subset=['F1 Score'], cmap='RdYlGn'),
                    width='stretch'
                )

                st.markdown("---")

                # Model comparison visualizations
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### F1 Score Comparison")

                    fig = go.Figure(data=[
                        go.Bar(
                            x=[r['model'] for r in model_comparison_results],
                            y=[r['f1'] for r in model_comparison_results],
                            marker_color=[r['f1']
                                          for r in model_comparison_results],
                            marker=dict(colorscale='RdYlGn', showscale=False),
                            text=[
                                f"{r['f1']:.2%}" for r in model_comparison_results],
                            textposition='auto'
                        )
                    ])

                    fig.update_layout(
                        xaxis_title="Model",
                        yaxis_title="F1 Score",
                        height=400,
                        yaxis=dict(range=[0, 1.05])
                    )

                    st.plotly_chart(fig, width='stretch')

                with col2:
                    st.markdown("#### Processing Time Comparison")

                    fig = go.Figure(data=[
                        go.Bar(
                            x=[r['model'] for r in model_comparison_results],
                            y=[r['time'] for r in model_comparison_results],
                            marker_color='lightblue',
                            text=[
                                f"{r['time']:.2f}s" for r in model_comparison_results],
                            textposition='auto'
                        )
                    ])

                    fig.update_layout(
                        xaxis_title="Model",
                        yaxis_title="Time (seconds)",
                        height=400
                    )

                    st.plotly_chart(fig, width='stretch')

                # Speed vs Accuracy
                st.markdown("---")
                st.markdown("#### Speed vs Accuracy Trade-off")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=[r['time'] for r in model_comparison_results],
                    y=[r['f1'] * 100 for r in model_comparison_results],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color=[r['f1'] for r in model_comparison_results],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="F1 Score")
                    ),
                    text=[r['model'] for r in model_comparison_results],
                    textposition="top center",
                    hovertemplate='<b>%{text}</b><br>Time: %{x:.2f}s<br>F1: %{y:.1f}%<extra></extra>'
                ))

                fig.update_layout(
                    xaxis_title="Processing Time (seconds)",
                    yaxis_title="F1 Score (%)",
                    title="Model Performance Trade-off",
                    height=500
                )

                st.plotly_chart(fig, width='stretch')

                # Model recommendations
                st.markdown("---")
                st.markdown("#### 💡 Model Recommendations")

                st.success(f"""
                **🏆 Recommended for Production: {best_model['model']}**
                
                - **F1 Score:** {best_model['f1']:.2%}
                - **Precision:** {best_model['precision']:.2%}
                - **Recall:** {best_model['recall']:.2%}
                - **Processing Time:** {best_model['time']:.2f}s
                
                This model provides the best balance of accuracy and performance for your dataset.
                """)

                if fastest_model['model'] != best_model['model']:
                    st.info(f"""
                    **⚡ Alternative (Fastest): {fastest_model['model']}**
                    
                    - **F1 Score:** {fastest_model['f1']:.2%}
                    - **Processing Time:** {fastest_model['time']:.2f}s
                    
                    Consider this if speed is critical and you can accept slightly lower accuracy.
                    """)

        elif len(selected_models) == 1:
            st.warning("Please select at least 2 models to compare")
        else:
            st.warning("Please select models to compare")

    with tab3:
        st.markdown("### 🎯 Optimal Configuration")

        if st.session_state.get('optimization_results'):

            st.success(
                "Based on your optimization results, here's the recommended production configuration:")

            # Get best configuration
            if len(selected_models) > 0:
                # Use results from tab1 if available
                if selected_models and threshold_results:
                    best_threshold = best_by_f1['threshold']
                    best_f1_score = best_by_f1['f1']
                    best_precision = best_by_f1['precision']
                    best_recall = best_by_f1['recall']
                    recommended_model = selected_model_display
                else:
                    best_threshold = 0.30
                    best_f1_score = 0.75
                    best_precision = 0.80
                    best_recall = 0.70
                    recommended_model = "MiniLM-L6 (Fast)"

                # Display configuration card
                st.markdown("### 📋 Production Configuration")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.code(f"""
# Optimal Semantic Cache Configuration

EMBEDDING_MODEL = "{selected_model_name if selected_models else 'all-MiniLM-L6-v2'}"
DISTANCE_THRESHOLD = {best_threshold:.2f}

# Expected Performance
# - Precision: {best_precision:.2%}
# - Recall: {best_recall:.2%}
# - F1 Score: {best_f1_score:.2%}
                    """, language="python")

                with col2:
                    st.metric("Expected F1", f"{best_f1_score:.2%}")
                    st.metric("Precision", f"{best_precision:.2%}")
                    st.metric("Recall", f"{best_recall:.2%}")

                st.markdown("---")

                # Implementation example
                st.markdown("### 💻 Implementation Example")

                st.code(f"""
from sentence_transformers import SentenceTransformer
from your_cache_module import SemanticCache

# Initialize with optimal configuration
encoder = SentenceTransformer("{selected_model_name if selected_models else 'all-MiniLM-L6-v2'}")
cache = SemanticCache(
    encoder=encoder,
    distance_threshold={best_threshold:.2f}
)

# Hydrate cache with your FAQ data
qa_pairs = [
    ("How do I get a refund?", "Visit orders page..."),
    ("Can I reset my password?", "Click Forgot Password..."),
    # ... more pairs
]
cache.add_many(qa_pairs)

# Use in production
def get_answer(user_query: str) -> str:
    result = cache.check(user_query)
    
    if result:
        # Cache hit - return cached answer
        return result['answer']
    else:
        # Cache miss - call LLM
        return call_llm_api(user_query)
                """, language="python")

                st.markdown("---")

                # Deployment checklist
                st.markdown("### ✅ Deployment Checklist")

                checklist = [
                    ("Download and save your embedding model", False),
                    ("Set up cache hydration pipeline", False),
                    ("Implement cache-first query logic", False),
                    ("Add monitoring for hit rate", False),
                    ("Set up A/B testing (optional)", False),
                    ("Configure cache refresh strategy", False)
                ]

                for item, _ in checklist:
                    st.checkbox(item, key=f"checklist_{item}")

                st.markdown("---")

                # Export configuration
                st.markdown("### 💾 Export Configuration")

                col1, col2 = st.columns(2)

                with col1:
                    config_text = f"""# Semantic Cache Configuration
# Generated by Cache Optimizer Dashboard
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

[model]
name = "{selected_model_name if selected_models else 'all-MiniLM-L6-v2'}"
type = "{selected_model_type if selected_models else 'sentence-transformers'}"

[cache]
distance_threshold = {best_threshold:.2f}

[performance]
expected_precision = {best_precision:.2%}
expected_recall = {best_recall:.2%}
expected_f1 = {best_f1_score:.2%}

[dataset]
faq_entries = {len(ground_truth_df)}
test_queries = {len(test_df)}
"""
                    st.download_button(
                        "📥 Download Config (TOML)",
                        config_text,
                        "semantic_cache_config.toml",
                        "text/plain",
                        width='stretch'
                    )

                with col2:
                    json_config = f"""{{
  "model": {{
    "name": "{selected_model_name if selected_models else 'all-MiniLM-L6-v2'}",
    "type": "{selected_model_type if selected_models else 'sentence-transformers'}"
  }},
  "cache": {{
    "distance_threshold": {best_threshold:.2f}
  }},
  "performance": {{
    "expected_precision": {best_precision:.4f},
    "expected_recall": {best_recall:.4f},
    "expected_f1": {best_f1_score:.4f}
  }}
}}"""
                    st.download_button(
                        "📥 Download Config (JSON)",
                        json_config,
                        "semantic_cache_config.json",
                        "application/json",
                        width='stretch'
                    )
        else:
            st.info("Run optimization to see recommended configuration")

    with tab4:
        st.markdown("### 📋 Detailed Metrics & Analysis")

        if threshold_results:

            st.markdown("#### Complete Threshold Analysis")

            # Full results table with all metrics
            detailed_df = pd.DataFrame(threshold_results)

            st.dataframe(
                detailed_df.style.format({
                    'threshold': '{:.3f}',
                    'precision': '{:.3f}',
                    'recall': '{:.3f}',
                    'f1': '{:.3f}',
                    'accuracy': '{:.3f}',
                    'hit_rate': '{:.3f}'
                }),
                width='stretch',
                height=500
            )

            # Download detailed results
            csv = detailed_df.to_csv(index=False)
            st.download_button(
                "📥 Download Detailed Results (CSV)",
                csv,
                "threshold_analysis_detailed.csv",
                "text/csv",
                width='stretch'
            )

            st.markdown("---")

            # Statistical summary
            st.markdown("#### Statistical Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Precision**")
                st.write(f"Mean: {detailed_df['precision'].mean():.2%}")
                st.write(f"Std: {detailed_df['precision'].std():.2%}")
                st.write(f"Min: {detailed_df['precision'].min():.2%}")
                st.write(f"Max: {detailed_df['precision'].max():.2%}")

            with col2:
                st.markdown("**Recall**")
                st.write(f"Mean: {detailed_df['recall'].mean():.2%}")
                st.write(f"Std: {detailed_df['recall'].std():.2%}")
                st.write(f"Min: {detailed_df['recall'].min():.2%}")
                st.write(f"Max: {detailed_df['recall'].max():.2%}")

            with col3:
                st.markdown("**F1 Score**")
                st.write(f"Mean: {detailed_df['f1'].mean():.2%}")
                st.write(f"Std: {detailed_df['f1'].std():.2%}")
                st.write(f"Min: {detailed_df['f1'].min():.2%}")
                st.write(f"Max: {detailed_df['f1'].max():.2%}")
        else:
            st.info("Run optimization to see detailed metrics")

else:
    # Show what will be tested
    st.markdown("## 📝 Configuration Summary")

    st.info(f"""
    **Ready to optimize with:**
    - {len(thresholds)} thresholds from {threshold_min:.2f} to {threshold_max:.2f}
    - {len(selected_models)} embedding models
    - {len(test_df)} test queries
    
    Click "Run Optimization" to begin systematic analysis.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 Tip: Start with a broad threshold range (0.10-0.50), then narrow down based on results</p>
</div>
""", unsafe_allow_html=True)
