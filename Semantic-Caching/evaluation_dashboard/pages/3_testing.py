"""
Interactive Cache Testing & Experimentation
"""
from sentence_transformers import SentenceTransformer
from cachelab.cache.semantic_match_cache import SemanticCache
from cachelab.cache.fuzzy_match_cache import FuzzyCache
from cachelab.cache.exact_match_cache import ExactMatchCache
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from typing import List, Dict

st.set_page_config(page_title="Cache Testing", page_icon="🧪", layout="wide")

st.title("🧪 Interactive Cache Testing")
st.markdown("### Run experiments and tune parameters in real-time")

st.markdown("---")

# Check if data is loaded
if st.session_state.get('ground_truth_df') is None or st.session_state.get('test_df') is None:
    st.warning("⚠️ Please load data first!")
    if st.button("Go to Data Page"):
        st.switch_page("pages/2_data.py")
    st.stop()

ground_truth_df = st.session_state.ground_truth_df
test_df = st.session_state.test_df

# Sidebar - Configuration
st.sidebar.markdown("## ⚙️ Cache Configuration")

# Select which caches to test
st.sidebar.markdown("### Select Strategies")
test_exact = st.sidebar.checkbox("Exact Match Cache", value=True)
test_fuzzy = st.sidebar.checkbox("Fuzzy Match Cache", value=True)
test_semantic = st.sidebar.checkbox("Semantic Cache", value=True)

st.sidebar.markdown("---")

# Fuzzy cache parameters
st.sidebar.markdown("### 🔤 Fuzzy Match Settings")
fuzzy_threshold = st.sidebar.slider(
    "Fuzzy Distance Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Maximum allowed Levenshtein distance ratio"
)

st.sidebar.markdown("---")

# Semantic cache parameters
st.sidebar.markdown("### 🧠 Semantic Cache Settings")

embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
    help="Sentence transformer model for embeddings"
)

semantic_threshold = st.sidebar.slider(
    "Semantic Distance Threshold",
    min_value=0.05,
    max_value=0.80,
    value=0.42,
    step=0.05,
    help="Maximum cosine distance for cache hit"
)

top_k_display = st.sidebar.slider(
    "Top-K Results to Display",
    min_value=1,
    max_value=10,
    value=5,
    help="Show top-K closest matches for analysis"
)

st.sidebar.markdown("---")

# Test query selection
st.sidebar.markdown("### 🎯 Test Query Options")
num_test_queries = st.sidebar.slider(
    "Number of Test Queries",
    min_value=5,
    max_value=len(test_df),
    value=min(20, len(test_df)),
    help="How many test queries to run"
)

query_filter = st.sidebar.radio(
    "Query Filter",
    ["All Queries", "Should Hit Only", "Should Miss Only"],
    help="Filter test queries by expected behavior"
)

# Main content
st.markdown("## 🚀 Run Cache Tests")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.info(f"""
    **Configuration:**
    - Strategies: {sum([test_exact, test_fuzzy, test_semantic])} selected
    - Test queries: {num_test_queries}
    - Fuzzy threshold: {fuzzy_threshold}
    - Semantic threshold: {semantic_threshold}
    """)

with col2:
    st.success("""
    **✅ Ready:**
    Cache classes loaded from `cachelab` package.
    """)

with col3:
    run_tests = st.button("▶️ Run Tests", type="primary",
                          width='stretch')

st.markdown("---")

# All caches are implemented and imported

# Filter test queries
if query_filter == "Should Hit Only":
    filtered_test_df = test_df[test_df['cache_hit'] == True]
elif query_filter == "Should Miss Only":
    filtered_test_df = test_df[test_df['cache_hit'] == False]
else:
    filtered_test_df = test_df

# Sample queries
test_queries_df = filtered_test_df.sample(
    min(num_test_queries, len(filtered_test_df)))

# Run tests section
# Run tests section
if run_tests:
    # Store results in session state
    st.session_state['test_results'] = None
    st.session_state['test_config'] = {
        'fuzzy_threshold': fuzzy_threshold,
        'semantic_threshold': semantic_threshold,
        'embedding_model': embedding_model,
        'top_k_display': top_k_display,
        'test_exact': test_exact,  # Store which tests were enabled
        'test_fuzzy': test_fuzzy,
        'test_semantic': test_semantic,
        'num_test_queries': num_test_queries,
        'query_filter': query_filter
    }

    st.markdown("## 📊 Test Results")

    # Initialize caches with st.spinner
    with st.spinner("Initializing caches..."):
        # Initialize encoder for semantic cache
        encoder = SentenceTransformer(embedding_model)

        # Create cache instances
        exact_cache = ExactMatchCache()
        fuzzy_cache = FuzzyCache(threshold=fuzzy_threshold)
        semantic_cache = SemanticCache(
            encoder, distance_threshold=semantic_threshold)

        # Hydrate all caches
        exact_cache.hydrate_from_df(ground_truth_df)
        fuzzy_cache.hydrate_from_df(ground_truth_df)
        semantic_cache.hydrate_from_df(ground_truth_df)

    # Initialize results storage
    results = {
        'query': [],
        'expected_hit': [],
        'exact_hit': [],
        'fuzzy_hit': [],
        'semantic_hit': [],
        'semantic_distance': [],
        'semantic_top_matches': []
    }

    # Run actual cache tests
    with st.spinner("Running cache tests..."):
        for idx, row in test_queries_df.iterrows():
            query = row['question']
            expected_hit = row['cache_hit']

            results['query'].append(query)
            results['expected_hit'].append(expected_hit)

            # Test exact match cache
            if test_exact:
                exact_result = exact_cache.check(query)
                exact_hit = exact_result.hit if exact_result else False
                results['exact_hit'].append(exact_hit)
            else:
                results['exact_hit'].append(None)

            # Test fuzzy match cache
            if test_fuzzy:
                fuzzy_result = fuzzy_cache.check(query)
                fuzzy_hit = fuzzy_result.hit if fuzzy_result else False
                results['fuzzy_hit'].append(fuzzy_hit)
            else:
                results['fuzzy_hit'].append(None)

            # Test semantic cache
            if test_semantic:
                # Get ALL distances FIRST (before check, to ensure consistency)
                all_distances = semantic_cache.get_all_distances(query)
                top_matches = all_distances[:top_k_display]

                # Then do the check
                semantic_result = semantic_cache.check(query)
                semantic_hit = semantic_result.hit if semantic_result else False

                # Get distance - use the first distance from all_distances
                semantic_distance = all_distances[0][1] if all_distances else 1.0

                results['semantic_hit'].append(semantic_hit)
                results['semantic_distance'].append(semantic_distance)
                results['semantic_top_matches'].append(top_matches)
            else:
                results['semantic_hit'].append(None)
                results['semantic_distance'].append(None)
                results['semantic_top_matches'].append([])

    results_df = pd.DataFrame(results)

    # **CRITICAL: Store in session state**
    st.session_state['test_results'] = results_df
    st.session_state['exact_cache'] = exact_cache if test_exact else None
    st.session_state['fuzzy_cache'] = fuzzy_cache if test_fuzzy else None
    st.session_state['semantic_cache'] = semantic_cache if test_semantic else None
    # Store the test queries too
    st.session_state['test_queries_df'] = test_queries_df

# **NEW: Check if we have stored results**
if st.session_state.get('test_results') is not None:
    results_df = st.session_state['test_results']

    # Retrieve config
    config = st.session_state.get('test_config', {})
    semantic_threshold = config.get('semantic_threshold', semantic_threshold)
    top_k_display = config.get('top_k_display', top_k_display)

    # Retrieve which tests were enabled
    test_exact = config.get('test_exact', test_exact)
    test_fuzzy = config.get('test_fuzzy', test_fuzzy)
    test_semantic = config.get('test_semantic', test_semantic)

    # Retrieve cache objects
    exact_cache = st.session_state.get('exact_cache')
    fuzzy_cache = st.session_state.get('fuzzy_cache')
    semantic_cache = st.session_state.get('semantic_cache')
    test_queries_df = st.session_state.get('test_queries_df')

    # Summary metrics
    st.markdown("### 📈 Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_tests = len(results_df)
        st.metric("Total Tests", total_tests)

    with col2:
        expected_hits = results_df['expected_hit'].sum()
        st.metric("Expected Hits",
                  f"{expected_hits} ({expected_hits/total_tests:.1%})")

    with col3:
        if test_semantic:
            actual_hits = results_df['semantic_hit'].sum()
            st.metric("Semantic Hits",
                      f"{actual_hits} ({actual_hits/total_tests:.1%})")

    with col4:
        if test_semantic:
            avg_distance = results_df['semantic_distance'].mean()
            st.metric("Avg Distance", f"{avg_distance:.3f}")

    st.markdown("---")

    # Detailed results table
    st.markdown("### 📋 Detailed Results")

    # Prepare display dataframe
    display_df = pd.DataFrame({
        'Query': results_df['query'],
        'Expected': results_df['expected_hit'].apply(lambda x: '✅ Hit' if x else '❌ Miss'),
    })

    if test_exact:
        display_df['Exact'] = results_df['exact_hit'].apply(
            lambda x: '✅' if x else '❌' if x is not None else 'N/A'
        )

    if test_fuzzy:
        display_df['Fuzzy'] = results_df['fuzzy_hit'].apply(
            lambda x: '✅' if x else '❌' if x is not None else 'N/A'
        )

    if test_semantic:
        display_df['Semantic'] = results_df['semantic_hit'].apply(
            lambda x: '✅' if x else '❌' if x is not None else 'N/A'
        )
        display_df['Distance'] = results_df['semantic_distance'].apply(
            lambda x: f"{x:.3f}" if x is not None else 'N/A'
        )

    st.dataframe(display_df, width='stretch', height=400)

    st.markdown("---")

    # Distance analysis
    if test_semantic:
        st.markdown("### 📊 Distance Distribution Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Distance histogram
            fig = go.Figure()

            hit_distances = results_df[results_df['expected_hit']
                                       == True]['semantic_distance'].dropna()
            miss_distances = results_df[results_df['expected_hit']
                                        == False]['semantic_distance'].dropna()

            fig.add_trace(go.Histogram(
                x=hit_distances,
                name='Should Hit',
                marker_color='green',
                opacity=0.7,
                nbinsx=20
            ))

            fig.add_trace(go.Histogram(
                x=miss_distances,
                name='Should Miss',
                marker_color='red',
                opacity=0.7,
                nbinsx=20
            ))

            # Add threshold line
            fig.add_vline(
                x=semantic_threshold,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Threshold: {semantic_threshold}",
                annotation_position="top"
            )

            fig.update_layout(
                title="Distance Distribution by Expected Behavior",
                xaxis_title="Cosine Distance",
                yaxis_title="Count",
                barmode='overlay',
                height=400
            )

            st.plotly_chart(fig, width='stretch')

        with col2:
            # Box plot
            fig = go.Figure()

            fig.add_trace(go.Box(
                y=hit_distances,
                name='Should Hit',
                marker_color='green',
                boxmean='sd'
            ))

            fig.add_trace(go.Box(
                y=miss_distances,
                name='Should Miss',
                marker_color='red',
                boxmean='sd'
            ))

            fig.add_hline(
                y=semantic_threshold,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Threshold: {semantic_threshold}"
            )

            fig.update_layout(
                title="Distance Distribution (Box Plot)",
                yaxis_title="Cosine Distance",
                height=400
            )

            st.plotly_chart(fig, width='stretch')

        # Distance insights
        st.info(f"""
        **📊 Distance Analysis:**
        - Queries that should hit: Mean distance = {hit_distances.mean():.3f}, Std = {hit_distances.std():.3f}
        - Queries that should miss: Mean distance = {miss_distances.mean():.3f}, Std = {miss_distances.std():.3f}
        - Current threshold: {semantic_threshold}
        - Optimal threshold is where these distributions separate best
        """)

    st.markdown("---")

    # Detailed query inspection
    # Detailed query inspection
    st.markdown("### 🔍 Query-Level Inspection")

    st.markdown("Click on a query to see its top-K matches and distances:")

    # Query selector - ADD A UNIQUE KEY
    selected_query_idx = st.selectbox(
        "Select Query to Inspect",
        range(len(results_df)),
        # Show more text
        format_func=lambda i: f"{i+1}. {results_df.iloc[i]['query'][:80]}",
        key="query_selector"  # Add unique key
    )

    if selected_query_idx is not None:
        selected_row = results_df.iloc[selected_query_idx]
        selected_query = selected_row['query']

        # Add debug info to verify query is changing
        st.caption(f"Selected query index: {selected_query_idx}")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Query Details")
            st.markdown(f"**Query:** {selected_query}")
            st.markdown(
                f"**Expected:** {'✅ Should Hit' if selected_row['expected_hit'] else '❌ Should Miss'}")

            if test_semantic and selected_row['semantic_hit'] is not None:
                st.markdown(
                    f"**Semantic Result:** {'✅ Hit' if selected_row['semantic_hit'] else '❌ Miss'}")
                st.markdown(
                    f"**Best Match Distance:** {selected_row['semantic_distance']:.4f}")

                # Threshold comparison
                if selected_row['semantic_distance'] <= semantic_threshold:
                    st.success(
                        f"✅ Distance ({selected_row['semantic_distance']:.4f}) ≤ Threshold ({semantic_threshold})")
                else:
                    st.error(
                        f"❌ Distance ({selected_row['semantic_distance']:.4f}) > Threshold ({semantic_threshold})")

        with col2:
            # **FIX: Recalculate top-K matches on the fly**
            if test_semantic and semantic_cache is not None:
                st.markdown("#### Top-K Closest Matches")

                # CRITICAL: Get fresh top-K matches for the selected query
                # Use the actual query text, not the index
                all_distances = semantic_cache.get_all_distances(
                    selected_query)
                top_matches = all_distances[:top_k_display]

                # Debug: Show how many matches we got
                st.caption(
                    f"Retrieved {len(top_matches)} matches for this query")

                if top_matches:
                    matches_data = []
                    for rank, (match_q, dist) in enumerate(top_matches, 1):
                        matches_data.append({
                            'Rank': rank,
                            'Match': match_q[:60] + '...' if len(match_q) > 60 else match_q,
                            'Distance': f"{dist:.4f}",
                            'Status': '✅' if dist <= semantic_threshold else '❌'
                        })

                    matches_df = pd.DataFrame(matches_data)
                    st.dataframe(matches_df, width='stretch', hide_index=True)
                else:
                    st.warning("No matches found for this query")

        # Visualization of distances for this query
        if test_semantic and semantic_cache is not None:
            st.markdown("#### Distance Visualization")

            # IMPORTANT: Don't call get_all_distances again, reuse the same data
            if top_matches:
                ranks = [i+1 for i in range(len(top_matches))]
                distances = [d for _, d in top_matches]

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=ranks,
                    y=distances,
                    mode='lines+markers',
                    marker=dict(
                        size=10,
                        color=distances,
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Distance")
                    ),
                    line=dict(color='gray', width=2),
                    name='Matches'
                ))

                fig.add_hline(
                    y=semantic_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold: {semantic_threshold}",
                    annotation_position="right"
                )

                fig.update_layout(
                    title=f"Top-{top_k_display} Matches Distance Decay",
                    xaxis_title="Rank",
                    yaxis_title="Cosine Distance",
                    height=300
                )

                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No distance data to visualize")

    st.markdown("---")

    # Comparison across strategies
    if sum([test_exact, test_fuzzy, test_semantic]) > 1:
        st.markdown("### ⚖️ Strategy Comparison")

        comparison_data = {
            'Metric': ['Hit Rate', 'Accuracy', 'Avg Distance'],
            'Expected': [
                f"{expected_hits/total_tests:.1%}",
                '100%',
                'N/A'
            ]
        }

        if test_exact:
            exact_hits = results_df['exact_hit'].sum()
            exact_correct = sum(
                results_df['exact_hit'] == results_df['expected_hit'])
            comparison_data['Exact'] = [
                f"{exact_hits/total_tests:.1%}",
                f"{exact_correct/total_tests:.1%}",
                'N/A'
            ]

        if test_fuzzy:
            fuzzy_hits = results_df['fuzzy_hit'].sum()
            fuzzy_correct = sum(
                results_df['fuzzy_hit'] == results_df['expected_hit'])
            comparison_data['Fuzzy'] = [
                f"{fuzzy_hits/total_tests:.1%}",
                f"{fuzzy_correct/total_tests:.1%}",
                'N/A'
            ]

        if test_semantic:
            semantic_hits = results_df['semantic_hit'].sum()
            semantic_correct = sum(
                results_df['semantic_hit'] == results_df['expected_hit'])
            avg_dist = results_df['semantic_distance'].mean()
            comparison_data['Semantic'] = [
                f"{semantic_hits/total_tests:.1%}",
                f"{semantic_correct/total_tests:.1%}",
                f"{avg_dist:.3f}"
            ]

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, width='stretch', hide_index=True)

    st.markdown("---")

    st.markdown("### ⚡ Performance Benchmark")

    st.info("""
    **Performance Analysis:** Measure speed vs accuracy trade-offs across caching strategies.
    This helps determine if the extra latency of semantic matching is worth the improved hit rate.
    """)

    # **FIX: Only run benchmarks if we have cache objects**
    if test_queries_df is not None and any([exact_cache, fuzzy_cache, semantic_cache]):
        # Actual benchmark function with timing
        def benchmark_cache(cache, queries: List[str], name: str) -> Dict:
            """Benchmark a cache implementation with actual timing."""
            start = time.time()
            results = cache.check_many(queries)
            elapsed = time.time() - start

            hits = sum(1 for r in results if r.hit)

            return {
                "name": name,
                "queries": len(queries),
                "hits": hits,
                "hit_rate": hits / len(queries),
                "total_time_ms": elapsed * 1000,
                "avg_time_ms": (elapsed * 1000) / len(queries)
            }

        # Get list of test queries
        test_queries_list = test_queries_df['question'].tolist()

        # Run benchmarks for enabled strategies
        benchmark_results = []

        with st.spinner("Running performance benchmarks..."):
            if test_exact and exact_cache is not None:
                benchmark_results.append(benchmark_cache(
                    exact_cache, test_queries_list, "Exact Match"))
            if test_fuzzy and fuzzy_cache is not None:
                benchmark_results.append(benchmark_cache(
                    fuzzy_cache, test_queries_list, "Fuzzy Match"))
            if test_semantic and semantic_cache is not None:
                benchmark_results.append(benchmark_cache(
                    semantic_cache, test_queries_list, "Semantic"))

    if benchmark_results:
        # Display benchmark table
        benchmark_df = pd.DataFrame(benchmark_results)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Benchmark Results")

            display_benchmark = benchmark_df[[
                'name', 'hit_rate', 'avg_time_ms', 'total_time_ms']].copy()
            display_benchmark.columns = [
                'Strategy', 'Hit Rate', 'Avg Time (ms)', 'Total Time (ms)']
            display_benchmark['Hit Rate'] = display_benchmark['Hit Rate'].apply(
                lambda x: f"{x:.1%}")
            display_benchmark['Avg Time (ms)'] = display_benchmark['Avg Time (ms)'].apply(
                lambda x: f"{x:.3f}")
            display_benchmark['Total Time (ms)'] = display_benchmark['Total Time (ms)'].apply(
                lambda x: f"{x:.1f}")

            st.dataframe(display_benchmark,
                         width='stretch', hide_index=True)

        with col2:
            st.markdown("#### Performance Insights")

            # Find fastest and highest hit rate
            fastest = min(benchmark_results, key=lambda x: x['avg_time_ms'])
            best_hit_rate = max(benchmark_results, key=lambda x: x['hit_rate'])

            st.metric("⚡ Fastest", fastest['name'],
                      f"{fastest['avg_time_ms']:.3f}ms")
            st.metric("🎯 Best Hit Rate",
                      best_hit_rate['name'], f"{best_hit_rate['hit_rate']:.1%}")

        # Speed vs Hit Rate visualization
        st.markdown("#### Speed vs Hit Rate Trade-off")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[r['avg_time_ms'] for r in benchmark_results],
            y=[r['hit_rate'] * 100 for r in benchmark_results],
            mode='markers+text',
            marker=dict(
                size=[r['hits'] for r in benchmark_results],
                sizemode='area',
                sizeref=2.*max([r['hits']
                               for r in benchmark_results])/(40.**2),
                sizemin=4,
                color=[r['hit_rate'] for r in benchmark_results],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Hit Rate")
            ),
            text=[r['name'] for r in benchmark_results],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Avg Time: %{x:.3f}ms<br>Hit Rate: %{y:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            xaxis_title="Average Query Time (ms)",
            yaxis_title="Hit Rate (%)",
            title="Performance Trade-off Analysis",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, width='stretch')

        st.success("""
        **💡 Analysis:**
        - **Exact Match**: Fastest but lowest hit rate (only exact strings)
        - **Fuzzy Match**: Catches typos, medium speed
        - **Semantic**: Highest hit rate, understands meaning but slower
        
        **Trade-off:** Semantic caching is slower but catches way more valid hits. 
        For production LLM applications, the extra latency (2-5ms) is negligible compared 
        to LLM API calls (100-1000ms), making the improved hit rate well worth it!
        """)

    st.markdown("---")

    # Confusion Matrix Section
    st.markdown("### 🎯 Confusion Matrix Analysis")

    st.info("""
    **Confusion Matrix:** Shows how well each strategy identifies true hits vs misses.
    - **TP (True Positive):** Correctly returned cached answer
    - **TN (True Negative):** Correctly identified cache miss
    - **FP (False Positive):** ⚠️ Returned WRONG cached answer (bad!)
    - **FN (False Negative):** Missed a valid cache hit (wasted API call)
    """)

    # Calculate confusion matrices for each strategy
    confusion_matrices = {}

    y_true = results_df['expected_hit'].values

    if test_exact and results_df['exact_hit'].notna().any():
        y_pred_exact = results_df['exact_hit'].values
        # Create confusion matrix: [[TN, FP], [FN, TP]]
        tn_exact = sum((~y_true) & (~y_pred_exact))
        fp_exact = sum((~y_true) & (y_pred_exact))
        fn_exact = sum((y_true) & (~y_pred_exact))
        tp_exact = sum((y_true) & (y_pred_exact))
        confusion_matrices['Exact Match'] = np.array(
            [[tn_exact, fp_exact], [fn_exact, tp_exact]])

    if test_fuzzy and results_df['fuzzy_hit'].notna().any():
        y_pred_fuzzy = results_df['fuzzy_hit'].values
        tn_fuzzy = sum((~y_true) & (~y_pred_fuzzy))
        fp_fuzzy = sum((~y_true) & (y_pred_fuzzy))
        fn_fuzzy = sum((y_true) & (~y_pred_fuzzy))
        tp_fuzzy = sum((y_true) & (y_pred_fuzzy))
        confusion_matrices['Fuzzy Match'] = np.array(
            [[tn_fuzzy, fp_fuzzy], [fn_fuzzy, tp_fuzzy]])

    if test_semantic and results_df['semantic_hit'].notna().any():
        y_pred_semantic = results_df['semantic_hit'].values
        tn_semantic = sum((~y_true) & (~y_pred_semantic))
        fp_semantic = sum((~y_true) & (y_pred_semantic))
        fn_semantic = sum((y_true) & (~y_pred_semantic))
        tp_semantic = sum((y_true) & (y_pred_semantic))
        confusion_matrices['Semantic Cache'] = np.array(
            [[tn_semantic, fp_semantic], [fn_semantic, tp_semantic]])

    if confusion_matrices:
        # Display confusion matrices side by side
        cols = st.columns(len(confusion_matrices))

        for idx, (strategy_name, cm) in enumerate(confusion_matrices.items()):
            with cols[idx]:
                st.markdown(f"#### {strategy_name}")

                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Miss', 'Hit'],
                    y=['Miss', 'Hit'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 16},
                    showscale=False,
                    hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
                ))

                fig.update_layout(
                    xaxis_title='Predicted',
                    yaxis_title='Actual',
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig, width='stretch')

                # Calculate and display metrics
                tn, fp, fn, tp = cm.ravel()

                accuracy = (tp + tn) / (tp + tn + fp +
                                        fn) if (tp + tn + fp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision +
                                                 recall) if (precision + recall) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                st.markdown(f"""
                **Metrics:**
                - Accuracy: {accuracy:.1%}
                - Precision: {precision:.1%}
                - Recall: {recall:.1%}
                - F1 Score: {f1:.1%}
                - Specificity: {specificity:.1%}
                """)

        # Detailed metrics table
        st.markdown("---")
        st.markdown("#### 📊 Detailed Performance Metrics")

        detailed_metrics = []

        for strategy_name, cm in confusion_matrices.items():
            tn, fp, fn, tp = cm.ravel()

            accuracy = (tp + tn) / (tp + tn + fp +
                                    fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision +
                                             recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            detailed_metrics.append({
                'Strategy': strategy_name,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'Accuracy': f"{accuracy:.1%}",
                'Precision': f"{precision:.1%}",
                'Recall': f"{recall:.1%}",
                'F1': f"{f1:.1%}",
                'Specificity': f"{specificity:.1%}"
            })

        detailed_df = pd.DataFrame(detailed_metrics)
        st.dataframe(detailed_df, width='stretch', hide_index=True)

        st.success("""
        **💡 Key Insights:**
        
        1. **EXACT MATCH:**
           - Perfect precision (no false positives)
           - Very low recall (misses semantic variations)
           - Best for: Scenarios where false positives are unacceptable
        
        2. **FUZZY MATCH:**
           - Catches typos and minor variations
           - Moderate recall improvement
           - Best for: User input with potential spelling errors
        
        3. **SEMANTIC CACHE:**
           - Highest recall (catches semantic variations)
           - May have some false positives
           - Best for: Maximizing cache utilization and understanding intent
        
        **Trade-off:** Choose based on your application's tolerance for:
        - **False Positives:** Returning wrong cached answer
        - **False Negatives:** Missing a cache hit and wasting compute/cost
        """)

    # Export results
    st.markdown("---")
    st.markdown("### 💾 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results (CSV)",
            csv,
            "cache_test_results.csv",
            "text/csv",
            width='stretch'
        )

    with col2:
        st.download_button(
            "📥 Download Configuration",
            f"""Cache Test Configuration
            
Fuzzy Threshold: {fuzzy_threshold}
Semantic Threshold: {semantic_threshold}
Embedding Model: {embedding_model}
Top-K: {top_k_display}
Test Queries: {num_test_queries}
Query Filter: {query_filter}
            """,
            "cache_config.txt",
            "text/plain",
            width='stretch'
        )

else:
    st.info("👆 Configure settings in the sidebar and click 'Run Tests' to begin")

    # Show example queries
    st.markdown("### 📝 Preview Test Queries")
    st.markdown(
        f"These {min(10, len(test_queries_df))} queries will be tested (showing preview):")

    preview_df = test_queries_df.head(10)[['question', 'cache_hit']].copy()
    preview_df['cache_hit'] = preview_df['cache_hit'].apply(
        lambda x: '✅ Should Hit' if x else '❌ Should Miss')
    preview_df.columns = ['Query', 'Expected Behavior']

    st.dataframe(preview_df, width='stretch', hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 Tip: Adjust thresholds in real-time and re-run tests to find optimal settings</p>
</div>
""", unsafe_allow_html=True)
