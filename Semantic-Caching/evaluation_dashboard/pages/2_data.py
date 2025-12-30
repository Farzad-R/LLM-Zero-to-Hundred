"""
Data Visualization & Upload Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

st.title("📊 Data Overview & Upload")
st.markdown("### Visualize and analyze your ground truth and test datasets")

st.markdown("---")

# File paths
DATA_DIR = Path("data")
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.csv"
TEST_DATA_PATH = DATA_DIR / "test_dataset.csv"

# Initialize session state
if 'ground_truth_df' not in st.session_state:
    st.session_state.ground_truth_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None

# Data requirements info
with st.expander("📋 Data Format Requirements", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Ground Truth Dataset
        **File:** `ground_truth.csv`
        
        **Required Columns:**
        - 'id' - Unique identifier for each FAQ entry
        - `question` - The FAQ question
        - `answer` - The cached answer
        
        **Purpose:** This is your cache content - the questions and answers that will be stored in the cache.
        
        **Example:**
        ```csv
        question,answer
        How do I get a refund?,"Visit your orders page and click refund"
        Can I reset my password?,"Click 'Forgot Password' on login page"
        Where is my order?,"Track your order in My Orders section"
        ```
        """)

    with col2:
        st.markdown("""
        ### Test Dataset
        **File:** `test_dataset.csv`
        
        **Required Columns:**
        - `question` - Test query
        - `answer` - Expected answer
        - `src_question_id` - ID linking to ground truth question
        - `cache_hit` - True/False (should this hit cache?)
        
        **Purpose:** Test queries that represent your real business needs. Should include:
        - ✅ Variations of cached questions (should hit)
        - ❌ Unrelated questions (should miss)
        - 🔄 Edge cases and paraphrases
        
        **Example:**
        ```csv
        question,answer,cache_hit
        I want my money back,"Visit your orders...",True
        refund process help,"Visit your orders...",True
        What's the weather today,"I don't have that info",False
        ```
        """)

st.markdown("---")

# Data loading section
st.markdown("## 📂 Load Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Option 1: Use Default Data")

    # Check if default files exist
    ground_truth_exists = GROUND_TRUTH_PATH.exists()
    test_exists = TEST_DATA_PATH.exists()

    if ground_truth_exists:
        st.success(f"✅ Found: `{GROUND_TRUTH_PATH}`")
    else:
        st.warning(f"⚠️ Not found: `{GROUND_TRUTH_PATH}`")

    if test_exists:
        st.success(f"✅ Found: `{TEST_DATA_PATH}`")
    else:
        st.warning(f"⚠️ Not found: `{TEST_DATA_PATH}`")

    if ground_truth_exists and test_exists:
        if st.button("📥 Load Default Data", type="primary", use_container_width="True"):
            try:
                st.session_state.ground_truth_df = pd.read_csv(
                    GROUND_TRUTH_PATH)
                st.session_state.test_df = pd.read_csv(TEST_DATA_PATH)
                st.success("✅ Default data loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading default data: {str(e)}")

with col2:
    st.markdown("### Option 2: Upload Your Own Data")

    ground_truth_file = st.file_uploader(
        "Upload Ground Truth CSV",
        type=['csv'],
        key="ground_truth_upload",
        help="CSV with 'question' and 'answer' columns"
    )

    test_file = st.file_uploader(
        "Upload Test Dataset CSV",
        type=['csv'],
        key="test_upload",
        help="CSV with 'question', 'answer', and 'cache_hit' columns"
    )

    if ground_truth_file and test_file:
        if st.button("📤 Load Uploaded Data", type="primary", width='stretch'):
            try:
                # Load dataframes
                ground_truth_df = pd.read_csv(ground_truth_file)
                test_df = pd.read_csv(test_file)

                # Validate columns
                gt_cols = set(ground_truth_df.columns)
                test_cols = set(test_df.columns)

                required_gt_cols = {'question', 'answer'}
                required_test_cols = {'question', 'answer', 'cache_hit'}

                if not required_gt_cols.issubset(gt_cols):
                    missing = required_gt_cols - gt_cols
                    st.error(f"❌ Ground truth missing columns: {missing}")
                elif not required_test_cols.issubset(test_cols):
                    missing = required_test_cols - test_cols
                    st.error(f"❌ Test dataset missing columns: {missing}")
                else:
                    st.session_state.ground_truth_df = ground_truth_df
                    st.session_state.test_df = test_df
                    st.success("✅ Uploaded data loaded successfully!")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error loading uploaded data: {str(e)}")

# Display data if loaded
if st.session_state.ground_truth_df is not None and st.session_state.test_df is not None:

    ground_truth_df = st.session_state.ground_truth_df
    test_df = st.session_state.test_df

    st.markdown("---")
    st.markdown("## 📈 Data Analysis")

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Ground Truth Entries",
            len(ground_truth_df),
            help="Number of FAQ questions in cache"
        )

    with col2:
        st.metric(
            "Test Queries",
            len(test_df),
            help="Total number of test queries"
        )

    with col3:
        should_hit = test_df['cache_hit'].sum()
        hit_rate = should_hit / len(test_df) * 100
        st.metric(
            "Should Hit Cache",
            f"{should_hit} ({hit_rate:.1f}%)",
            help="Queries that should return a cached answer"
        )

    with col4:
        should_miss = (~test_df['cache_hit']).sum()
        miss_rate = should_miss / len(test_df) * 100
        st.metric(
            "Should Miss Cache",
            f"{should_miss} ({miss_rate:.1f}%)",
            help="Queries that should not match any cached answer"
        )

    st.markdown("---")

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Preview",
        "📊 Distribution Analysis",
        "🔍 Text Analytics",
        "💡 Data Insights"
    ])

    with tab1:
        st.markdown("### Ground Truth Dataset")
        st.dataframe(
            ground_truth_df,
            width='stretch',
            height=300
        )

        st.markdown("### Test Dataset")
        st.dataframe(
            test_df,
            width='stretch',
            height=300
        )

    with tab2:
        st.markdown("### Cache Hit Distribution")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Pie chart
            hit_counts = test_df['cache_hit'].value_counts()
            fig = px.pie(
                values=hit_counts.values,
                names=['Should Hit' if x else 'Should Miss' for x in hit_counts.index],
                title="Expected Cache Behavior",
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, width='stretch')

        with col2:
            # Bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=['Should Hit', 'Should Miss'],
                    y=[should_hit, should_miss],
                    marker_color=['#2ecc71', '#e74c3c'],
                    text=[should_hit, should_miss],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Test Query Distribution",
                yaxis_title="Number of Queries",
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')

        # Coverage analysis
        st.markdown("### Coverage Analysis")
        st.info(f"""
        **Dataset Balance:**
        - Your test set has {hit_rate:.1f}% positive samples (should hit cache)
        - For robust evaluation, aim for 30-50% positive samples
        - Current ratio: {should_hit}:{should_miss} (hit:miss)
        """)

    with tab3:
        st.markdown("### Text Length Analysis")

        # Calculate text lengths
        ground_truth_df['question_length'] = ground_truth_df['question'].str.len()
        ground_truth_df['answer_length'] = ground_truth_df['answer'].str.len()
        test_df['question_length'] = test_df['question'].str.len()

        col1, col2 = st.columns(2)

        with col1:
            # Ground truth question lengths
            fig = px.histogram(
                ground_truth_df,
                x='question_length',
                nbins=20,
                title="Ground Truth Question Lengths",
                labels={'question_length': 'Characters', 'count': 'Frequency'},
                color_discrete_sequence=['#3498db']
            )
            fig.add_vline(
                x=ground_truth_df['question_length'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {ground_truth_df['question_length'].mean():.0f}",
                annotation_position="top right"
            )
            st.plotly_chart(fig, width='stretch')

            st.metric(
                "Avg Question Length",
                f"{ground_truth_df['question_length'].mean():.0f} chars"
            )

        with col2:
            # Test query lengths
            fig = px.histogram(
                test_df,
                x='question_length',
                nbins=20,
                title="Test Query Lengths",
                labels={'question_length': 'Characters', 'count': 'Frequency'},
                color_discrete_sequence=['#9b59b6']
            )
            fig.add_vline(
                x=test_df['question_length'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {test_df['question_length'].mean():.0f}",
                annotation_position="top right"
            )
            st.plotly_chart(fig, width='stretch')

            st.metric(
                "Avg Query Length",
                f"{test_df['question_length'].mean():.0f} chars"
            )

        # Answer lengths
        st.markdown("### Answer Length Distribution")
        fig = px.histogram(
            ground_truth_df,
            x='answer_length',
            nbins=20,
            title="Cached Answer Lengths",
            labels={'answer_length': 'Characters', 'count': 'Frequency'},
            color_discrete_sequence=['#1abc9c']
        )
        fig.add_vline(
            x=ground_truth_df['answer_length'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {ground_truth_df['answer_length'].mean():.0f}",
            annotation_position="top right"
        )
        st.plotly_chart(fig, width='stretch')

    with tab4:
        st.markdown("### 💡 Data Quality Insights")

        # Calculate various metrics
        gt_count = len(ground_truth_df)
        test_count = len(test_df)
        hit_ratio = should_hit / test_count

        # Insights
        insights = []

        # Dataset size
        if gt_count < 10:
            insights.append(("⚠️ Small Ground Truth Set",
                             f"You have only {gt_count} FAQ entries. Consider adding more for better coverage.",
                             "warning"))
        elif gt_count > 50:
            insights.append(("✅ Good Ground Truth Size",
                             f"You have {gt_count} FAQ entries - good coverage for testing.",
                             "success"))
        else:
            insights.append(("✅ Adequate Ground Truth Size",
                             f"You have {gt_count} FAQ entries - reasonable starting point.",
                             "info"))

        # Test set size
        if test_count < 50:
            insights.append(("⚠️ Small Test Set",
                             f"Only {test_count} test queries. Aim for 50-100+ for reliable evaluation.",
                             "warning"))
        else:
            insights.append(("✅ Good Test Set Size",
                             f"You have {test_count} test queries - sufficient for evaluation.",
                             "success"))

        # Balance
        if hit_ratio < 0.2:
            insights.append(("⚠️ Imbalanced Dataset",
                             f"Only {hit_ratio:.1%} of queries should hit cache. This may lead to poor recall estimates.",
                             "warning"))
        elif hit_ratio > 0.7:
            insights.append(("⚠️ Too Many Positive Samples",
                             f"{hit_ratio:.1%} of queries should hit cache. Consider adding more negative examples.",
                             "warning"))
        else:
            insights.append(("✅ Well-Balanced Dataset",
                             f"{hit_ratio:.1%} of queries should hit cache - good balance for evaluation.",
                             "success"))

        # Coverage ratio
        coverage_ratio = test_count / gt_count if gt_count > 0 else 0
        if coverage_ratio < 3:
            insights.append(("⚠️ Low Test Coverage",
                             f"Test-to-ground-truth ratio is {coverage_ratio:.1f}:1. Aim for at least 5:1 for robust testing.",
                             "warning"))
        else:
            insights.append(("✅ Good Test Coverage",
                             f"Test-to-ground-truth ratio is {coverage_ratio:.1f}:1 - excellent coverage.",
                             "success"))

        # Length consistency
        gt_avg_len = ground_truth_df['question_length'].mean()
        test_avg_len = test_df['question_length'].mean()
        len_diff = abs(gt_avg_len - test_avg_len) / gt_avg_len

        if len_diff > 0.5:
            insights.append(("⚠️ Length Mismatch",
                             f"Ground truth questions ({gt_avg_len:.0f} chars) differ significantly from test queries ({test_avg_len:.0f} chars). This may affect results.",
                             "warning"))
        else:
            insights.append(("✅ Consistent Question Lengths",
                             f"Ground truth and test query lengths are similar ({gt_avg_len:.0f} vs {test_avg_len:.0f} chars).",
                             "success"))

        # Display insights
        for title, message, level in insights:
            if level == "success":
                st.success(f"**{title}**\n\n{message}")
            elif level == "warning":
                st.warning(f"**{title}**\n\n{message}")
            else:
                st.info(f"**{title}**\n\n{message}")

        st.markdown("---")

        # Recommendations
        st.markdown("### 📋 Recommendations for Your Dataset")

        recommendations = []

        if gt_count < 20:
            recommendations.append(
                "📈 **Expand Ground Truth:** Add more FAQ entries to improve cache coverage")

        if test_count < 50:
            recommendations.append(
                "📊 **Increase Test Set:** Add more test queries for reliable evaluation")

        if hit_ratio < 0.25:
            recommendations.append(
                "✅ **Add Positive Examples:** Include more paraphrases of existing FAQs")

        if hit_ratio > 0.65:
            recommendations.append(
                "❌ **Add Negative Examples:** Include more out-of-scope queries to test false positive rate")

        if len_diff > 0.3:
            recommendations.append(
                "📝 **Normalize Query Lengths:** Ensure test queries match real user query patterns")

        if not recommendations:
            st.success(
                "✅ **Your dataset looks good!** You're ready to proceed with cache analysis.")
        else:
            for rec in recommendations:
                st.info(rec)

    st.markdown("---")

    # Action buttons
    st.markdown("## 🚀 Next Steps")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Reload Data", width='stretch'):
            st.session_state.ground_truth_df = None
            st.session_state.test_df = None
            st.rerun()

    with col2:
        st.download_button(
            "📥 Export Analysis",
            data=f"""Dataset Analysis Summary
            
Ground Truth: {len(ground_truth_df)} entries
Test Queries: {len(test_df)} queries
Should Hit: {should_hit} ({hit_rate:.1f}%)
Should Miss: {should_miss} ({miss_rate:.1f}%)
Avg Question Length: {gt_avg_len:.0f} chars
Avg Query Length: {test_avg_len:.0f} chars
""",
            file_name="data_analysis.txt",
            mime="text/plain",
            width='stretch'
        )

    with col3:
        if st.button("➡️ Start Analysis", type="primary", width='stretch'):
            st.info("Analysis page coming next! 🚀")

else:
    st.info("👆 Load your data using one of the options above to begin analysis")

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 Tip: Ensure your test dataset represents real business queries for accurate evaluation</p>
</div>
""", unsafe_allow_html=True)
