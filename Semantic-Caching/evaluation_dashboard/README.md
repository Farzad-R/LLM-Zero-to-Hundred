# 🧪 Semantic Cache Evaluation Dashboard

Interactive dashboard for testing, analyzing, and optimizing semantic caching strategies for LLM applications.

## Purpose

Evaluate and compare different caching strategies (exact match, fuzzy match, semantic) to find optimal configurations for production deployment. Visualize cache performance, measure precision/recall trade-offs, and identify the best threshold and embedding model for your use case.

## Features

- **Data Management**: Upload and validate ground truth FAQs and test datasets
- **Interactive Testing**: Run cache tests with configurable thresholds and compare strategies
- **Threshold Optimization**: Systematically test multiple thresholds to find optimal settings
- **Model Comparison**: Compare different sentence-transformer embedding models
- **Performance Analysis**: Confusion matrices, precision-recall curves, and hit rate metrics
- **Real-time Visualization**: Distance distributions, benchmark results, and cache behavior

## Installation

```bash
# Install the package
pip install -e .

# Run the dashboard
streamlit run app.py
```

## Usage

1. **Load Data** (Page 2 - Data): Upload your `ground_truth.csv` and `test_dataset.csv`
2. **Run Tests** (Page 3 - Testing): Configure parameters and test cache strategies
3. **Optimize** (Page 4 - Optimization): Find optimal threshold and embedding model
4. **Deploy**: Export configuration for production use

### Dataset Format

**Ground Truth** (`ground_truth.csv`):
```csv
id,question,answer
0,How do I reset my password?,Click 'Forgot Password'...
```

**Test Dataset** (`test_dataset.csv`):
```csv
question,answer,src_question_id,cache_hit
How can I reset my password?,Click 'Forgot Password'...,0,True
What is your refund policy?,Not covered in FAQ,0,False
```

## Configuration

The dashboard uses three caching strategies:

- **Exact Match**: Direct string comparison (fastest, lowest recall)
- **Fuzzy Match**: Levenshtein distance for typo tolerance
- **Semantic Cache**: Vector similarity for meaning-based matching (highest recall)

Adjust thresholds in real-time to find the optimal balance between precision and recall.

## Output

- Cache test results (CSV)
- Configuration files (JSON/TOML)
- Performance metrics and visualizations
- Production-ready threshold recommendations

## Project Structure

```
├── app.py                          # Main dashboard entry point
├── pyproject.toml                  # Package configuration
├── requirements.txt                # Dependencies
├── data/                           # Sample datasets
├── pages/                          # Dashboard pages
│   ├── 1_about.py                  # Introduction and overview
│   ├── 2_data.py                   # Data upload and validation
│   ├── 3_testing.py                # Interactive cache testing
│   ├── 4_optimization.py           # Threshold and model optimization
│   └── 5_reranker.py               # Reranking strategies
├── visualization/
└── src/cachelab/                   # Core package
    ├── cache/                      # Cache implementations
    │   ├── __init__.py
    │   ├── exact_match_cache.py
    │   ├── fuzzy_match_cache.py
    │   └── semantic_match_cache.py
    ├── evaluate/                   # Evaluation framework
    │   ├── __init__.py
    │   ├── cache_evaluator.py
    │   ├── evaluatable_cache.py
    │   └── evaluation_result.py
    ├── reranker/                   # Reranking strategies
    │   ├── __init__.py
    │   ├── adaptors.py
    │   ├── cross_encoder.py
    │   ├── llm_reranker.py
    │   ├── reranked_cache.py
    │   └── simple_keyword_reranker.py
    └── utils/                      # Utility functions
        ├── __init__.py
        ├── cache_utils.py
        └── embedding_utils.py
```

## Requirements

- Python ≥3.11
- Streamlit
- sentence-transformers
- pandas, numpy, plotly