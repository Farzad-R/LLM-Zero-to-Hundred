from typing import List, Dict
from cachelab.reranker.simple_keyword_reranker import simple_keyword_reranker
from cachelab.reranker.cross_encoder import CrossEncoderReranker
from cachelab.reranker.llm_reranker import LLMReranker

# Create adapter wrappers that normalize the reranker interfaces


def simple_keyword_reranker_adapter(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    Adapter for simple_keyword_reranker.
    Converts 'question' key to match what simple_keyword_reranker expects.
    """
    if not candidates:
        return []

    # simple_keyword_reranker expects 'question' key (which we already have)
    reranked = simple_keyword_reranker(query, candidates)

    # Normalize the output: ensure 'final_score' exists
    for candidate in reranked:
        if 'final_score' not in candidate:
            candidate['final_score'] = 1 - candidate.get('distance', 1.0)

    return reranked


def cross_encoder_reranker_adapter(cross_encoder_reranker: CrossEncoderReranker):
    """
    Adapter for CrossEncoderReranker.
    Converts 'question' to 'prompt' and adds 'final_score'.
    """
    def adapter(query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []

        # Convert 'question' to 'prompt' for cross encoder
        adapted_candidates = []
        for c in candidates:
            adapted = c.copy()
            adapted['prompt'] = c.get('question', '')
            adapted_candidates.append(adapted)

        # Call cross encoder
        reranked = cross_encoder_reranker(query, adapted_candidates)

        # Normalize output: add 'final_score' from 'reranker_score'
        for candidate in reranked:
            candidate['final_score'] = candidate.get('reranker_score', 0.0)
            # Keep 'question' key for consistency
            if 'prompt' in candidate and 'question' not in candidate:
                candidate['question'] = candidate['prompt']

        return reranked

    return adapter


def llm_reranker_adapter(llm_reranker: LLMReranker):
    """
    Adapter for LLMReranker.
    Converts 'question' to 'prompt' and adds 'final_score'.
    """
    def adapter(query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []

        # Convert 'question' to 'prompt' for LLM
        adapted_candidates = []
        for c in candidates:
            adapted = c.copy()
            adapted['prompt'] = c.get('question', '')
            adapted_candidates.append(adapted)

        # Call LLM reranker
        reranked = llm_reranker(query, adapted_candidates)

        # Normalize output: add 'final_score' from 'reranker_score'
        for candidate in reranked:
            candidate['final_score'] = candidate.get('reranker_score', 0.0)
            # Keep 'question' key for consistency
            if 'prompt' in candidate and 'question' not in candidate:
                candidate['question'] = candidate['prompt']

        return reranked

    return adapter


print("✅ Reranker adapters created")
