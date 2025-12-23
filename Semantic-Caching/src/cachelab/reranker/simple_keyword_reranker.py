from typing import List, Dict

# Simple reranker example (in practice, use a cross-encoder model)


def simple_keyword_reranker(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    Simple reranker that boosts candidates with keyword overlap.
    (In production, use a cross-encoder model instead!)
    """
    query_words = set(query.lower().split())

    for candidate in candidates:
        candidate_words = set(candidate["question"].lower().split())
        overlap = len(query_words & candidate_words)
        # Combine embedding distance with keyword overlap
        candidate["final_score"] = (
            1 - candidate["distance"]) + (overlap * 0.1)

    return sorted(candidates, key=lambda x: x["final_score"], reverse=True)
