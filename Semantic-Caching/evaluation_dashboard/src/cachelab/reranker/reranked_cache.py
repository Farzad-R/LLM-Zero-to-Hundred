from cachelab.evaluate.evaluatable_cache import EvaluatableCache
from typing import Callable, Dict, Optional
import numpy as np


class RerankedCache:
    """Cache with two-stage retrieval: embedding + reranking."""

    def __init__(self, cache: EvaluatableCache, reranker_fn: Callable):
        self.cache = cache
        self.reranker_fn = reranker_fn

    def check(self, query: str, top_k: int = 5, threshold: float = 0.3) -> Optional[Dict]:
        """
        Two-stage retrieval:
        1. Get top-K candidates by embedding similarity
        2. Rerank candidates using reranker function
        """
        # Stage 1: Get top-K candidates
        query_embedding = self.cache.encoder.encode(
            [query], show_progress_bar=False)[0]
        embeddings = self.cache._get_embedding_matrix()

        dot_products = np.dot(embeddings, query_embedding)
        norms = np.linalg.norm(embeddings, axis=1) * \
            np.linalg.norm(query_embedding)
        distances = 1 - (dot_products / norms)

        # Get top-K indices
        top_k_indices = np.argsort(distances)[:top_k]

        candidates = [
            {
                "question": self.cache.entries[i].question,
                "answer": self.cache.entries[i].answer,
                "distance": float(distances[i])
            }
            for i in top_k_indices
            if distances[i] <= threshold * 2  # Loose threshold for candidates
        ]

        if not candidates:
            return None

        # Stage 2: Rerank
        reranked = self.reranker_fn(query, candidates)
        # print(reranked)
        if reranked and reranked[0]["final_score"] > 0.5:
            best = reranked[0]

            return {
                "matched_question": best["question"],
                "answer": best["answer"],
                "distance": best["distance"],
                "reranked_score": best["final_score"]
            }
        return None
