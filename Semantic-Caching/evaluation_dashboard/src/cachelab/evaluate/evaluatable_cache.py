from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class CacheEntry:
    question: str
    answer: str
    embedding: np.ndarray


class EvaluatableCache:
    """Semantic cache with evaluation-friendly interface."""

    def __init__(self, encoder, distance_threshold: float = 0.3):
        self.encoder = encoder
        self.distance_threshold = distance_threshold
        self.entries: List[CacheEntry] = []
        self._embedding_matrix: Optional[np.ndarray] = None

    def add_many(self, qa_pairs: List[Tuple[str, str]]) -> None:
        questions = [q for q, _ in qa_pairs]
        embeddings = self.encoder.encode(questions, show_progress_bar=False)

        for (question, answer), embedding in zip(qa_pairs, embeddings):
            self.entries.append(CacheEntry(question, answer, embedding))

        self._embedding_matrix = None

    def _get_embedding_matrix(self) -> np.ndarray:
        if self._embedding_matrix is None:
            self._embedding_matrix = np.array(
                [e.embedding for e in self.entries])
        return self._embedding_matrix

    def check(self, query: str, threshold_override: float = None) -> Optional[Dict]:
        """Check cache with optional threshold override."""
        if not self.entries:
            return None

        threshold = threshold_override if threshold_override is not None else self.distance_threshold

        query_embedding = self.encoder.encode(
            [query], show_progress_bar=False)[0]
        embeddings = self._get_embedding_matrix()

        # Vectorized cosine distance
        dot_products = np.dot(embeddings, query_embedding)
        norms = np.linalg.norm(embeddings, axis=1) * \
            np.linalg.norm(query_embedding)
        distances = 1 - (dot_products / norms)

        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        if best_distance <= threshold:
            return {
                "matched_question": self.entries[best_idx].question,
                "answer": self.entries[best_idx].answer,
                "distance": float(best_distance)
            }

        return None

    def get_all_distances(self, query: str) -> List[Tuple[str, float]]:
        """Get distances to all entries (for analysis)."""
        query_embedding = self.encoder.encode(
            [query], show_progress_bar=False)[0]
        embeddings = self._get_embedding_matrix()

        dot_products = np.dot(embeddings, query_embedding)
        norms = np.linalg.norm(embeddings, axis=1) * \
            np.linalg.norm(query_embedding)
        distances = 1 - (dot_products / norms)

        return [(self.entries[i].question, distances[i]) for i in range(len(self.entries))]
