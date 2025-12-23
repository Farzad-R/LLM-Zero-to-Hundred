from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from src.utils.cache_utils import CacheResult, CacheResults
from src.utils.embedding_utils import cosine_distance_batch


class SemanticCache:
    """
    Semantic cache using embeddings for meaning-based matching.

    This is the star of the show - understands that "I want my money back"
    means the same thing as "How do I get a refund?"
    """

    def __init__(self, encoder, distance_threshold: float = 0.3):
        """
        Args:
            encoder: SentenceTransformer model for embeddings
            distance_threshold: Maximum cosine distance for a match (default 0.3)
                              Lower = stricter matching, Higher = looser matching
        """
        self.encoder = encoder
        self.distance_threshold = distance_threshold
        self.entries: List[Tuple[str, str]] = []  # (question, answer)
        self._embedding_matrix: Optional[np.ndarray] = None

    def hydrate_from_df(self, df: pd.DataFrame, q_col: str = "question",
                        a_col: str = "answer", clear: bool = True):
        if clear:
            self.entries = []
            self._embedding_matrix = None

        questions = df[q_col].tolist()
        answers = df[a_col].tolist()

        # Batch encode for efficiency
        embeddings = self.encoder.encode(questions, show_progress_bar=False)

        for q, a in zip(questions, answers):
            self.entries.append((q, a))

        self._embedding_matrix = embeddings

    def hydrate_from_pairs(self, pairs: List[Tuple[str, str]], clear: bool = True):
        if clear:
            self.entries = []
            self._embedding_matrix = None

        questions = [q for q, _ in pairs]
        self.entries = list(pairs)
        self._embedding_matrix = self.encoder.encode(
            questions, show_progress_bar=False)

    def _get_embeddings(self) -> np.ndarray:
        if self._embedding_matrix is None:
            raise ValueError("Cache not hydrated. Call hydrate_from_df first.")
        return self._embedding_matrix

    def check(self, query: str, distance_threshold: Optional[float] = None,
              num_results: int = 1) -> CacheResults:
        """
        Check cache for semantic matches.

        Args:
            query: The query string
            distance_threshold: Override default threshold
            num_results: Number of results to return

        Returns:
            CacheResults with matches (empty if none within threshold)
        """
        if not self.entries:
            return CacheResults(query=query, matches=[])

        threshold = distance_threshold if distance_threshold is not None else self.distance_threshold

        # Embed query
        query_embedding = self.encoder.encode(
            [query], show_progress_bar=False)[0]

        # Calculate distances to all entries
        embeddings = self._get_embeddings()
        distances = cosine_distance_batch(embeddings, query_embedding)

        # Get top matches within threshold
        sorted_indices = np.argsort(distances)
        matches = []

        for idx in sorted_indices[:num_results]:
            dist = distances[idx]
            if dist <= threshold:
                question, answer = self.entries[idx]
                matches.append(CacheResult(
                    prompt=question,
                    response=answer,
                    vector_distance=float(dist),
                    cosine_similarity=float(1 - dist)
                ))

        return CacheResults(query=query, matches=matches)

    def check_many(self, queries: List[str],
                   distance_threshold: Optional[float] = None,
                   num_results: int = 1,
                   show_progress: bool = False) -> List[CacheResults]:
        """Check multiple queries."""
        from tqdm.auto import tqdm
        iterator = tqdm(queries, disable=not show_progress)
        return [self.check(q, distance_threshold, num_results) for q in iterator]

    def get_all_distances(self, query: str) -> List[Tuple[str, float]]:
        """Get distances to all entries (for analysis)."""
        query_embedding = self.encoder.encode(
            [query], show_progress_bar=False)[0]
        embeddings = self._get_embeddings()
        distances = cosine_distance_batch(embeddings, query_embedding)
        return [(self.entries[i][0], distances[i]) for i in range(len(self.entries))]
