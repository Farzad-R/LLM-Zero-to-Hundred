from typing import List, Optional, Tuple
import pandas as pd
from src.utils.cache_utils import CacheResult, CacheResults


class FuzzyCache:
    """
    Fuzzy string matching cache using Levenshtein distance.

    Great for catching typos and minor variations, but doesn't
    understand meaning - "refund" and "money back" are still different.
    """

    def __init__(self, threshold: float = 0.4):
        """
        Args:
            threshold: Maximum distance (0-1) for a match.
                      Lower = stricter, Higher = looser
        """
        self.store: List[Tuple[str, str]] = []
        self.threshold = threshold

    def hydrate_from_df(self, df: pd.DataFrame, q_col: str = "question",
                        a_col: str = "answer", clear: bool = True):
        if clear:
            self.store = []
        for _, row in df.iterrows():
            self.store.append((row[q_col], row[a_col]))

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """
        Calculate similarity ratio using Levenshtein distance.

        Returns value between 0 (completely different) and 1 (identical).
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def check(self, query: str, distance_threshold: Optional[float] = None
              ) -> CacheResults:
        threshold = distance_threshold if distance_threshold is not None else self.threshold

        best_ratio = 0
        best_match = None

        for question, answer in self.store:
            ratio = self._levenshtein_ratio(query, question)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = (question, answer)

        # Convert ratio to distance (1 - ratio)
        distance = 1 - best_ratio

        if distance <= threshold and best_match:
            return CacheResults(
                query=query,
                matches=[CacheResult(
                    prompt=best_match[0],
                    response=best_match[1],
                    vector_distance=distance,
                    cosine_similarity=best_ratio
                )]
            )

        return CacheResults(query=query, matches=[])

    def check_many(self, queries: List[str],
                   distance_threshold: Optional[float] = None) -> List[CacheResults]:
        return [self.check(q, distance_threshold) for q in queries]
