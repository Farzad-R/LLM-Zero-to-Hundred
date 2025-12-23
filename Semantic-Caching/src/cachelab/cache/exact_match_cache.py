from typing import Dict, List
import pandas as pd
from cachelab.utils.cache_utils import CacheResult, CacheResults


class ExactMatchCache:
    """
    Traditional exact-match cache (case-insensitive).

    The simplest cache - only matches if the query is exactly
    the same as something we've seen before.
    """

    def __init__(self):
        self.cache: Dict[str, str] = {}

    def hydrate_from_df(self, df: pd.DataFrame, q_col: str = "question",
                        a_col: str = "answer", clear: bool = True):
        if clear:
            self.cache = {}
        for _, row in df.iterrows():
            self.cache[row[q_col].lower().strip()] = row[a_col]

    def check(self, query: str) -> CacheResults:
        key = query.lower().strip()
        if key in self.cache:
            return CacheResults(
                query=query,
                matches=[CacheResult(
                    prompt=key,
                    response=self.cache[key],
                    vector_distance=0.0,
                    cosine_similarity=1.0
                )]
            )
        return CacheResults(query=query, matches=[])

    def check_many(self, queries: List[str], **kwargs) -> List[CacheResults]:
        return [self.check(q) for q in queries]
