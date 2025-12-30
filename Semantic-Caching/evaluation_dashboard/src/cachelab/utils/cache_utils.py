from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CacheResult:
    """Standardized result for cache lookups."""
    prompt: str
    response: str
    vector_distance: float
    cosine_similarity: float

    # Optional metadata for rerankers
    reranker_type: Optional[str] = None
    reranker_score: Optional[float] = None
    reranker_reason: Optional[str] = None


@dataclass
class CacheResults:
    """Container for query results with multiple potential matches."""
    query: str
    matches: List[CacheResult] = field(default_factory=list)

    def __repr__(self):
        return f"CacheResults(query='{self.query}', matches={len(self.matches)})"

    @property
    def hit(self) -> bool:
        return len(self.matches) > 0

    @property
    def best_match(self) -> Optional[CacheResult]:
        return self.matches[0] if self.matches else None
