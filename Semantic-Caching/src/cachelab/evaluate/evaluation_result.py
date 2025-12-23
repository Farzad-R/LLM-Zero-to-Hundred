from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvaluationResult:
    """Holds evaluation metrics for a cache configuration."""

    # Core metrics
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    # Details for debugging
    predictions: List[Dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Of all hits, how many were correct?"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Of all that should hit, how many did?"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        total = self.true_positives + self.false_positives + \
            self.true_negatives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        """What percentage resulted in cache hits?"""
        total = self.true_positives + self.false_positives + \
            self.true_negatives + self.false_negatives
        return (self.true_positives + self.false_positives) / total if total > 0 else 0.0

    def summary(self) -> str:
        return f"""
        Precision: {self.precision:.2%}  (Of hits, how many correct?)
        Recall:    {self.recall:.2%}  (Of should-hits, how many did?)
        F1 Score:  {self.f1_score:.2%}  (Balance of both)
        Accuracy:  {self.accuracy:.2%}  (Overall correctness)
        Hit Rate:  {self.hit_rate:.2%}  (Total cache hits)
        
        Confusion Matrix:
                        Predicted
                    HIT      MISS
        Should HIT  {self.true_positives:4d}     {self.false_negatives:4d}
        Should MISS {self.false_positives:4d}     {self.true_negatives:4d}
        """
