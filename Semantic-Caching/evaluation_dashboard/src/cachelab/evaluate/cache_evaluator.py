from typing import Callable, Dict, Optional
import pandas as pd
from cachelab.evaluate.evaluation_result import EvaluationResult


class CacheEvaluator:
    """Evaluates semantic cache performance against labeled data."""

    def __init__(self, faq_df: pd.DataFrame, test_df: pd.DataFrame):
        self.faq_df = faq_df
        self.test_df = test_df

        # Build lookup for expected answers
        self.question_to_idx = {
            q: i for i, q in enumerate(faq_df["question"])
        }

    def _get_expected_match(self, row) -> Optional[str]:
        """Get the expected cache match for a test query."""
        if not row["cache_hit"]:
            return None  # Should miss

        src_id = row["src_question_id"]
        return self.faq_df.iloc[src_id]["question"]

    def evaluate(
        self,
        cache_check_fn: Callable[[str], Optional[Dict]],
        verbose: bool = False
    ) -> EvaluationResult:
        """
        Evaluate a cache implementation.

        Args:
            cache_check_fn: Function that takes query, returns match dict or None
            verbose: Print detailed results

        Returns:
            EvaluationResult with metrics
        """
        result = EvaluationResult()

        for _, row in self.test_df.iterrows():
            query = row["question"]
            should_hit = row["cache_hit"]
            expected_match = self._get_expected_match(row)

            # Check the cache
            cache_result = cache_check_fn(query)
            did_hit = cache_result is not None

            # Determine if hit was correct
            if did_hit:
                actual_match = cache_result.get("matched_question", "")
                # Check if matched the right question
                correct_match = (
                    expected_match is not None and
                    actual_match == expected_match
                )
            else:
                actual_match = None
                correct_match = False

            # Update counts
            if should_hit:
                if did_hit and correct_match:
                    result.true_positives += 1
                elif did_hit and not correct_match:
                    result.false_positives += 1  # Hit wrong answer!
                else:
                    result.false_negatives += 1
            else:  # Should miss
                if did_hit:
                    result.false_positives += 1
                else:
                    result.true_negatives += 1

            # Store for debugging
            result.predictions.append({
                "query": query,
                "should_hit": should_hit,
                "did_hit": did_hit,
                "expected": expected_match,
                "actual": actual_match,
                "correct": (did_hit == should_hit) and (not did_hit or correct_match)
            })

        if verbose:
            print("\n📋 Detailed Results:")
            for p in result.predictions:
                status = "✅" if p["correct"] else "❌"
                print(f"{status} '{p['query'][:40]}'")
                if not p["correct"]:
                    print(f"   Expected: {p['expected']}")
                    print(f"   Got: {p['actual']}")

        return result
