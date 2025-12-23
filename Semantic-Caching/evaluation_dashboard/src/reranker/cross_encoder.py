from typing import List, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoder:
    def __init__(self, model_name_or_path="Alibaba-NLP/gte-reranker-modernbert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
        )
        self.model = model.eval()

    def pair_distance(self, query: str, context: str) -> float:
        return 1 - self.predict([query], [context])[0]

    def predict(self, queries: List[str], contexts: List[str]) -> List[float]:
        """
        Direct cross encoder prediction for query-context pairs.

        Args:
            queries: List of query strings
            contexts: List of context strings (same length as queries)

        Returns:
            List of similarity scores [0.0-1.0] for each query-context pair
        """
        pairs = list(zip(queries, contexts))
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(
                **inputs, return_dict=True).logits.view(-1).float()
        probs = torch.sigmoid(outputs).numpy()
        return probs.tolist()

    def create_reranker(self):
        return CrossEncoderReranker(self)


class CrossEncoderReranker:
    def __init__(self, cross_encoder: CrossEncoder):
        self.cross_encoder = cross_encoder

    def __call__(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Cross encoder reranker function for semantic cache integration.

        Args:
            query: The search query
            candidates: List of cache candidate dictionaries

        Returns:
            Filtered and reordered candidates with cross encoder metadata
        """
        if not candidates:
            return []

        # Extract prompts for cross encoder scoring
        prompts = [c.get("prompt", "") for c in candidates]

        # Get cross encoder scores
        scores = self.cross_encoder.predict([query] * len(prompts), prompts)

        # Create scored candidates with metadata using dict comprehension
        validated_candidates = [
            (
                {
                    **candidate,
                    "reranker_type": "cross_encoder",
                    "reranker_score": float(score),
                    "reranker_distance": 1 - float(score),
                },
                score,
            )
            for candidate, score in zip(candidates, scores)
        ]
        # Sort by cross encoder score (highest first)
        validated_candidates.sort(key=lambda x: x[1], reverse=True)

        # Return just the enriched candidates
        return [candidate for candidate, _ in validated_candidates]
