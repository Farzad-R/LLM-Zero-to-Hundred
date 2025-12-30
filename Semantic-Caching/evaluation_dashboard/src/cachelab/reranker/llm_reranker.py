from typing import List, Tuple, Dict
import pandas as pd
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from tqdm import tqdm
import math


class SimilarityResult(BaseModel):
    reason: str = Field(
        description="Explanation of why the answer was True or False")
    is_similar: bool = Field(
        description="True if the sentences mean the same thing, False otherwise"
    )


@dataclass(frozen=True)
class LLMEvaluationResult:
    resulting_items: List[SimilarityResult]

    @property
    def df(self):
        return pd.DataFrame([dict(it) for it in self.resulting_items])


def batch_iterable(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i: i + batch_size]


DEFAULT_COMPARE_PROMPT_TEMPLATE = """
You are a helpful assistant that determines if two sentences mean the same thing.
You will be given two sentences and you will need to determine if they mean the same thing.
You will need to return a JSON object with the following fields:
- is_similar: True if the sentences mean the same thing, False otherwise
- reason: Explanation of why the answer was True or False
The two sentences are:
- sentence1: {sentence1}
- sentence2: {sentence2}
"""


class LLMEvaluator:
    @staticmethod
    def construct_with_gpt(
        prompt=DEFAULT_COMPARE_PROMPT_TEMPLATE, model="gpt-4.1-mini"
    ):
        llm = ChatOpenAI(model=model).with_structured_output(SimilarityResult)
        return LLMEvaluator(llm, prompt)

    def __init__(self, llm: BaseChatModel, prompt: str):
        prompt = PromptTemplate(
            template=prompt,
            input_variables=["sentence1", "sentence2"],
        )
        # IMPORTANT: do NOT bind tools when using structured output
        self.prompt = prompt
        self.chain: Runnable = prompt | llm

    def predict(
        self,
        dataset: List[Tuple[str, str]],
        batch_size: int,
        show_progress: bool = True,
    ) -> LLMEvaluationResult:
        all_results = []
        dataset = list(dataset)
        num_batches = math.ceil(len(dataset) / batch_size)

        for batch in tqdm(
            batch_iterable(dataset, batch_size),
            total=num_batches,
            disable=not show_progress,
        ):
            batch_payload = [{"sentence1": s1, "sentence2": s2}
                             for s1, s2 in batch]
            try:
                batch_results = self.chain.batch(batch_payload)
                # Ensure we have a list of SimilarityResult (some providers may return dicts)
                for r in batch_results:
                    if isinstance(r, SimilarityResult):
                        all_results.append(r)
                    else:
                        all_results.append(SimilarityResult.model_validate(r))
            except Exception as e:
                print(f"Error in batch: {e}")
                # Optional: append a default negative result instead of raising
                for _ in batch_payload:
                    all_results.append(SimilarityResult(
                        is_similar=False, reason=str(e)))

        return LLMEvaluationResult(all_results)

    def create_reranker(self, batch_size: int = 5) -> "LLMReranker":
        return LLMReranker(self, batch_size=batch_size)


class LLMReranker:
    def __init__(self, llm_evaluator: LLMEvaluator, batch_size: int = 5):
        self.llm_evaluator = llm_evaluator
        self.batch_size = batch_size

    def __call__(self, query: str, candidates: List[Dict]):
        if not candidates:
            return []

        # Prepare query-prompt pairs for LLM validation
        validation_pairs = []
        for candidate in candidates:
            prompt = candidate.get("prompt", "")
            validation_pairs.append((query, prompt))

        # Get LLM validation results
        llm_result = self.llm_evaluator.predict(
            validation_pairs, batch_size=self.batch_size, show_progress=False
        )

        # Filter and enrich candidates based on LLM validation
        validated_candidates = [
            {
                **candidate,
                "reranker_type": "llm",
                "reranker_score": 1.0 if validation.is_similar else 0.0,
                "reranker_distance": 0.0 if validation.is_similar else 1.0,
                "reranker_reason": validation.reason,
            }
            for candidate, validation in zip(candidates, llm_result.resulting_items)
            if validation.is_similar
        ]
        return validated_candidates
