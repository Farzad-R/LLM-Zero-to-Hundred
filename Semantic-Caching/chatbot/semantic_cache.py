"""
Semantic Cache Module for RAG Chatbot
Uses OpenAI embeddings for semantic matching of cached Q&A pairs.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document


@dataclass
class CacheResult:
    """Standardized result for cache lookups."""
    prompt: str
    response: str
    vector_distance: float
    cosine_similarity: float


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


class SemanticCache:
    """
    Semantic cache using OpenAI embeddings for meaning-based matching.
    
    This cache understands that "I want my money back" means the same 
    thing as "How do I get a refund?"
    """

    def __init__(self, distance_threshold: float = 0.3):
        """
        Args:
            distance_threshold: Maximum cosine distance for a match (default 0.3)
                              Lower = stricter matching, Higher = looser matching
        """
        self.distance_threshold = distance_threshold
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore: Optional[InMemoryVectorStore] = None
        self.qa_pairs: List[Tuple[str, str]] = []

    def hydrate_from_pairs(self, pairs: List[Tuple[str, str]], clear: bool = True):
        """
        Load Q&A pairs into the cache.
        
        Args:
            pairs: List of (question, answer) tuples
            clear: Whether to clear existing cache
        """
        if clear:
            self.qa_pairs = []
            self.vectorstore = None

        self.qa_pairs.extend(pairs)
        
        # Create documents with questions as content and answers in metadata
        documents = [
            Document(
                page_content=question,
                metadata={"answer": answer, "question": question}
            )
            for question, answer in self.qa_pairs
        ]
        
        # Create or update vectorstore
        self.vectorstore = InMemoryVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

    def add_pair(self, question: str, answer: str):
        """
        Add a single Q&A pair to the cache dynamically.
        
        Args:
            question: The question text
            answer: The answer text
        """
        self.qa_pairs.append((question, answer))
        
        # Add to existing vectorstore
        doc = Document(
            page_content=question,
            metadata={"answer": answer, "question": question}
        )
        
        if self.vectorstore is None:
            self.vectorstore = InMemoryVectorStore.from_documents(
                documents=[doc],
                embedding=self.embeddings
            )
        else:
            self.vectorstore.add_documents([doc])

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
        if self.vectorstore is None or not self.qa_pairs:
            return CacheResults(query=query, matches=[])

        threshold = distance_threshold if distance_threshold is not None else self.distance_threshold

        # Use similarity search with scores
        results = self.vectorstore.similarity_search_with_score(
            query, k=num_results
        )
        
        matches = []
        for doc, score in results:
            # Convert similarity score to distance (lower is better)
            # LangChain returns similarity scores, we need distance
            distance = 1 - score if score <= 1 else score
            
            if distance <= threshold:
                matches.append(CacheResult(
                    prompt=doc.metadata["question"],
                    response=doc.metadata["answer"],
                    vector_distance=float(distance),
                    cosine_similarity=float(1 - distance)
                ))

        return CacheResults(query=query, matches=matches)

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """Get all cached Q&A pairs."""
        return self.qa_pairs.copy()

    def save_to_file(self, filepath: str):
        """Save cache pairs to a CSV file."""
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'answer'])
            writer.writerows(self.qa_pairs)

    @classmethod
    def load_from_file(cls, filepath: str, distance_threshold: float = 0.3):
        """Load cache from a CSV file."""
        import csv
        cache = cls(distance_threshold=distance_threshold)
        pairs = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row['question'], row['answer']))
        
        cache.hydrate_from_pairs(pairs)
        return cache
