import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine distance between two vectors.

    Cosine Distance = 1 - Cosine Similarity

    Think of it like this:
    - Two arrows pointing the same direction = distance 0
    - Two arrows perpendicular = distance 1
    - Two arrows pointing opposite = distance 2
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_similarity = dot_product / (norm_a * norm_b)
    return 1 - cosine_similarity


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate straight-line distance between two vectors."""
    return np.linalg.norm(a - b)


def cosine_distance_batch(embeddings: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    """Vectorized cosine distance for efficiency."""
    dot_products = np.dot(embeddings, query_emb)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    return 1 - (dot_products / norms)
