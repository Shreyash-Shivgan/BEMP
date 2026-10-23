"""Embedding generation and distance calculation for attack deduplication.

Supports real embeddings via sentence-transformers / Ollama all-MiniLM-L6-v2,
and deterministic pseudo-embeddings for mock/CI execution without GPU or weight downloads.
"""

import hashlib
import re
from typing import List, Optional
import numpy as np

from strikeshield.config import settings
from strikeshield.errors import StrikeShieldError

EMBEDDING_DIM = 384
_SENTENCE_TRANSFORMER_MODEL = None


def _deterministic_mock_embed(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Generates a deterministic 384-dimensional unit vector from text tokens.

    Uses SHA-256 seeded pseudo-random projections of character n-grams and words.
    Guarantees:
    - Identical text -> cosine distance == 0.0
    - Distinct text -> distinct directions with high probability
    """
    cleaned = text.strip().lower()
    if not cleaned:
        # Return deterministic zero-centered unit vector
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = 1.0
        return vec

    tokens = re.findall(r"\w+", cleaned)
    # Also include 3-character n-grams for fine-grained morphological matching
    ngrams = [cleaned[i : i + 3] for i in range(max(0, len(cleaned) - 2))]
    all_features = tokens + ngrams

    combined = np.zeros(dim, dtype=np.float32)

    for feat in all_features:
        # Seed NumPy generator deterministically per feature
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big") % (2**32)
        rng = np.random.default_rng(seed)
        feature_vec = rng.standard_normal(dim).astype(np.float32)
        combined += feature_vec

    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    else:
        combined[0] = 1.0

    return combined


def embed(texts: List[str]) -> np.ndarray:
    """Embeds a list of strings into an (N, 384) float32 numpy array.

    In LLM_BACKEND=mock mode, deterministic hash-based pseudo-embeddings are used
    to enable full pipeline tests in CI without model weights.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    if settings.llm_backend == "mock":
        vectors = [_deterministic_mock_embed(t) for t in texts]
        return np.vstack(vectors).astype(np.float32)

    # Real mode execution
    global _SENTENCE_TRANSFORMER_MODEL
    try:
        if _SENTENCE_TRANSFORMER_MODEL is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model_name = settings.models.embeddings.model
            if not model_name.startswith("sentence-transformers/"):
                model_name = f"sentence-transformers/{model_name}"
            _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer(model_name)

        embeddings = _SENTENCE_TRANSFORMER_MODEL.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)
    except Exception as e:
        raise StrikeShieldError(
            f"Failed to generate real embeddings using sentence-transformers: {e}",
            context={"model": settings.models.embeddings.model, "n_texts": len(texts)},
        ) from e


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine distance: 1 - cosine_similarity. Clamped to [0.0, 2.0]."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    similarity = max(-1.0, min(1.0, similarity))
    return float(1.0 - similarity)


def mean_pairwise_distance(embeddings: np.ndarray) -> float:
    """Calculates mean pairwise cosine distance across all vector pairs."""
    n = len(embeddings)
    if n <= 1:
        return 0.0

    # Ensure L2 normalized
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms

    sim_matrix = np.dot(normalized, normalized.T)
    # Cosine distance = 1 - sim
    dist_matrix = 1.0 - sim_matrix
    # Extract strictly upper triangle
    upper_indices = np.triu_indices(n, k=1)
    pairwise_distances = dist_matrix[upper_indices]
    return float(np.mean(pairwise_distances))
