"""
Mock embedding provider for fast, deterministic testing.

This mock provider generates embeddings without loading heavy ML models,
making tests 100x faster while maintaining the same interface.
"""

import hashlib

import numpy as np

from ctxai.config import EmbeddingConfig
from ctxai.embeddings import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Fast mock embeddings using deterministic hashing.

    Uses MD5 hash of text content as seed for reproducible random vectors.
    This ensures the same text always produces the same embedding without
    needing to load a real ML model.
    """

    def __init__(self, config: EmbeddingConfig = None, dimension: int = 384):
        """
        Initialize mock embedding provider.

        Args:
            config: Embedding configuration (can be None for testing)
            dimension: Embedding vector dimension (default: 384, same as all-MiniLM-L6-v2)
        """
        # Create dummy config if none provided
        if config is None:
            config = EmbeddingConfig(
                provider="mock",
                model="mock-model",
                batch_size=32
            )

        super().__init__(config)
        self._dimension = dimension

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate deterministic mock embeddings.

        Uses MD5 hash of text as seed for numpy random number generator,
        ensuring the same text always produces the same embedding.

        Args:
            texts: List of text strings to embed

        Returns:
            List of normalized embedding vectors
        """
        if not texts:
            return []

        embeddings = []
        for text in texts:
            # Use hash of text as seed for reproducibility
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)

            # Generate random vector
            embedding = rng.rand(self._dimension).astype(float)

            # Normalize to unit vector (like real embeddings)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            embeddings.append(embedding.tolist())

        return embeddings

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding vector dimension
        """
        return self._dimension
