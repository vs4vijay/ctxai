"""
Embeddings generation module with support for multiple providers.
Converts code chunks into vector representations for semantic search.

Supported providers:
- local: sentence-transformers (default, no API key needed)
- openai: OpenAI embeddings API
- huggingface: HuggingFace Inference API
- azure: Azure OpenAI
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional

from .config import EmbeddingConfig


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the provider.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self.batch_size = config.batch_size

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        pass

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local embeddings using sentence-transformers (default)."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize local embedding provider."""
        super().__init__(config)
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            )
        
        # Use a good default model for code
        model_name = config.model or "all-MiniLM-L6-v2"
        print(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating local embeddings: {e}")
            return [[0.0] * self._dimension] * len(texts)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embeddings provider."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize OpenAI embedding provider."""
        super().__init__(config)
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install it with: pip install openai"
            )
        
        # Get API key from config or environment
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or configure it in .ctxai/config.json"
            )
        
        self.model = config.model or "text-embedding-3-small"
        self.client = OpenAI(api_key=api_key)
        
        # Determine dimension based on model
        if "3-small" in self.model:
            self._dimension = 1536
        elif "3-large" in self.model:
            self._dimension = 3072
        elif "ada-002" in self.model:
            self._dimension = 1536
        else:
            self._dimension = 1536  # Default

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating OpenAI embeddings for batch {i // self.batch_size}: {e}")
                all_embeddings.extend([[0.0] * self._dimension] * len(batch))

        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace Inference API embeddings provider."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize HuggingFace embedding provider."""
        super().__init__(config)
        
        # Get API key from config or environment
        api_key = config.api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError(
                "HuggingFace API key is required. Set HUGGINGFACE_API_KEY environment variable "
                "or configure it in .ctxai/config.json"
            )
        
        self.model = config.model or "sentence-transformers/all-MiniLM-L6-v2"
        self.api_key = api_key
        self._dimension = 384  # Default for all-MiniLM-L6-v2

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using HuggingFace API."""
        import requests
        
        if not texts:
            return []

        all_embeddings = []
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            
            try:
                response = requests.post(api_url, headers=headers, json={"inputs": batch})
                response.raise_for_status()
                batch_embeddings = response.json()
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating HuggingFace embeddings for batch {i // self.batch_size}: {e}")
                all_embeddings.extend([[0.0] * self._dimension] * len(batch))

        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


class EmbeddingsFactory:
    """Factory for creating embedding providers."""

    _providers = {
        "local": LocalEmbeddingProvider,
        "openai": OpenAIEmbeddingProvider,
        "huggingface": HuggingFaceEmbeddingProvider,
    }

    @classmethod
    def create(cls, config: EmbeddingConfig) -> BaseEmbeddingProvider:
        """
        Create an embedding provider based on configuration.

        Args:
            config: Embedding configuration

        Returns:
            Embedding provider instance

        Raises:
            ValueError: If provider is not supported
        """
        provider_class = cls._providers.get(config.provider)
        if provider_class is None:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown embedding provider: {config.provider}. "
                f"Available providers: {available}"
            )
        
        return provider_class(config)

    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """
        Register a custom embedding provider.

        Args:
            name: Provider name
            provider_class: Provider class (must inherit from BaseEmbeddingProvider)
        """
        if not issubclass(provider_class, BaseEmbeddingProvider):
            raise ValueError("Provider class must inherit from BaseEmbeddingProvider")
        cls._providers[name] = provider_class


# Backwards compatibility alias
EmbeddingsGenerator = BaseEmbeddingProvider
