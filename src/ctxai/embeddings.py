"""
Embeddings generation module using OpenAI's embedding API.
Converts code chunks into vector representations for semantic search.
"""

import os
from typing import List, Optional

from openai import OpenAI


class EmbeddingsGenerator:
    """Generate embeddings for code chunks using OpenAI API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100,
    ):
        """
        Initialize the embeddings generator.

        Args:
            model: OpenAI embedding model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            batch_size: Number of chunks to process in a single API call
        """
        self.model = model
        self.batch_size = batch_size
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=api_key)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches to avoid API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error generating embeddings for batch {i // self.batch_size}: {e}")
                # Return zero vectors for failed batches
                embedding_dim = 1536 if "3-small" in self.model else 3072
                all_embeddings.extend([[0.0] * embedding_dim] * len(batch))

        return all_embeddings

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector as a list of floats
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
