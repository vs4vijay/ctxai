"""
Vector database storage module using ChromaDB.
Stores and retrieves code embeddings for semantic search.
"""

import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .chunking import CodeChunk


class VectorStore:
    """Vector database for storing and querying code embeddings."""

    def __init__(self, storage_path: Path, collection_name: str):
        """
        Initialize the vector store.

        Args:
            storage_path: Path to store the ChromaDB database
            collection_name: Name of the collection (index name)
        """
        # The canonical layout is <indexes_dir>/<collection_name>.  Accepting
        # the parent directory here preserves the public constructor used by
        # older callers without opening a second, empty database.
        storage_path = Path(storage_path)
        if storage_path.name != collection_name:
            storage_path = storage_path / collection_name
        self.storage_path = storage_path
        self.collection_name = collection_name

        # Create storage directory if it doesn't exist
        storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(storage_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Code embeddings for {collection_name}"},
        )

    def add_chunks(
        self,
        chunks: list[CodeChunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> None:
        """
        Add code chunks with their embeddings to the vector store.

        Args:
            chunks: List of CodeChunk objects
            embeddings: List of embedding vectors
            batch_size: Number of chunks to add in a single batch
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            # Prepare data for ChromaDB
            ids = [self._generate_chunk_id(chunk) for chunk in batch_chunks]
            documents = [chunk.content for chunk in batch_chunks]
            metadatas = [self._chunk_to_metadata(chunk) for chunk in batch_chunks]

            try:
                self.collection.upsert(
                    ids=ids,
                    embeddings=batch_embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            except Exception as exc:
                raise VectorStoreWriteError(f"Failed to store batch {i // batch_size}: {exc}") from exc

    def delete_files(self, file_paths: list[str]) -> None:
        """Delete all chunks belonging to the supplied canonical paths."""
        for file_path in file_paths:
            try:
                self.collection.delete(where={"file_path": file_path})
            except Exception as exc:
                raise VectorStoreWriteError(f"Failed to delete stale chunks for {file_path}: {exc}") from exc

    def clear(self) -> None:
        """Remove all chunks while keeping the collection available."""
        try:
            existing = self.collection.get(include=[])
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        except Exception as exc:
            raise VectorStoreWriteError(f"Failed to clear index '{self.collection_name}': {exc}") from exc

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """
        Search for similar code chunks.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of dictionaries containing chunk information and similarity scores
        """
        try:
            count = self.collection.count()
            if count == 0:
                return []
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, count),
                where=filter_dict,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                        }
                    )

            return formatted_results

        except Exception as exc:
            raise VectorStoreQueryError(f"Failed to search index '{self.collection_name}': {exc}") from exc

    def get_chunks(self) -> list[dict]:
        """Return all stored chunks for local lexical and symbol retrieval."""
        try:
            results = self.collection.get(include=["documents", "metadatas"])
            return [
                {"id": chunk_id, "content": document, "metadata": metadata}
                for chunk_id, document, metadata in zip(
                    results.get("ids", []),
                    results.get("documents", []),
                    results.get("metadatas", []),
                )
            ]
        except Exception as exc:
            raise VectorStoreQueryError(f"Failed to read index '{self.collection_name}': {exc}") from exc

    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with stats like total chunks, unique files, etc.
        """
        try:
            count = self.collection.count()

            # Get sample of metadata to analyze
            sample_size = min(count, 1000)
            results = self.collection.get(limit=sample_size, include=["metadatas"])

            unique_files = set()
            languages = {}

            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    file_path = metadata.get("file_path", "")
                    if file_path:
                        unique_files.add(file_path)

                    language = metadata.get("language", "unknown")
                    languages[language] = languages.get(language, 0) + 1

            return {
                "total_chunks": count,
                "unique_files": len(unique_files),
                "total_files": len(unique_files),
                "languages": languages,
                "collection_name": self.collection_name,
            }
        except Exception as exc:
            raise VectorStoreError(f"Failed to read index statistics: {exc}") from exc

    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as exc:
            raise VectorStoreWriteError(f"Failed to delete index '{self.collection_name}': {exc}") from exc

    def _generate_chunk_id(self, chunk: CodeChunk) -> str:
        """
        Generate a unique ID for a chunk.

        Args:
            chunk: CodeChunk object
        Returns:
            Unique string ID
        """
        identity = "\0".join(
            (str(chunk.file_path.resolve()), str(chunk.start_line), str(chunk.end_line), chunk.content)
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _chunk_to_metadata(self, chunk: CodeChunk) -> dict[str, str]:
        """
        Convert CodeChunk to metadata dictionary.

        Args:
            chunk: CodeChunk object

        Returns:
            Metadata dictionary
        """
        metadata = {
            "file_path": str(chunk.file_path),
            "start_line": str(chunk.start_line),
            "end_line": str(chunk.end_line),
            "chunk_type": chunk.chunk_type,
            "language": chunk.language,
        }

        # Add additional metadata from chunk
        for key, value in chunk.metadata.items():
            metadata[f"meta_{key}"] = str(value)

        return metadata


class VectorStoreError(RuntimeError):
    """Base class for persistent index failures."""


class VectorStoreWriteError(VectorStoreError):
    """Raised when a vector-store mutation fails."""


class VectorStoreQueryError(VectorStoreError):
    """Raised when a vector-store query fails."""
