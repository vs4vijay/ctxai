"""Small subprocess driver used to prove Chroma persistence across processes."""

from __future__ import annotations

import sys
from pathlib import Path

from ctxai.chunking import CodeChunker
from ctxai.vector_store import VectorStore


def embedding(text: str) -> list[float]:
    words = ("greet", "calculator", "python", "javascript")
    lowered = text.lower()
    return [float(word in lowered) for word in words]


root, index_path, action = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
store = VectorStore(index_path, "restart-index")
if action == "build":
    chunks = []
    chunker = CodeChunker()
    for source in sorted(item for item in root.iterdir() if item.is_file()):
        chunks.extend(chunker.chunk_file(source))
    store.add_chunks(chunks, [embedding(f"{chunk.language} {chunk.content}") for chunk in chunks])
    print(store.get_stats()["total_chunks"])
else:
    results = store.search(embedding(sys.argv[4]), n_results=1)
    print(results[0]["metadata"]["file_path"])
