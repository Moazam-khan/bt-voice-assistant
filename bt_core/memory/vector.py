"""Semantic memory for BT, using ChromaDB + Ollama embeddings.

ChromaDB's client is synchronous, so its calls run via asyncio.to_thread;
the Ollama embedding call is natively async. Both remember() and recall()
catch their own failures and log them rather than raising, so a memory
outage never breaks a conversation turn — the pipeline just runs without
that turn's context/storage.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import chromadb
import ollama
from pydantic import BaseModel

from bt_core.logging_setup import get_logger

log = get_logger(__name__)

_COLLECTION_NAME = "bt_memory"


class MemoryMatch(BaseModel):
    """One semantically relevant past memory."""

    text: str
    distance: float


class SemanticMemory:
    """Stores and recalls conversation snippets by semantic similarity."""

    def __init__(self, chroma_path: Path, embed_model: str, ollama_host: str) -> None:
        """Initialize the vector store and embedding client.

        Args:
            chroma_path: Directory where ChromaDB persists its data.
            embed_model: Ollama embedding model name (e.g. nomic-embed-text).
            ollama_host: Ollama server URL.
        """
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = self._client.get_or_create_collection(_COLLECTION_NAME)
        self._embed_model = embed_model
        self._ollama = ollama.AsyncClient(host=ollama_host)

    async def _embed(self, text: str) -> list[float]:
        """Get an embedding vector for text via Ollama."""
        response = await self._ollama.embed(model=self._embed_model, input=text)
        return list(response.embeddings[0])

    async def remember(self, text: str) -> None:
        """Store a piece of text for later semantic recall.

        Args:
            text: The text to remember (e.g. a summarized conversation turn).
        """
        try:
            embedding = await self._embed(text)
            await asyncio.to_thread(
                self._collection.add,
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                documents=[text],
            )
            log.info("memory_remembered", text_length=len(text))
        except Exception:
            log.error("memory_remember_failed", exc_info=True)

    async def recall(self, query: str, limit: int = 3) -> list[MemoryMatch]:
        """Find past memories semantically relevant to a query.

        Args:
            query: The text to find related memories for.
            limit: Max number of matches to return.

        Returns:
            Relevant past memories, most similar first. Empty list if
            recall fails or nothing relevant exists yet.
        """
        try:
            embedding = await self._embed(query)
            results = await asyncio.to_thread(
                self._collection.query, query_embeddings=[embedding], n_results=limit
            )
            documents = results.get("documents") or [[]]
            distances = results.get("distances") or [[]]
            return [
                MemoryMatch(text=doc, distance=dist)
                for doc, dist in zip(documents[0], distances[0], strict=True)
            ]
        except Exception:
            log.error("memory_recall_failed", exc_info=True)
            return []
