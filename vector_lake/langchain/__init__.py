"""Wrapper around VectorLakeStore embeddings."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

from vector_lake import VectorLake

logger = logging.getLogger()
DEFAULT_K = 4


class VectorLakeStore(VectorStore):
    """Interface for vector lake stores."""

    def __init__(
        self,
        location: str = ".vector_lake",
        dimension: int = 384,
        embedding: Embeddings | None = None,
        **opts: Any,
    ) -> None:
        """Initialize with VectorLake client."""

        defaults = {"approx_shards": 10}
        opts = {**defaults, **opts}
        self._client = VectorLake(location=location, dimension=dimension, **opts)
        self._embedding_function = embedding
        if not self._embedding_function:
            from langchain.embeddings.sentence_transformer import (
                SentenceTransformerEmbeddings,
            )

            self._embedding_function = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        ids = []
        embeddings = self._embedding_function.embed_documents(list(texts))
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            metadata = metadatas[idx] if metadatas else {}
            uid = self._client.add(embedding, metadata, text)
            ids.append(uid)
        self._client.persist()
        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the vectorstore."""
        await asyncio.to_thread(self.add_texts, texts, metadatas, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Return docs most similar to query."""
        embedding = self._embedding_function.embed_query(query)
        docs = self._client.query(embedding, n_results=k)
        return [
            Document(
                page_content=row["document"],
                metadata=row["metadata"],
            )
            for row in docs
        ]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> VectorLakeStore:
        """Return VectorStore initialized from texts and embeddings."""
        instance = cls(embedding=embedding, **kwargs)
        instance.add_texts(texts, metadatas=metadatas, **kwargs)
        return instance

    @classmethod
    async def afrom_texts(
        cls: type[VectorLakeStore],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> VectorLakeStore:
        """Return VectorStore initialized from texts and embeddings."""
        asyncio.to_thread(cls.from_texts, texts, embedding, metadatas, **kwargs)
