"""Wrapper around ChromaDB embeddings platform."""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import xor_args
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    import chromadb
    import chromadb.config

logger = logging.getLogger()
DEFAULT_K = 4  # Number of Documents to return.


def _results_to_docs(results: Any) -> list[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> list[tuple[Document, float]]:
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


class Chroma(VectorStore):
    """Wrapper around ChromaDB embeddings platform.

    To use, you should have the ``chromadb`` python package installed.

    Example:
        .. code-block:: python

                from langchain.vectorstores import Chroma
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = Chroma("langchain_store", embeddings.embed_query)
    """

    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Embeddings | None = None,
        persist_directory: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        collection_metadata: dict | None = None,
        client: chromadb.Client | None = None,
    ) -> None:
        """Initialize with Chroma client."""
        try:
            import chromadb
            import chromadb.config
        except ImportError:
            raise ValueError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )

        if client is not None:
            self._client = client
        else:
            if client_settings:
                self._client_settings = client_settings
            else:
                self._client_settings = chromadb.config.Settings()
                if persist_directory is not None:
                    self._client_settings = chromadb.config.Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=persist_directory,
                    )
            self._client = chromadb.Client(self._client_settings)

        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function.embed_documents
            if self._embedding_function is not None
            else None,
            metadata=collection_metadata,
        )

    @xor_args(("query_texts", "query_embeddings"))
    def __query_collection(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 4,
        where: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Query the chroma collection."""
        try:
            import chromadb
        except ImportError:
            raise ValueError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )

        for i in range(n_results, 0, -1):
            try:
                return self._collection.query(
                    query_texts=query_texts,
                    query_embeddings=query_embeddings,
                    n_results=i,
                    where=where,
                    **kwargs,
                )
            except chromadb.errors.NotEnoughElementsException:
                logger.error(
                    f"Chroma collection {self._collection.name} "
                    f"contains fewer than {i} elements."
                )
        raise chromadb.errors.NotEnoughElementsException(
            f"No documents found for Chroma collection {self._collection.name}"
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(list(texts))
        self._collection.add(
            metadatas=metadatas, embeddings=embeddings, documents=texts, ids=ids
        )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Run similarity search with Chroma.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding (str): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            List of Documents most similar to the query vector.
        """
        results = self.__query_collection(
            query_embeddings=embedding, n_results=k, where=filter
        )
        return _results_to_docs(results)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query], n_results=k, where=filter
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding], n_results=k, where=filter
            )

        return _results_to_docs_and_scores(results)

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return self.similarity_search_with_score(query, k)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        results = self.__query_collection(
            query_embeddings=embedding,
            n_results=fetch_k,
            where=filter,
            include=["metadatas", "documents", "distances", "embeddings"],
        )
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            results["embeddings"][0],
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = _results_to_docs(results)

        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        if self._embedding_function is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function on" "creation."
            )

        embedding = self._embedding_function.embed_query(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mul=lambda_mult, filter=filter
        )
        return docs

    def delete_collection(self) -> None:
        """Delete the collection."""
        self._client.delete_collection(self._collection.name)

    def get(self, include: list[str] | None = None) -> dict[str, Any]:
        if include is not None:
            return self._collection.get(include=include)
        else:
            return self._collection.get()

    def persist(self) -> None:
        if self._persist_directory is None:
            raise ValueError(
                "You must specify a persist_directory on"
                "creation to persist the collection."
            )
        self._client.persist()

    def update_document(self, document_id: str, document: Document) -> None:
        text = document.page_content
        metadata = document.metadata
        if self._embedding_function is None:
            raise ValueError(
                "For update, you must specify an embedding function on creation."
            )
        embeddings = self._embedding_function.embed_documents([text])

        self._collection.update(
            ids=[document_id],
            embeddings=embeddings,
            documents=[text],
            metadatas=[metadata],
        )

    @classmethod
    def from_texts(
        cls: type[Chroma],
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        client: chromadb.Client | None = None,
        **kwargs: Any,
    ) -> Chroma:
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
        )
        chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return chroma_collection

    @classmethod
    def from_documents(
        cls: type[Chroma],
        documents: list[Document],
        embedding: Embeddings | None = None,
        ids: list[str] | None = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        client: chromadb.Client | None = None,  # Add this line
        **kwargs: Any,
    ) -> Chroma:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
        )
