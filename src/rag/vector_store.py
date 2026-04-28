from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from ..ingestion.chunker import Chunk
from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store for climate report chunks and IPCC benchmarks."""

    def __init__(self, persist_dir: str, embedding_model: EmbeddingModel):
        self.persist_dir = Path(persist_dir)
        self.embedding_model = embedding_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import chromadb

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def get_or_create_collection(self, name: str):
        client = self._get_client()
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[Chunk], collection_name: str) -> int:
        if not chunks:
            return 0

        collection = self.get_or_create_collection(collection_name)
        texts = [c.content for c in chunks]
        embeddings = self.embedding_model.embed(texts, batch_size=32)

        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "company_name": c.company_name,
                    "chunk_index": c.chunk_index,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in chunks
            ],
        )
        logger.info(f"Upserted {len(chunks)} chunks into '{collection_name}'")
        return len(chunks)

    def upsert_texts(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
        collection_name: str,
    ) -> int:
        collection = self.get_or_create_collection(collection_name)
        embeddings = self.embedding_model.embed(texts, batch_size=32)
        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(ids)} texts into '{collection_name}'")
        return len(ids)

    def query(
        self,
        query_text: str,
        collection_name: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        collection = self.get_or_create_collection(collection_name)
        query_embedding = self.embedding_model.embed_single(query_text)

        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "content": doc,
                    "metadata": meta,
                    "similarity": float(1 - dist),  # cosine distance → similarity
                }
            )
        return hits

    def delete_collection(self, collection_name: str):
        client = self._get_client()
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception:
            pass

    def collection_count(self, collection_name: str) -> int:
        collection = self.get_or_create_collection(collection_name)
        return collection.count()
