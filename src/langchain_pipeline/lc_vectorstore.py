"""
LangChain + FAISS vector store.

Uses langchain_community.vectorstores.FAISS with HuggingFace embeddings
(sentence-transformers/all-MiniLM-L6-v2) — fully local, no API key needed.

Provides a query() method that returns the same dict format as VectorStore
and FaissVectorStore so the Retriever works with any backend.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class LangChainFaissStore:
    """
    LangChain FAISS vector store backed by HuggingFace sentence-transformers.

    Persists to <persist_dir>/<collection>/ directory.
    """

    def __init__(self, persist_dir: str, model_name: str = "all-MiniLM-L6-v2"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._stores: dict = {}     # collection → FAISS store
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def _collection_dir(self, name: str) -> Path:
        return self.persist_dir / f"lc_{name}"

    def _load_or_create(self, collection: str):
        if collection in self._stores:
            return
        from langchain_community.vectorstores import FAISS
        col_dir = self._collection_dir(collection)
        emb = self._get_embeddings()
        if col_dir.exists():
            try:
                self._stores[collection] = FAISS.load_local(
                    str(col_dir), emb, allow_dangerous_deserialization=True
                )
                logger.debug(f"[LC-FAISS] Loaded '{collection}' from disk")
                return
            except Exception:
                pass
        # Create empty store (can't create truly empty; will add docs on first upsert)
        self._stores[collection] = None

    # ──────────────────────────────────────── write ──────────────────────

    def upsert_texts(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
        collection_name: str,
    ) -> int:
        if not texts:
            return 0
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document as LCDoc

        self._load_or_create(collection_name)
        emb = self._get_embeddings()
        lc_docs = [
            LCDoc(page_content=t, metadata={**m, "_lc_id": i})
            for t, m, i in zip(texts, metadatas, ids)
        ]

        if self._stores[collection_name] is None:
            store = FAISS.from_documents(lc_docs, emb)
        else:
            store = self._stores[collection_name]
            store.add_documents(lc_docs)

        self._stores[collection_name] = store
        col_dir = self._collection_dir(collection_name)
        col_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(col_dir))
        logger.info(f"[LC-FAISS] Upserted {len(ids)} docs into '{collection_name}'")
        return len(ids)

    def upsert_chunks(self, chunks: list[Chunk], collection_name: str) -> int:
        ids = [c.chunk_id for c in chunks]
        texts = [c.content for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "company_name": c.company_name,
                "chunk_index": str(c.chunk_index),
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]
        return self.upsert_texts(ids, texts, metadatas, collection_name)

    # ─────────────────────────────────────── query ───────────────────────

    def query(
        self,
        query_text: str,
        collection_name: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        self._load_or_create(collection_name)
        store = self._stores.get(collection_name)
        if store is None:
            return []

        fetch_k = top_k * 3 if where else top_k
        results_with_scores = store.similarity_search_with_score(query_text, k=fetch_k)

        hits = []
        for doc, dist in results_with_scores:
            meta = doc.metadata
            if where and not all(meta.get(fk) == fv for fk, fv in where.items()):
                continue
            # FAISS returns L2 distance; convert to similarity (1 / (1 + dist))
            similarity = float(1.0 / (1.0 + dist))
            hits.append({
                "content": doc.page_content,
                "metadata": meta,
                "similarity": similarity,
            })
            if len(hits) >= top_k:
                break
        return hits

    def collection_count(self, collection_name: str) -> int:
        self._load_or_create(collection_name)
        store = self._stores.get(collection_name)
        if store is None:
            return 0
        try:
            return store.index.ntotal
        except Exception:
            return 0

    def delete_collection(self, collection_name: str):
        self._stores.pop(collection_name, None)
        import shutil
        col_dir = self._collection_dir(collection_name)
        if col_dir.exists():
            shutil.rmtree(col_dir)
        logger.info(f"[LC-FAISS] Deleted collection: {collection_name}")
