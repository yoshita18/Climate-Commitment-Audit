"""
FAISS-based vector store — pure faiss-cpu (no LangChain dependency).

Drop-in alternative to VectorStore (ChromaDB).  Persists index + metadata
as a pair of files:  <persist_dir>/<collection>.index  and  .meta.pkl

Install:  pip install faiss-cpu
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from ..ingestion.chunker import Chunk
from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """
    In-process FAISS flat-L2 (cosine via normalization) vector store.

    Mirrors the VectorStore (ChromaDB) interface so either backend can be
    swapped transparently in the pipeline.
    """

    def __init__(self, persist_dir: str, embedding_model: EmbeddingModel):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self._indices: dict = {}      # collection → faiss.IndexFlatIP
        self._meta: dict = {}         # collection → list[dict]
        self._docs: dict = {}         # collection → list[str]

    # ──────────────────────────────── persistence ────────────────────────

    def _index_path(self, collection: str) -> Path:
        return self.persist_dir / f"{collection}.index"

    def _meta_path(self, collection: str) -> Path:
        return self.persist_dir / f"{collection}.meta.pkl"

    def _load_collection(self, collection: str):
        if collection in self._indices:
            return
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")

        ip = self._index_path(collection)
        mp = self._meta_path(collection)
        if ip.exists() and mp.exists():
            self._indices[collection] = faiss.read_index(str(ip))
            with open(mp, "rb") as f:
                saved = pickle.load(f)
            self._meta[collection] = saved["meta"]
            self._docs[collection] = saved["docs"]
            logger.debug(f"Loaded FAISS collection '{collection}' ({len(self._docs[collection])} docs)")
        else:
            dim = self.embedding_model.dimension
            self._indices[collection] = faiss.IndexFlatIP(dim)
            self._meta[collection] = []
            self._docs[collection] = []

    def _save_collection(self, collection: str):
        import faiss
        faiss.write_index(self._indices[collection], str(self._index_path(collection)))
        with open(self._meta_path(collection), "wb") as f:
            pickle.dump({"meta": self._meta[collection], "docs": self._docs[collection]}, f)

    # ──────────────────────────────── write ──────────────────────────────

    def upsert_texts(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
        collection_name: str,
    ) -> int:
        self._load_collection(collection_name)
        embeddings = self.embedding_model.embed(texts, batch_size=32)
        # Normalise to unit length → inner-product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

        self._indices[collection_name].add(embeddings.astype(np.float32))
        self._docs[collection_name].extend(texts)
        self._meta[collection_name].extend(
            [{**m, "_id": i} for i, m in zip(ids, metadatas)]
        )
        self._save_collection(collection_name)
        logger.info(f"[FAISS] Upserted {len(ids)} texts into '{collection_name}'")
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

    # ──────────────────────────────── query ──────────────────────────────

    def query(
        self,
        query_text: str,
        collection_name: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        self._load_collection(collection_name)
        total = len(self._docs[collection_name])
        if total == 0:
            return []

        q_vec = np.array([self.embedding_model.embed_single(query_text)], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        k = min(top_k * 3 if where else top_k, total)
        scores, idxs = self._indices[collection_name].search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            meta = self._meta[collection_name][idx]
            # Optional metadata filter
            if where and not all(meta.get(fk) == fv for fk, fv in where.items()):
                continue
            results.append({
                "content": self._docs[collection_name][idx],
                "metadata": meta,
                "similarity": float(score),
            })
            if len(results) >= top_k:
                break
        return results

    def collection_count(self, collection_name: str) -> int:
        self._load_collection(collection_name)
        return len(self._docs.get(collection_name, []))

    def delete_collection(self, collection_name: str):
        self._indices.pop(collection_name, None)
        self._meta.pop(collection_name, None)
        self._docs.pop(collection_name, None)
        for path in [self._index_path(collection_name), self._meta_path(collection_name)]:
            path.unlink(missing_ok=True)
        logger.info(f"[FAISS] Deleted collection: {collection_name}")
