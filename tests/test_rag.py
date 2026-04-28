"""Tests for the RAG pipeline components."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.ingestion.chunker import Chunk


class TestEmbeddingModel:
    def setup_method(self):
        self.model = EmbeddingModel("all-MiniLM-L6-v2")

    def test_embed_single_string(self):
        vec = self.model.embed("Global CO2 emissions must fall 43% by 2030")
        assert vec.shape == (1, self.model.dimension)

    def test_embed_batch(self):
        texts = [
            "Net-zero by 2050 is the IPCC 1.5°C target.",
            "Renewable energy must supply 90% of electricity.",
            "Methane reduction of 34% required by 2030.",
        ]
        vecs = self.model.embed(texts)
        assert vecs.shape == (3, self.model.dimension)

    def test_embed_single_returns_list(self):
        vec = self.model.embed_single("test sentence")
        assert isinstance(vec, list)
        assert len(vec) == self.model.dimension

    def test_embeddings_normalized(self):
        import numpy as np
        vec = self.model.embed("test sentence")
        norm = np.linalg.norm(vec[0])
        assert abs(norm - 1.0) < 1e-5


class TestVectorStore:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.embedding_model = EmbeddingModel("all-MiniLM-L6-v2")
        self.store = VectorStore(self.tmp_dir, self.embedding_model)
        self.collection = "test_collection"

    def test_upsert_and_query(self):
        ids = ["doc1", "doc2", "doc3"]
        texts = [
            "CO2 must fall 43% by 2030 for 1.5°C pathway.",
            "Net-zero by 2050 is required for Paris Agreement.",
            "Renewable energy must reach 90% of electricity by 2050.",
        ]
        metadatas = [{"category": f"cat_{i}"} for i in range(3)]

        count = self.store.upsert_texts(ids, texts, metadatas, self.collection)
        assert count == 3

        results = self.store.query("emission reduction target", self.collection, top_k=2)
        assert len(results) == 2
        assert "content" in results[0]
        assert "similarity" in results[0]
        assert 0.0 <= results[0]["similarity"] <= 1.0

    def test_collection_count(self):
        self.store.upsert_texts(
            ["a", "b"],
            ["text a", "text b"],
            [{"source": "test"}, {"source": "test"}],
            self.collection + "_count",
        )
        count = self.store.collection_count(self.collection + "_count")
        assert count == 2

    def test_upsert_chunks(self):
        chunks = [
            Chunk(
                chunk_id="test_chunk_0001",
                doc_id="test_doc",
                company_name="Test Corp",
                content="We commit to net-zero by 2050.",
                chunk_index=0,
                start_char=0,
                end_char=100,
                metadata={"report_year": 2024, "sector": "energy", "source_path": "/test", "report_type": "sr", "total_chunks": 1},
            )
        ]
        count = self.store.upsert_chunks(chunks, self.collection + "_chunks")
        assert count == 1

    def test_empty_query_returns_results(self):
        self.store.upsert_texts(["x"], ["some content"], [{"source": "test"}], self.collection + "_empty")
        results = self.store.query("anything", self.collection + "_empty", top_k=1)
        assert len(results) == 1


class TestRetriever:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.embedding_model = EmbeddingModel("all-MiniLM-L6-v2")
        self.store = VectorStore(self.tmp_dir, self.embedding_model)
        self.retriever = Retriever(self.store, top_k=3)

    def test_format_context_non_empty(self):
        hits = [
            {"content": "IPCC says 43% reduction by 2030.", "similarity": 0.9, "metadata": {"category": "global"}},
            {"content": "Net-zero by 2050.", "similarity": 0.8, "metadata": {"category": "global"}},
        ]
        context = self.retriever.format_context(hits)
        assert "43%" in context
        assert "0.9" in context

    def test_format_context_respects_max_chars(self):
        hits = [{"content": "x" * 500, "similarity": 0.9, "metadata": {}} for _ in range(10)]
        context = self.retriever.format_context(hits, max_chars=200)
        assert len(context) <= 400  # some buffer for headers
