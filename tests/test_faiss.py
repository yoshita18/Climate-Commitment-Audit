"""Tests for the FAISS vector store backend."""
from __future__ import annotations

import tempfile
import pytest

from src.rag.embeddings import EmbeddingModel
from src.rag.faiss_store import FaissVectorStore
from src.ingestion.chunker import Chunk


class TestFaissVectorStore:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.emb = EmbeddingModel("all-MiniLM-L6-v2")
        self.store = FaissVectorStore(self.tmp, self.emb)
        self.col = "test_faiss"

    def test_upsert_and_query_returns_results(self):
        ids = ["a", "b", "c"]
        texts = [
            "Global CO2 must fall 43% by 2030 for 1.5°C pathway.",
            "Net-zero by 2050 is required for the Paris Agreement.",
            "Renewable energy must reach 90% of electricity by 2050.",
        ]
        metas = [{"cat": f"cat_{i}"} for i in range(3)]
        count = self.store.upsert_texts(ids, texts, metas, self.col)
        assert count == 3

        results = self.store.query("emission reduction", self.col, top_k=2)
        assert len(results) == 2
        assert "content" in results[0]
        assert "similarity" in results[0]
        assert results[0]["similarity"] > 0

    def test_collection_count(self):
        self.store.upsert_texts(["x", "y"], ["text x", "text y"], [{}, {}], self.col + "_cnt")
        assert self.store.collection_count(self.col + "_cnt") == 2

    def test_upsert_chunks(self):
        chunk = Chunk(
            chunk_id="faiss_chunk_001",
            doc_id="doc1",
            company_name="TestCo",
            content="We target net-zero by 2050.",
            chunk_index=0,
            start_char=0,
            end_char=50,
            metadata={"report_year": 2024, "sector": "energy", "source_path": "/t",
                      "report_type": "sr", "total_chunks": 1},
        )
        count = self.store.upsert_chunks([chunk], self.col + "_chunks")
        assert count == 1

    def test_results_are_sorted_by_similarity(self):
        self.store.upsert_texts(
            ["i1", "i2", "i3"],
            [
                "Carbon capture and storage technology reduces industrial emissions.",
                "Renewable solar energy reduces electricity sector emissions.",
                "Employee wellbeing programmes improve productivity.",
            ],
            [{}, {}, {}],
            self.col + "_sorted",
        )
        results = self.store.query("solar renewable energy", self.col + "_sorted", top_k=3)
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_empty_collection_returns_empty(self):
        results = self.store.query("anything", self.col + "_empty", top_k=5)
        assert results == []

    def test_delete_collection(self):
        self.store.upsert_texts(["d1"], ["some text"], [{}], self.col + "_del")
        assert self.store.collection_count(self.col + "_del") == 1
        self.store.delete_collection(self.col + "_del")
        assert self.store.collection_count(self.col + "_del") == 0

    def test_persistence(self):
        """Reload the store from disk and check data is preserved."""
        self.store.upsert_texts(["p1"], ["persistent text about climate"], [{}], self.col + "_p")
        # Create new store instance pointing to same dir
        store2 = FaissVectorStore(self.tmp, self.emb)
        count = store2.collection_count(self.col + "_p")
        assert count == 1
