from __future__ import annotations
import logging
from typing import Optional

from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """High-level retrieval interface for the auditing pipeline."""

    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve_benchmarks(self, claim: str, top_k: Optional[int] = None) -> list[dict]:
        """Retrieve IPCC benchmark passages relevant to a climate claim."""
        from config import get_settings
        settings = get_settings()
        return self.vector_store.query(
            query_text=claim,
            collection_name=settings.benchmarks_collection,
            top_k=top_k or self.top_k,
        )

    def retrieve_report_context(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve passages from climate reports, optionally filtered to one company."""
        from config import get_settings
        settings = get_settings()
        where = {"doc_id": doc_id} if doc_id else None
        return self.vector_store.query(
            query_text=query,
            collection_name=settings.reports_collection,
            top_k=top_k or self.top_k,
            where=where,
        )

    def retrieve_cross_company(self, claim: str, top_k: Optional[int] = None) -> list[dict]:
        """Retrieve similar claims from other companies' reports for comparison."""
        from config import get_settings
        settings = get_settings()
        return self.vector_store.query(
            query_text=claim,
            collection_name=settings.reports_collection,
            top_k=top_k or self.top_k * 2,
        )

    def format_context(self, hits: list[dict], max_chars: int = 3000) -> str:
        parts = []
        total = 0
        for i, hit in enumerate(hits, 1):
            snippet = hit["content"].strip()
            sim = hit["similarity"]
            meta = hit.get("metadata", {})
            source = meta.get("company_name", meta.get("category", "Unknown"))
            header = f"[{i}] Source: {source} (relevance: {sim:.2f})"
            entry = f"{header}\n{snippet}"
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n".join(parts)
