"""
LangChain text splitter for climate reports.

Uses RecursiveCharacterTextSplitter — which splits on paragraph breaks,
newlines, and sentences — giving more semantically coherent chunks than
naive fixed-word splitting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..ingestion.document_loader import ClimateDocument
from ..ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class LangChainTextSplitter:
    """
    Wraps LangChain's RecursiveCharacterTextSplitter.

    Returns the same Chunk dataclass as the original TextChunker so
    both splitters are interchangeable in the pipeline.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = None

    def _get_splitter(self):
        if self._splitter is None:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size * 5,       # chars ≈ words * 5
                chunk_overlap=self.chunk_overlap * 5,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        return self._splitter

    def chunk_document(self, doc: ClimateDocument) -> list[Chunk]:
        splitter = self._get_splitter()
        lc_chunks = splitter.split_text(doc.content)

        result = []
        pos = 0
        for idx, text in enumerate(lc_chunks):
            text = text.strip()
            if not text:
                continue
            start = doc.content.find(text[:50], pos)
            start = max(start, pos)
            end = start + len(text)
            pos = max(pos, end - self.chunk_overlap * 5)

            result.append(Chunk(
                chunk_id=f"{doc.doc_id}_lc_{idx:04d}",
                doc_id=doc.doc_id,
                company_name=doc.company_name,
                content=text,
                chunk_index=idx,
                start_char=start,
                end_char=end,
                metadata={
                    "source_path": doc.source_path,
                    "report_year": doc.report_year,
                    "sector": doc.sector,
                    "report_type": doc.report_type,
                    "total_chunks": 0,
                    "splitter": "langchain_recursive",
                },
            ))

        for chunk in result:
            chunk.metadata["total_chunks"] = len(result)

        logger.debug(f"[LC] Split '{doc.doc_id}' into {len(result)} chunks")
        return result

    def chunk_documents(self, docs: list[ClimateDocument]) -> list[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
