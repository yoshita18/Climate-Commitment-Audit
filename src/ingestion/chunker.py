from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

from .document_loader import ClimateDocument


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    company_name: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict


class TextChunker:
    """Splits climate documents into overlapping chunks for RAG indexing."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: ClimateDocument) -> list[Chunk]:
        sentences = self._split_sentences(doc.content)
        chunks = self._merge_sentences_into_chunks(sentences)

        result = []
        pos = 0
        for idx, text in enumerate(chunks):
            start = doc.content.find(text, pos)
            if start == -1:
                start = pos
            end = start + len(text)
            pos = max(pos, end - self.chunk_overlap * 4)

            result.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_chunk_{idx:04d}",
                    doc_id=doc.doc_id,
                    company_name=doc.company_name,
                    content=text.strip(),
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "source_path": doc.source_path,
                        "report_year": doc.report_year,
                        "sector": doc.sector,
                        "report_type": doc.report_type,
                        "total_chunks": 0,  # filled below
                    },
                )
            )

        for chunk in result:
            chunk.metadata["total_chunks"] = len(result)

        return result

    def chunk_documents(self, docs: list[ClimateDocument]) -> list[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _split_sentences(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_sentences_into_chunks(self, sentences: list[str]) -> list[str]:
        chunks = []
        current_words: list[str] = []
        current_len = 0

        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)

            if current_len + word_count > self.chunk_size and current_words:
                chunks.append(" ".join(current_words))
                # keep overlap
                overlap_words = current_words[-self.chunk_overlap:] if self.chunk_overlap else []
                current_words = overlap_words + words
                current_len = len(current_words)
            else:
                current_words.extend(words)
                current_len += word_count

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks
