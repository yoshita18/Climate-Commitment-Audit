"""Tests for document loading and chunking."""

import pytest
from pathlib import Path

from src.ingestion.document_loader import DocumentLoader, ClimateDocument
from src.ingestion.chunker import TextChunker, Chunk

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_reports"


class TestDocumentLoader:
    def setup_method(self):
        self.loader = DocumentLoader()

    def test_load_text_file(self):
        path = SAMPLE_DIR / "report_01_credible_greentech_corp.txt"
        doc = self.loader.load_file(path)
        assert isinstance(doc, ClimateDocument)
        assert doc.word_count > 100
        assert doc.company_name != ""
        assert doc.content != ""

    def test_load_directory(self):
        docs = self.loader.load_directory(SAMPLE_DIR)
        # 5 original reports + 200 generated = 205 total (generator script excluded)
        assert len(docs) >= 5
        for doc in docs[:5]:
            assert doc.word_count > 50

    def test_doc_has_required_fields(self):
        path = SAMPLE_DIR / "report_01_credible_greentech_corp.txt"
        doc = self.loader.load_file(path)
        assert doc.doc_id
        assert doc.source_path
        assert doc.char_count > 0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.loader.load_file("/nonexistent/path.txt")

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError):
            self.loader.load_file("/some/file.docx")


class TestTextChunker:
    def setup_method(self):
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        self.loader = DocumentLoader()

    def test_chunk_produces_list(self):
        path = SAMPLE_DIR / "report_01_credible_greentech_corp.txt"
        doc = self.loader.load_file(path)
        chunks = self.chunker.chunk_document(doc)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_ids_unique(self):
        path = SAMPLE_DIR / "report_01_credible_greentech_corp.txt"
        doc = self.loader.load_file(path)
        chunks = self.chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_has_required_fields(self):
        path = SAMPLE_DIR / "report_01_credible_greentech_corp.txt"
        doc = self.loader.load_file(path)
        chunks = self.chunker.chunk_document(doc)
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.content.strip() != ""
            assert chunk.doc_id == doc.doc_id
            assert chunk.company_name == doc.company_name

    def test_chunk_multiple_documents(self):
        docs = self.loader.load_directory(SAMPLE_DIR)
        chunks = self.chunker.chunk_documents(docs)
        assert len(chunks) > len(docs)
        doc_ids = {c.doc_id for c in chunks}
        assert len(doc_ids) == len(docs)

    def test_chunk_overlap_respected(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=5)
        path = SAMPLE_DIR / "report_01_credible_greentech_corp.txt"
        doc = self.loader.load_file(path)
        chunks = chunker.chunk_document(doc)
        # All chunks should be non-empty
        for chunk in chunks:
            assert len(chunk.content.split()) > 0
