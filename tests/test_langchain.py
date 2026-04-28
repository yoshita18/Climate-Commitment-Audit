"""Tests for LangChain pipeline components (no LLM calls needed)."""
from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.langchain_pipeline.lc_loader import LangChainDocumentLoader
from src.langchain_pipeline.lc_splitter import LangChainTextSplitter
from src.ingestion.document_loader import ClimateDocument

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_reports"
SAMPLE_REPORT = SAMPLE_DIR / "report_01_credible_greentech_corp.txt"


class TestLangChainDocumentLoader:
    def setup_method(self):
        self.loader = LangChainDocumentLoader()

    def test_load_file_returns_climate_document(self):
        doc = self.loader.load_file(SAMPLE_REPORT)
        assert isinstance(doc, ClimateDocument)
        assert len(doc.content) > 100
        assert doc.doc_id != ""

    def test_load_file_unsupported_extension_raises(self):
        with pytest.raises(ValueError):
            self.loader.load_file("/some/file.docx")

    def test_load_file_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            self.loader.load_file("/nonexistent/file.txt")

    def test_load_directory(self):
        docs = self.loader.load_directory(SAMPLE_DIR)
        # At least the 5 original reports + some generated ones
        assert len(docs) >= 5
        for doc in docs[:5]:
            assert isinstance(doc, ClimateDocument)

    def test_infer_name_strips_common_words(self):
        name = LangChainDocumentLoader._infer_name("report_sustainability_greentech")
        assert "report" not in name.lower()
        assert "sustainability" not in name.lower()


class TestLangChainTextSplitter:
    def setup_method(self):
        self.splitter = LangChainTextSplitter(chunk_size=100, chunk_overlap=10)
        self.loader = LangChainDocumentLoader()

    def test_split_returns_chunks(self):
        doc = self.loader.load_file(SAMPLE_REPORT)
        chunks = self.splitter.chunk_document(doc)
        assert len(chunks) > 0

    def test_chunk_ids_unique(self):
        doc = self.loader.load_file(SAMPLE_REPORT)
        chunks = self.splitter.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_splitter_metadata_tag(self):
        doc = self.loader.load_file(SAMPLE_REPORT)
        chunks = self.splitter.chunk_document(doc)
        assert chunks[0].metadata["splitter"] == "langchain_recursive"

    def test_chunk_multiple_documents(self):
        docs = self.loader.load_directory(SAMPLE_DIR)[:3]
        all_chunks = self.splitter.chunk_documents(docs)
        doc_ids = {c.doc_id for c in all_chunks}
        assert len(doc_ids) == 3

    def test_no_empty_chunks(self):
        doc = self.loader.load_file(SAMPLE_REPORT)
        chunks = self.splitter.chunk_document(doc)
        for c in chunks:
            assert c.content.strip() != ""


class TestLangChainAuditChain:
    """Tests that verify LangChain chain construction (no real LLM calls)."""

    def test_chain_extract_claims_bad_response(self):
        """If LLM returns garbage, extract_claims should return empty list."""
        from src.langchain_pipeline.lc_chains import LangChainAuditChain

        chain = LangChainAuditChain(provider="groq", api_key="test_key")

        # Patch the internal _build_chain to return a mock that yields bad JSON
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "not json at all"
        chain._build_chain = MagicMock(return_value=mock_chain)

        # Force _llm to exist (skip provider init)
        chain._llm = MagicMock()

        result = chain.extract_claims("some report text")
        assert result == []

    def test_parse_json_strips_fences(self):
        from src.langchain_pipeline.lc_chains import LangChainAuditChain
        chain = LangChainAuditChain(provider="groq", api_key="k")
        # JSON wrapped in markdown code fence
        text = '```json\n{"key": "value"}\n```'
        result = chain._parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_plain(self):
        from src.langchain_pipeline.lc_chains import LangChainAuditChain
        chain = LangChainAuditChain(provider="groq", api_key="k")
        result = chain._parse_json('{"score": 0.8}')
        assert result["score"] == 0.8

    def test_parse_json_invalid_returns_empty(self):
        from src.langchain_pipeline.lc_chains import LangChainAuditChain
        chain = LangChainAuditChain(provider="groq", api_key="k")
        result = chain._parse_json("definitely not json")
        assert result == {}


class TestReportGenerator:
    def test_generate_reports(self):
        from data.sample_reports.generate_reports import generate_all
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_all(count=10, output_dir=Path(tmp), seed=99)
            assert len(paths) == 10
            for p in paths:
                assert p.exists()
                content = p.read_text()
                assert len(content) > 100

    def test_generate_grade_distribution(self):
        from data.sample_reports.generate_reports import generate_all, GRADES
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_all(count=50, output_dir=Path(tmp), seed=7)
            grades = set(p.stem.split("_")[2].upper() for p in paths)
            # Should have multiple grade levels
            assert len(grades) >= 3

    def test_generate_company_names_unique(self):
        from data.sample_reports.generate_reports import generate_all
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_all(count=30, output_dir=Path(tmp), seed=5)
            stems = [p.stem for p in paths]
            # All stems should be unique
            assert len(stems) == len(set(stems))
