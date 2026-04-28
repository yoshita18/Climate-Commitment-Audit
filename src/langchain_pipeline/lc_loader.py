"""
LangChain document loading for climate reports.

Wraps LangChain's TextLoader / PyPDFLoader and converts to our
ClimateDocument dataclass so the rest of the pipeline stays consistent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..ingestion.document_loader import ClimateDocument

logger = logging.getLogger(__name__)

SUPPORTED = {".txt", ".md", ".pdf"}


class LangChainDocumentLoader:
    """Load climate reports using LangChain loaders, return ClimateDocument list."""

    def load_file(
        self,
        path: str | Path,
        company_name: Optional[str] = None,
        report_year: int = 2024,
        sector: str = "unknown",
    ) -> ClimateDocument:
        path = Path(path)
        if path.suffix not in SUPPORTED:
            raise ValueError(f"Unsupported extension: {path.suffix}")
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

        content = self._load(path)
        name = company_name or self._infer_name(path.stem)
        doc_id = path.stem.lower().replace(" ", "_")

        return ClimateDocument(
            doc_id=doc_id,
            company_name=name,
            report_year=report_year,
            content=content,
            source_path=str(path),
            sector=sector,
        )

    def load_directory(
        self, directory: str | Path, report_year: int = 2024
    ) -> list[ClimateDocument]:
        directory = Path(directory)
        docs = []
        for p in sorted(directory.iterdir()):
            if p.suffix in SUPPORTED and not p.name.startswith("_"):
                try:
                    docs.append(self.load_file(p, report_year=report_year))
                    logger.info(f"[LC] Loaded: {p.name}")
                except Exception as exc:
                    logger.error(f"[LC] Failed {p.name}: {exc}")
        return docs

    # ─────────────────────────────────────────────────

    def _load(self, path: Path) -> str:
        if path.suffix == ".pdf":
            return self._load_pdf(path)
        # txt / md — use LangChain TextLoader
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        return "\n\n".join(d.page_content for d in docs)

    def _load_pdf(self, path: Path) -> str:
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(path))
            pages = loader.load()
            return "\n\n".join(p.page_content for p in pages if p.page_content.strip())
        except ImportError:
            # Fall back to pdfplumber if langchain PDF deps missing
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n\n".join(
                    p.extract_text() for p in pdf.pages if p.extract_text()
                )

    @staticmethod
    def _infer_name(stem: str) -> str:
        skip = {"report", "disclosure", "sustainability", "climate", "esg"}
        parts = [p.title() for p in stem.split("_") if p.lower() not in skip]
        return " ".join(parts) if parts else stem.replace("_", " ").title()
