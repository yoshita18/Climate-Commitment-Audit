from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ClimateDocument:
    doc_id: str
    company_name: str
    report_year: int
    content: str
    source_path: str
    sector: str = "unknown"
    report_type: str = "sustainability_report"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.word_count = len(self.content.split())
        self.char_count = len(self.content)


class DocumentLoader:
    """Loads climate commitment reports from text or PDF files."""

    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}

    def __init__(self):
        self._pdf_available = self._check_pdf_support()

    def _check_pdf_support(self) -> bool:
        try:
            import pdfplumber  # noqa: F401
            return True
        except ImportError:
            logger.warning("pdfplumber not installed; PDF loading disabled.")
            return False

    def load_file(
        self,
        path: str | Path,
        company_name: Optional[str] = None,
        report_year: int = 2024,
        sector: str = "unknown",
    ) -> ClimateDocument:
        path = Path(path)
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        if not path.exists():
            raise FileNotFoundError(f"Report not found: {path}")

        if path.suffix == ".pdf":
            content = self._load_pdf(path)
        else:
            content = path.read_text(encoding="utf-8")

        doc_id = path.stem.lower().replace(" ", "_")
        name = company_name or self._infer_company_name(path.stem)

        return ClimateDocument(
            doc_id=doc_id,
            company_name=name,
            report_year=report_year,
            content=content,
            source_path=str(path),
            sector=sector,
        )

    def load_directory(
        self,
        directory: str | Path,
        report_year: int = 2024,
    ) -> list[ClimateDocument]:
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        documents = []
        for path in sorted(directory.iterdir()):
            if path.suffix in self.SUPPORTED_EXTENSIONS and not path.name.startswith("_"):
                try:
                    doc = self.load_file(path, report_year=report_year)
                    documents.append(doc)
                    logger.info(f"Loaded: {path.name} ({doc.word_count} words)")
                except Exception as exc:
                    logger.error(f"Failed to load {path.name}: {exc}")

        return documents

    def _load_pdf(self, path: Path) -> str:
        import pdfplumber

        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
        return "\n\n".join(pages)

    @staticmethod
    def _infer_company_name(stem: str) -> str:
        parts = stem.split("_")
        skip = {"report", "disclosure", "sustainability", "climate", "esg", "2023", "2024", "2025"}
        name_parts = [p.title() for p in parts if p.lower() not in skip]
        return " ".join(name_parts) if name_parts else stem.replace("_", " ").title()
