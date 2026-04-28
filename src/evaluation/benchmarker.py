"""
Reproducible benchmarking methodology for auditing 200+ climate reports.

Orchestrates the full pipeline for a batch of documents:
  load → chunk → index → extract claims → RAG retrieve → audit → adversarial → metrics → report
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config import get_settings
from ..llm.factory import get_llm_client
from ..llm.base import LLMClient
from ..ingestion.document_loader import DocumentLoader, ClimateDocument
from ..ingestion.chunker import TextChunker
from ..rag.embeddings import EmbeddingModel
from ..rag.vector_store import VectorStore
from ..rag.retriever import Retriever
from ..benchmarks.ipcc_targets import IPCCBenchmarkLoader
from ..audit.consistency_checker import ConsistencyChecker
from ..audit.credibility_scorer import CredibilityScorer
from ..audit.greenwashing_detector import GreenwashingDetector
from ..adversarial.adversarial_prompter import AdversarialPrompter
from .metrics import EvaluationMetrics, ReportMetrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    doc_id: str
    company_name: str
    report_year: int
    sector: str
    metrics: ReportMetrics
    processing_time_seconds: float
    num_chunks: int
    num_claims: int
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "company_name": self.company_name,
            "report_year": self.report_year,
            "sector": self.sector,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "num_chunks": self.num_chunks,
            "num_claims": self.num_claims,
            "error": self.error,
        }


@dataclass
class BenchmarkSummary:
    run_id: str
    timestamp: str
    total_reports: int
    successful_audits: int
    failed_audits: int
    results: list[BenchmarkResult] = field(default_factory=list)

    # Aggregate stats
    avg_hallucination_rate: float = 0.0
    avg_factual_accuracy: float = 0.0
    avg_policy_alignment: float = 0.0
    avg_greenwashing_index: float = 0.0
    avg_overall_score: float = 0.0
    grade_distribution: dict = field(default_factory=dict)

    def compute_aggregates(self):
        valid = [r for r in self.results if r.metrics and not r.error]
        if not valid:
            return
        n = len(valid)
        self.avg_hallucination_rate = sum(r.metrics.hallucination_rate for r in valid) / n
        self.avg_factual_accuracy = sum(r.metrics.factual_accuracy for r in valid) / n
        self.avg_policy_alignment = sum(r.metrics.policy_alignment_score for r in valid) / n
        self.avg_greenwashing_index = sum(r.metrics.greenwashing_index for r in valid) / n
        self.avg_overall_score = sum(r.metrics.overall_audit_score for r in valid) / n

        grades = {}
        for r in valid:
            g = r.metrics.grade
            grades[g] = grades.get(g, 0) + 1
        self.grade_distribution = grades

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_reports": self.total_reports,
            "successful_audits": self.successful_audits,
            "failed_audits": self.failed_audits,
            "aggregate_metrics": {
                "avg_hallucination_rate": round(self.avg_hallucination_rate, 4),
                "avg_factual_accuracy": round(self.avg_factual_accuracy, 4),
                "avg_policy_alignment": round(self.avg_policy_alignment, 4),
                "avg_greenwashing_index": round(self.avg_greenwashing_index, 4),
                "avg_overall_score": round(self.avg_overall_score, 4),
                "grade_distribution": self.grade_distribution,
            },
            "results": [r.to_dict() for r in self.results],
        }


class PipelineBenchmarker:
    """End-to-end pipeline orchestrator for auditing climate commitment reports at scale."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        settings = get_settings()
        self.settings = settings

        # LLM client: accept pre-built client, or auto-build from provider/key
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = get_llm_client(
                provider=llm_provider or settings.llm_provider,
                api_key=api_key,
            )
        logger.info(f"LLM backend: {self.llm_client}")

        self.embedding_model = EmbeddingModel(settings.embedding_model)
        self.vector_store = VectorStore(settings.chroma_persist_dir, self.embedding_model)
        self.retriever = Retriever(self.vector_store, top_k=settings.top_k_retrieval)
        self.loader = DocumentLoader()
        self.chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)

        self.consistency_checker = ConsistencyChecker(self.llm_client, self.retriever)
        self.credibility_scorer = CredibilityScorer(self.llm_client)
        self.greenwashing_detector = GreenwashingDetector(self.llm_client)
        self.adversarial_prompter = AdversarialPrompter(self.llm_client, self.retriever)
        self.eval_metrics = EvaluationMetrics(self.llm_client)

        self._benchmarks_loaded = False

    def setup(self, force_reindex: bool = False):
        """Index IPCC benchmarks into vector store."""
        loader = IPCCBenchmarkLoader()
        loader.index_into_vector_store(
            self.vector_store,
            self.settings.benchmarks_collection,
            force_reindex=force_reindex,
        )
        self._benchmarks_loaded = True
        logger.info("Pipeline setup complete.")

    def audit_document(self, doc: ClimateDocument) -> BenchmarkResult:
        """Run the full audit pipeline on a single document."""
        start = time.time()
        try:
            # 1. Chunk and index
            chunks = self.chunker.chunk_document(doc)
            self.vector_store.upsert_chunks(chunks, self.settings.reports_collection)

            # 2. Extract and assess claims
            assessments = self.consistency_checker.check_document(
                doc.content, doc_id=doc.doc_id
            )

            # 3. Credibility scoring
            cred_report = self.credibility_scorer.score(
                doc.company_name, doc.content, assessments
            )

            # 4. Greenwashing detection
            gw_report = self.greenwashing_detector.detect(doc.company_name, doc.content)

            # 5. Adversarial stress-testing
            claims_raw = [
                {"claim_text": a.claim_text, "claim_type": a.claim_type}
                for a in assessments
            ]
            positive_summary = cred_report.credibility_label + ": " + ", ".join(cred_report.key_strengths)
            adv_result = self.adversarial_prompter.run_full_adversarial_suite(
                doc.company_name, doc.content, claims_raw, positive_summary
            )

            # 6. Evaluation metrics
            metrics = self.eval_metrics.compute(
                doc.company_name, assessments, cred_report, gw_report, adv_result
            )

            elapsed = time.time() - start
            return BenchmarkResult(
                doc_id=doc.doc_id,
                company_name=doc.company_name,
                report_year=doc.report_year,
                sector=doc.sector,
                metrics=metrics,
                processing_time_seconds=elapsed,
                num_chunks=len(chunks),
                num_claims=len(assessments),
            )

        except Exception as exc:
            logger.error(f"Audit failed for {doc.doc_id}: {exc}", exc_info=True)
            elapsed = time.time() - start
            return BenchmarkResult(
                doc_id=doc.doc_id,
                company_name=doc.company_name,
                report_year=doc.report_year,
                sector=doc.sector,
                metrics=None,
                processing_time_seconds=elapsed,
                num_chunks=0,
                num_claims=0,
                error=str(exc),
            )

    def run_benchmark(
        self,
        documents: list[ClimateDocument],
        output_path: Optional[Path] = None,
    ) -> BenchmarkSummary:
        """Run the full benchmark across multiple documents."""
        if not self._benchmarks_loaded:
            self.setup()

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = BenchmarkSummary(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            total_reports=len(documents),
            successful_audits=0,
            failed_audits=0,
        )

        for doc in tqdm(documents, desc="Auditing reports"):
            result = self.audit_document(doc)
            summary.results.append(result)
            if result.error:
                summary.failed_audits += 1
            else:
                summary.successful_audits += 1

        summary.compute_aggregates()

        output_path = output_path or (self.settings.output_dir / f"benchmark_{run_id}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info(f"Benchmark results saved to: {output_path}")

        return summary

    def run_benchmark_from_directory(
        self,
        reports_dir: str | Path,
        output_path: Optional[Path] = None,
    ) -> BenchmarkSummary:
        docs = self.loader.load_directory(reports_dir)
        logger.info(f"Loaded {len(docs)} reports from {reports_dir}")
        return self.run_benchmark(docs, output_path)
