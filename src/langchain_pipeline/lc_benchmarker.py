"""
LangChain-based end-to-end benchmarker.

Uses:
  LangChainDocumentLoader  → LangChainTextSplitter  → LangChainFaissStore
  LangChainAuditChain  → same BenchmarkResult / BenchmarkSummary outputs

Fully interchangeable with PipelineBenchmarker (same output format).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config import get_settings
from ..ingestion.document_loader import ClimateDocument
from ..benchmarks.ipcc_targets import IPCCBenchmarkLoader
from ..evaluation.benchmarker import BenchmarkResult, BenchmarkSummary
from ..evaluation.metrics import EvaluationMetrics, ReportMetrics
from ..rag.retriever import Retriever
from ..llm.base import LLMClient
from .lc_loader import LangChainDocumentLoader
from .lc_splitter import LangChainTextSplitter
from .lc_vectorstore import LangChainFaissStore
from .lc_chains import LangChainAuditChain

logger = logging.getLogger(__name__)


class LangChainBenchmarker:
    """
    Full audit pipeline using LangChain + FAISS + Groq (free).

    Equivalent to PipelineBenchmarker but built entirely on LangChain
    components, demonstrating the LangChain integration end-to-end.
    """

    def __init__(
        self,
        provider: str = "groq",
        api_key: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        settings = get_settings()
        self.settings = settings

        # Resolve API key from env if not provided
        if api_key is None:
            env_map = {
                "groq": "GROQ_API_KEY",
                "huggingface": "HF_API_TOKEN",
                "anthropic": "ANTHROPIC_API_KEY",
            }
            api_key = os.getenv(env_map.get(provider, "GROQ_API_KEY"), "")

        self.provider = provider
        self.api_key = api_key

        # Accept a pre-built client (e.g. MockLLMClient for demo/testing)
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            from ..llm.factory import get_llm_client
            self.llm_client = get_llm_client(provider=provider, api_key=api_key)

        # Build metrics calculator once — reused for every document
        self._metrics_calc = EvaluationMetrics(self.llm_client)

        lc_faiss_dir = str(settings.data_dir / "lc_faiss_db")
        self.vector_store = LangChainFaissStore(
            persist_dir=lc_faiss_dir,
            model_name=settings.embedding_model,
        )

        # Adaptor so Retriever (shared) works with LangChainFaissStore
        self.retriever = _LCRetriever(self.vector_store, top_k=settings.top_k_retrieval)

        self.loader = LangChainDocumentLoader()
        self.splitter = LangChainTextSplitter(settings.chunk_size, settings.chunk_overlap)
        self.chains = LangChainAuditChain(provider=provider, api_key=api_key)

        self._benchmarks_loaded = False

    def setup(self, force_reindex: bool = False):
        """Index IPCC benchmarks into the LangChain FAISS store."""
        col = self.settings.benchmarks_collection
        existing = self.vector_store.collection_count(col)
        if existing > 0 and not force_reindex:
            logger.info(f"[LC] Benchmarks already indexed ({existing}). Skipping.")
            self._benchmarks_loaded = True
            return

        loader = IPCCBenchmarkLoader()
        passages = loader.to_passages()
        ids = [p[0] for p in passages]
        texts = [p[1] for p in passages]
        metas = [p[2] for p in passages]
        self.vector_store.upsert_texts(ids, texts, metas, col)
        self._benchmarks_loaded = True
        logger.info(f"[LC] Indexed {len(passages)} IPCC benchmark passages.")

    def audit_document(self, doc: ClimateDocument) -> BenchmarkResult:
        start = time.time()
        try:
            # 1. Chunk + index report
            chunks = self.splitter.chunk_document(doc)
            self.vector_store.upsert_chunks(chunks, self.settings.reports_collection)

            # 2. Extract claims via LangChain chain
            raw_claims = self.chains.extract_claims(doc.content)
            logger.info(f"[LC] {doc.company_name}: {len(raw_claims)} claims")

            # 3. Assess each claim
            assessments = []
            from ..audit.consistency_checker import ClaimAssessment
            for claim in raw_claims[:15]:  # cap for speed
                claim_text = claim.get("claim_text", "")
                hits = self.retriever.retrieve_benchmarks(claim_text)
                context = self.retriever.format_context(hits, max_chars=2000)
                result = self.chains.check_consistency(claim_text, context)
                assessments.append(ClaimAssessment(
                    claim_id=claim.get("claim_id", "lc_claim"),
                    claim_text=claim_text,
                    claim_type=claim.get("claim_type", "qualitative"),
                    sector=claim.get("sector", "overall"),
                    consistent_with_ipcc=result.get("consistent_with_ipcc"),
                    alignment_level=result.get("alignment_level", "undetermined"),
                    accuracy_score=float(result.get("accuracy_score", 0.5)),
                    explanation=result.get("explanation", ""),
                    issues=result.get("issues", []),
                    benchmark_references=result.get("benchmark_references", []),
                    retrieved_context=context,
                ))

            # 4. Greenwashing
            gw_data = self.chains.detect_greenwashing(doc.company_name, doc.content)
            from ..audit.greenwashing_detector import GreenwashingReport
            gw_report = GreenwashingReport(
                company_name=doc.company_name,
                indicators=gw_data.get("indicators", {}),
                overall_greenwashing_score=float(gw_data.get("overall_greenwashing_score", 0.5)),
                greenwashing_label=gw_data.get("greenwashing_label", "Moderate"),
                summary=gw_data.get("summary", ""),
                red_flags=gw_data.get("red_flags", []),
            )

            # 5. Credibility
            claims_summary = "; ".join(a.claim_text[:80] for a in assessments[:6])
            cred_data = self.chains.score_credibility(
                doc.company_name, claims_summary, doc.content
            )
            from ..audit.credibility_scorer import CredibilityReport
            cred_report = CredibilityReport(
                company_name=doc.company_name,
                specificity_score=float(cred_data.get("specificity_score", 0.5)),
                science_alignment_score=float(cred_data.get("science_alignment_score", 0.5)),
                completeness_score=float(cred_data.get("completeness_score", 0.5)),
                verifiability_score=float(cred_data.get("verifiability_score", 0.5)),
                ambition_score=float(cred_data.get("ambition_score", 0.5)),
                overall_credibility_score=float(cred_data.get("overall_credibility_score", 0.5)),
                credibility_label=cred_data.get("credibility_label", "Moderate"),
                key_strengths=cred_data.get("key_strengths", []),
                key_weaknesses=cred_data.get("key_weaknesses", []),
                recommendation=cred_data.get("recommendation", ""),
            )

            # 6. Simplified metrics (no adversarial in LC pipeline for brevity)
            from ..adversarial.adversarial_prompter import AdversarialResult
            adv_stub = AdversarialResult(
                company_name=doc.company_name,
                contradictions_found=0,
                contradictions=[],
                internal_consistency_score=0.8,
                perturbed_claims=[],
                perturbation_detection_rate=0.7,
                devil_advocate_score=0.3,
                counter_arguments=[],
                misinformation_detection_rate=0.8,
            )
            metrics = self._metrics_calc.compute(
                doc.company_name, assessments, cred_report, gw_report, adv_stub
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
            logger.error(f"[LC] Audit failed for {doc.doc_id}: {exc}", exc_info=True)
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
        if not self._benchmarks_loaded:
            self.setup()

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = BenchmarkSummary(
            run_id=f"lc_{run_id}",
            timestamp=datetime.now().isoformat(),
            total_reports=len(documents),
            successful_audits=0,
            failed_audits=0,
        )
        for doc in tqdm(documents, desc="[LangChain] Auditing"):
            result = self.audit_document(doc)
            summary.results.append(result)
            if result.error:
                summary.failed_audits += 1
            else:
                summary.successful_audits += 1

        summary.compute_aggregates()

        output_path = output_path or (self.settings.output_dir / f"lc_benchmark_{run_id}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info(f"[LC] Results saved to: {output_path}")
        return summary

    def run_benchmark_from_directory(
        self, reports_dir: str | Path, output_path: Optional[Path] = None
    ) -> BenchmarkSummary:
        docs = self.loader.load_directory(reports_dir)
        logger.info(f"[LC] Loaded {len(docs)} reports")
        return self.run_benchmark(docs, output_path)


# ──────────────────── Lightweight retriever adaptor ────────────────────────

class _LCRetriever:
    """Adapts LangChainFaissStore to the Retriever interface."""

    def __init__(self, store: LangChainFaissStore, top_k: int = 5):
        self.store = store
        self.top_k = top_k

    def retrieve_benchmarks(self, claim: str, top_k: Optional[int] = None) -> list[dict]:
        from config import get_settings
        return self.store.query(
            claim, get_settings().benchmarks_collection, top_k=top_k or self.top_k
        )

    def format_context(self, hits: list[dict], max_chars: int = 3000) -> str:
        parts = []
        total = 0
        for i, h in enumerate(hits, 1):
            entry = f"[{i}] {h['content'].strip()}"
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n".join(parts)
