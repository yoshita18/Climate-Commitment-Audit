#!/usr/bin/env python3
"""
Climate Commitment LLM Auditing Pipeline
=========================================
RAG-based pipeline to audit corporate climate disclosures against IPCC AR6 benchmarks.

Free LLM options (no credit card required):
  --provider groq         Groq Cloud (Llama 3.3-70B) — fastest, recommended
                          Get free key at: https://console.groq.com
  --provider huggingface  HuggingFace Inference API (Mistral-7B)
                          Get free token at: https://huggingface.co/settings/tokens

Vector store options:
  --backend chroma        ChromaDB (default, persistent, no install)
  --backend faiss         FAISS (faiss-cpu, file-based)

Pipeline options:
  --pipeline original     Direct Anthropic SDK + ChromaDB/FAISS (default)
  --pipeline langchain    LangChain chains + FAISS + ChatGroq

Usage examples:
    # ── Demo / offline (no API key needed) ──────────────────────────────────
    python main.py --demo
    python main.py --demo --mode single --report data/sample_reports/report_01_credible_greentech_corp.txt
    python main.py --demo --mode benchmark --pipeline langchain --backend faiss

    # ── Live (set GROQ_API_KEY in .env first) ───────────────────────────────
    python main.py --mode benchmark --provider groq
    python main.py --mode single --report data/sample_reports/report_01_credible_greentech_corp.txt

    # Batch audit all 205 reports with LangChain + FAISS
    python main.py --mode benchmark --pipeline langchain --backend faiss --provider groq

    # Generate 200+ synthetic reports
    python main.py --mode generate --count 200

    # Re-index IPCC benchmarks
    python main.py --mode setup --force-reindex
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_banner():
    console.print(Panel.fit(
        "[bold green]Climate Commitment LLM Auditing Pipeline[/bold green]\n"
        "[dim]RAG · FAISS · LangChain · HuggingFace · Groq (FREE) · IPCC AR6[/dim]",
        border_style="green",
    ))


def _make_llm(args):
    """Return the right LLMClient based on --demo / --provider flags."""
    if getattr(args, "demo", False) or getattr(args, "provider", None) == "mock":
        from src.llm.mock import MockLLMClient
        console.print("[yellow][DEMO][/yellow] Using MockLLMClient — no API key required.")
        return MockLLMClient()

    from src.llm.factory import get_llm_client
    return get_llm_client(provider=args.provider, api_key=args.api_key)


def build_benchmarker(args):
    """Build the appropriate benchmarker based on --pipeline / --demo flags."""
    llm = _make_llm(args)

    if args.pipeline == "langchain":
        from src.langchain_pipeline.lc_benchmarker import LangChainBenchmarker
        return LangChainBenchmarker(
            provider=args.provider or "groq",
            api_key=args.api_key,
            llm_client=llm,
        )

    from src.evaluation.benchmarker import PipelineBenchmarker

    # Select vector store backend
    if args.backend == "faiss":
        from config import get_settings
        from src.rag.embeddings import EmbeddingModel
        from src.rag.faiss_store import FaissVectorStore
        from src.rag.retriever import Retriever

        settings = get_settings()
        emb = EmbeddingModel(settings.embedding_model)
        vs = FaissVectorStore(settings.faiss_persist_dir, emb)
        retriever = Retriever(vs, top_k=settings.top_k_retrieval)
        bm = PipelineBenchmarker(llm_client=llm)
        bm.vector_store = vs
        bm.retriever = retriever
        # Re-wire audit components with new retriever
        from src.audit.consistency_checker import ConsistencyChecker
        from src.adversarial.adversarial_prompter import AdversarialPrompter
        bm.consistency_checker = ConsistencyChecker(llm, retriever)
        bm.adversarial_prompter = AdversarialPrompter(llm, retriever)
        return bm

    return PipelineBenchmarker(llm_client=llm)


def print_results_table(results: list):
    table = Table(
        title="Audit Results Summary",
        box=box.ROUNDED, show_header=True, header_style="bold cyan"
    )
    table.add_column("Company", min_width=22)
    table.add_column("Grade", justify="center", min_width=6)
    table.add_column("Overall", justify="right", min_width=8)
    table.add_column("Accuracy", justify="right", min_width=8)
    table.add_column("Policy", justify="right", min_width=8)
    table.add_column("GreenWash", justify="right", min_width=10)
    table.add_column("Claims", justify="right", min_width=7)
    table.add_column("Time(s)", justify="right", min_width=8)

    colours = {"A": "green", "B": "blue", "C": "yellow", "D": "orange1", "F": "red"}

    for r in results:
        if r.error:
            table.add_row(r.company_name[:22], "[red]ERR[/red]", "—","—","—","—","—","—")
            continue
        m = r.metrics
        c = colours.get(m.grade, "white")
        table.add_row(
            r.company_name[:22],
            f"[{c}]{m.grade}[/{c}]",
            f"{m.overall_audit_score:.2f}",
            f"{m.factual_accuracy:.2f}",
            f"{m.policy_alignment_score:.2f}",
            f"{m.greenwashing_index:.2f}",
            str(r.num_claims),
            f"{r.processing_time_seconds:.1f}",
        )
    console.print(table)


def run_single(args):
    if not args.report:
        console.print("[red]--report required for --mode single[/red]")
        sys.exit(1)
    console.print(f"[cyan]Auditing:[/cyan] {args.report}")
    bm = build_benchmarker(args)
    bm.setup()
    doc = bm.loader.load_file(args.report)
    console.print(f"[green]Loaded:[/green] {doc.company_name} ({doc.word_count} words)")
    result = bm.audit_document(doc)
    if result.error:
        console.print(f"[red]Failed:[/red] {result.error}")
        return
    m = result.metrics
    console.print(Panel(
        f"[bold]Company:[/bold]           {result.company_name}\n"
        f"[bold]Overall Score:[/bold]     {m.overall_audit_score:.3f}  Grade: {m.grade}\n"
        f"[bold]Factual Accuracy:[/bold]  {m.factual_accuracy:.3f}\n"
        f"[bold]Policy Alignment:[/bold]  {m.policy_alignment_score:.3f}\n"
        f"[bold]Greenwashing Index:[/bold]{m.greenwashing_index:.3f}\n"
        f"[bold]Hallucination Rate:[/bold]{m.hallucination_rate:.3f}\n"
        f"[bold]Robustness Score:[/bold]  {m.robustness_score:.3f}\n"
        f"[bold]Claims Assessed:[/bold]   {result.num_claims}\n"
        f"[bold]Time:[/bold]              {result.processing_time_seconds:.1f}s\n"
        f"[bold]LLM Backend:[/bold]       {bm.llm_client}",
        title="[bold green]Audit Report[/bold green]",
        border_style="green",
    ))


def run_benchmark(args):
    reports_dir = args.reports_dir or "data/sample_reports"
    console.print(f"[cyan]Benchmarking:[/cyan] {reports_dir}")
    bm = build_benchmarker(args)
    bm.setup()
    out = Path("outputs") / f"benchmark_{args.pipeline}_{args.backend}.json"
    summary = bm.run_benchmark_from_directory(reports_dir, output_path=out)
    print_results_table(summary.results[:30])  # show first 30 rows
    console.print(Panel(
        f"[bold]Total Reports:[/bold]       {summary.total_reports}\n"
        f"[bold]Successful:[/bold]          {summary.successful_audits}\n"
        f"[bold]Failed:[/bold]              {summary.failed_audits}\n"
        f"[bold]Avg Overall Score:[/bold]   {summary.avg_overall_score:.3f}\n"
        f"[bold]Avg Factual Accuracy:[/bold]{summary.avg_factual_accuracy:.3f}\n"
        f"[bold]Avg Greenwashing:[/bold]    {summary.avg_greenwashing_index:.3f}\n"
        f"[bold]Grade Distribution:[/bold]  {summary.grade_distribution}\n"
        f"[bold]Results saved:[/bold]       {out}",
        title="[bold green]Benchmark Summary[/bold green]",
        border_style="green",
    ))


def run_setup(args):
    console.print("[cyan]Indexing IPCC AR6 benchmarks...[/cyan]")
    bm = build_benchmarker(args)
    bm.setup(force_reindex=args.force_reindex)
    console.print("[green]Setup complete.[/green]")


def run_generate(args):
    from data.sample_reports.generate_reports import generate_all
    count = args.count or 200
    console.print(f"[cyan]Generating {count} synthetic climate reports...[/cyan]")
    paths = generate_all(count=count, seed=args.seed or 42)
    console.print(f"[green]Done:[/green] {len(paths)} reports written to data/sample_reports/")


def main():
    parser = argparse.ArgumentParser(
        description="Climate Commitment LLM Auditing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run with mock LLM — no API key required. "
                             "Great for demos and offline testing.")
    parser.add_argument("--mode", choices=["single", "benchmark", "setup", "generate"],
                        default="single")
    parser.add_argument("--report", help="Path to single report (--mode single)",
                        default="data/sample_reports/report_01_credible_greentech_corp.txt")
    parser.add_argument("--reports-dir", default=None,
                        help="Directory of reports (--mode benchmark)")
    parser.add_argument("--provider", choices=["groq", "huggingface", "anthropic", "mock"],
                        default=None, help="LLM provider (default: auto-detect from env; use 'mock' for demo)")
    parser.add_argument("--api-key", default=None, help="API key (overrides env var)")
    parser.add_argument("--backend", choices=["chroma", "faiss"],
                        default="chroma", help="Vector store backend (default: chroma)")
    parser.add_argument("--pipeline", choices=["original", "langchain"],
                        default="original", help="Pipeline type (default: original)")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--count", type=int, default=200,
                        help="Number of reports to generate (--mode generate)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for report generation")

    args = parser.parse_args()
    print_banner()

    dispatch = {
        "single": run_single,
        "benchmark": run_benchmark,
        "setup": run_setup,
        "generate": run_generate,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
