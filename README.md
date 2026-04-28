# Climate Commitment LLM Auditing Pipeline

A RAG-based auditing pipeline that assesses the consistency and credibility of corporate climate disclosures against IPCC AR6 benchmark targets. The system uses adversarial prompting to stress-test LLM outputs for greenwashing and misinformation detection.

## Overview

```
Corporate Report (PDF/TXT)
        │
        ▼
┌──────────────────┐     ┌───────────────────┐
│  Document        │     │  IPCC AR6         │
│  Ingestion &     │     │  Benchmark        │
│  Chunking        │     │  Indexing         │
└────────┬─────────┘     └────────┬──────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────┐
│           ChromaDB Vector Store             │
│   (reports collection + benchmarks coll.)  │
└─────────────────────┬───────────────────────┘
                      │ RAG Retrieval
                      ▼
┌─────────────────────────────────────────────┐
│              Audit Engine                   │
│  ┌────────────────┐  ┌────────────────────┐ │
│  │  Consistency   │  │  Greenwashing      │ │
│  │  Checker       │  │  Detector          │ │
│  └────────────────┘  └────────────────────┘ │
│  ┌────────────────┐  ┌────────────────────┐ │
│  │  Credibility   │  │  Adversarial       │ │
│  │  Scorer        │  │  Prompter          │ │
│  └────────────────┘  └────────────────────┘ │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│           Evaluation Metrics                │
│  • Hallucination Rate                       │
│  • Factual Accuracy                         │
│  • Policy Alignment Score                   │
│  • Greenwashing Index                       │
│  • Robustness Score                         │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
             JSON Audit Report + Grade
```

## Features

- **RAG-based fact-checking**: Climate claims are retrieved against IPCC AR6 benchmarks using semantic search (ChromaDB + sentence-transformers)
- **Claim extraction**: Automatically extracts quantitative and qualitative climate commitments from reports
- **Consistency verification**: Each claim is checked against relevant IPCC 1.5°C/2°C pathway data
- **Greenwashing detection**: 8-indicator detection framework (offset dependency, Scope 3 omission, vague language, cherry-picked baselines, etc.)
- **Adversarial stress-testing**: Four attack strategies to test LLM robustness (contradiction probing, claim perturbation, devil's advocate, misinformation injection)
- **Evaluation metrics**: Hallucination rate, factual accuracy, policy alignment, and composite audit score with letter grades (A–F)
- **Batch benchmarking**: Reproducible methodology for auditing 200+ reports with JSON output

## Project Structure

```
Climate Commitment LLM Auditing Pipeline/
├── config/
│   └── settings.py             # Pydantic-based configuration
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py  # PDF/TXT loading
│   │   └── chunker.py          # Overlapping text chunking
│   ├── rag/
│   │   ├── embeddings.py       # sentence-transformers wrapper
│   │   ├── vector_store.py     # ChromaDB operations
│   │   └── retriever.py        # High-level retrieval interface
│   ├── benchmarks/
│   │   └── ipcc_targets.py     # IPCC AR6 data → vector store
│   ├── audit/
│   │   ├── consistency_checker.py   # Claim extraction + IPCC verification
│   │   ├── credibility_scorer.py    # Multi-dimensional credibility scoring
│   │   └── greenwashing_detector.py # 8-indicator greenwashing analysis
│   ├── adversarial/
│   │   └── adversarial_prompter.py  # 4-strategy adversarial testing
│   └── evaluation/
│       ├── metrics.py          # Hallucination, accuracy, alignment metrics
│       └── benchmarker.py      # Full pipeline orchestrator
├── data/
│   ├── ipcc_benchmarks/
│   │   └── ar6_targets.json    # IPCC AR6 structured benchmark data
│   └── sample_reports/         # 5 synthetic climate reports (A–F spectrum)
├── tests/                      # pytest test suite
├── outputs/                    # Generated audit results (JSON)
├── main.py                     # CLI entry point
├── requirements.txt
└── .env.example
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the pipeline

```bash
# Audit all 5 sample reports
python main.py --mode benchmark

# Audit a single report
python main.py --mode single --report data/sample_reports/report_01_credible_greentech_corp.txt

# Re-index IPCC benchmarks
python main.py --mode setup --force-reindex
```

## Evaluation Metrics

| Metric | Description | Weight |
|--------|-------------|--------|
| **Factual Accuracy** | % of claims consistent with IPCC AR6 data | 30% |
| **Policy Alignment** | Alignment with Paris Agreement / SBTi frameworks | 25% |
| **Anti-Greenwashing** | Inverse of greenwashing index (8 indicators) | 25% |
| **Robustness Score** | LLM consistency under adversarial perturbation | 20% |

**Grading Scale**: A (≥0.85) · B (≥0.70) · C (≥0.55) · D (≥0.40) · F (<0.40)

## Sample Reports

Five synthetic corporate climate reports covering the full credibility spectrum:

| Report | Company | Expected Grade | Key Issues |
|--------|---------|---------------|-----------|
| `report_01` | GreenTech Corp | A | Science-aligned, SBTi-validated, Scope 3 covered |
| `report_02` | Midstream Energy | C | No Scope 3, intensity vs absolute targets |
| `report_03` | FashionForward | D | 100% offsets, no Scope 3, vague language |
| `report_04` | PetroMax Industries | F | 2070 net-zero, absolute emissions rising, >80% offsets |
| `report_05` | Advanced SteelWorks | B | Hard-to-abate sector, credible H-DRI roadmap |

## IPCC AR6 Benchmarks (Key Targets)

| Pathway | Metric | Target |
|---------|--------|--------|
| 1.5°C | CO2 reduction by 2030 | −43 to −48% (vs 2019) |
| 1.5°C | Global net-zero year | 2050 |
| 1.5°C | Methane reduction by 2030 | −34% |
| 2°C | CO2 reduction by 2030 | −27 to −32% |
| Energy | Low-carbon electricity by 2050 | 90–99% |
| Transport | Emission reduction by 2050 | −50 to −90% |
| Industry | Emission reduction by 2050 | −63 to −90% |

## Adversarial Testing Strategies

1. **Contradiction Probing**: Find internal inconsistencies within a single report
2. **Claim Perturbation**: Mutate numerical targets (deflate/inflate) and test if LLM catches errors
3. **Devil's Advocate**: Argue against the LLM's own positive assessment
4. **Misinformation Injection**: Test detection of false IPCC data injected into prompts

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocked LLM responses (no API calls required for the unit test suite). Integration tests for the RAG pipeline use local ChromaDB and sentence-transformers.

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required for LLM calls |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformer model |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector store persistence path |
| `BATCH_SIZE` | 10 | Documents per batch |

## Output Format

```json
{
  "run_id": "20240424_143022",
  "timestamp": "2024-04-24T14:30:22",
  "total_reports": 5,
  "successful_audits": 5,
  "aggregate_metrics": {
    "avg_hallucination_rate": 0.08,
    "avg_factual_accuracy": 0.71,
    "avg_policy_alignment": 0.63,
    "avg_greenwashing_index": 0.41,
    "avg_overall_score": 0.64,
    "grade_distribution": {"A": 1, "B": 1, "C": 1, "D": 1, "F": 1}
  },
  "results": [...]
}
```
