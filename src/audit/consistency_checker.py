from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..llm.base import LLMClient
from ..llm.json_utils import parse_llm_json
from ..rag.retriever import Retriever

logger = logging.getLogger(__name__)

CLAIM_EXTRACTION_PROMPT = """\
You are an expert climate disclosure analyst. Extract every specific, verifiable climate commitment or claim from the following corporate report excerpt.

Focus on:
- Quantitative targets (% reductions, absolute numbers, timelines)
- Net-zero or carbon neutrality commitments with dates
- Scope 1, 2, and 3 coverage claims
- Carbon offset or removal plans
- Sectoral commitments (energy, transport, buildings, etc.)
- Baseline year references

Return a JSON array. Each element must have:
{{
  "claim_id": "<unique short id>",
  "claim_text": "<exact or paraphrased claim>",
  "claim_type": "<quantitative|qualitative|commitment|target>",
  "sector": "<energy|transport|buildings|industry|overall|other>",
  "numbers_mentioned": ["<any specific numbers, dates, percentages>"]
}}

Report excerpt:
\"\"\"
{text}
\"\"\"

Return ONLY the JSON array, no other text."""


CONSISTENCY_CHECK_PROMPT = """\
You are an IPCC climate science expert auditing a corporate climate commitment for scientific accuracy.

Corporate claim:
\"{claim}\"

Relevant IPCC AR6 benchmark context:
{benchmark_context}

Assess the claim on these dimensions:
1. **Scientific Accuracy**: Is the claim consistent with IPCC AR6 data?
2. **Ambition Level**: Is the target aligned with 1.5°C, 2°C, or insufficient?
3. **Completeness**: Are there important caveats or missing information?
4. **Credibility Issues**: Any signs of misleading framing?

Respond as JSON:
{{
  "consistent_with_ipcc": <true|false|null>,
  "alignment_level": "<1.5C_aligned|2C_aligned|insufficient|undetermined>",
  "accuracy_score": <0.0 to 1.0>,
  "explanation": "<2-3 sentence explanation>",
  "issues": ["<specific concerns, empty if none>"],
  "benchmark_references": ["<relevant IPCC targets>"]
}}

Return ONLY the JSON object."""


@dataclass
class ClaimAssessment:
    claim_id: str
    claim_text: str
    claim_type: str
    sector: str
    consistent_with_ipcc: Optional[bool]
    alignment_level: str
    accuracy_score: float
    explanation: str
    issues: list[str] = field(default_factory=list)
    benchmark_references: list[str] = field(default_factory=list)
    retrieved_context: str = ""


class ConsistencyChecker:
    """Extracts climate claims and verifies them against IPCC benchmarks via RAG."""

    SYSTEM_PROMPT = (
        "You are a climate science expert and sustainability disclosure auditor. "
        "Always respond with valid JSON as specified in the prompt."
    )

    def __init__(self, llm_client: LLMClient, retriever: Retriever):
        self.llm_client = llm_client
        self.retriever = retriever

    def extract_claims(self, text: str) -> list[dict]:
        prompt = CLAIM_EXTRACTION_PROMPT.format(text=text[:4000])
        response = self._call_llm(prompt, max_tokens=1500)
        claims = parse_llm_json(response, fallback=[], label="extract_claims")
        if not isinstance(claims, list):
            logger.warning("extract_claims: expected list, got %s — wrapping", type(claims))
            claims = list(claims.values()) if isinstance(claims, dict) else []
        return claims

    def assess_claim(self, claim: dict) -> ClaimAssessment:
        claim_text = claim.get("claim_text", "")
        hits = self.retriever.retrieve_benchmarks(claim_text, top_k=4)
        benchmark_context = self.retriever.format_context(hits, max_chars=2000)

        prompt = CONSISTENCY_CHECK_PROMPT.format(
            claim=claim_text,
            benchmark_context=benchmark_context,
        )
        response = self._call_llm(prompt, max_tokens=800)

        result = parse_llm_json(response, label="assess_claim") or {
            "consistent_with_ipcc": None,
            "alignment_level": "undetermined",
            "accuracy_score": 0.5,
            "explanation": "Could not parse LLM response.",
            "issues": [],
            "benchmark_references": [],
        }

        return ClaimAssessment(
            claim_id=claim.get("claim_id", "unknown"),
            claim_text=claim_text,
            claim_type=claim.get("claim_type", "qualitative"),
            sector=claim.get("sector", "overall"),
            consistent_with_ipcc=result.get("consistent_with_ipcc"),
            alignment_level=result.get("alignment_level", "undetermined"),
            accuracy_score=float(result.get("accuracy_score", 0.5)),
            explanation=result.get("explanation", ""),
            issues=result.get("issues", []),
            benchmark_references=result.get("benchmark_references", []),
            retrieved_context=benchmark_context,
        )

    def check_document(self, text: str, doc_id: str = "") -> list[ClaimAssessment]:
        logger.info(f"Extracting claims from document: {doc_id}")
        claims = self.extract_claims(text)
        logger.info(f"Found {len(claims)} claims")

        assessments = []
        for claim in claims:
            try:
                assessment = self.assess_claim(claim)
                assessments.append(assessment)
            except Exception as exc:
                logger.error(f"Failed to assess claim '{claim.get('claim_id')}': {exc}")

        return assessments

    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        return self.llm_client.complete(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
