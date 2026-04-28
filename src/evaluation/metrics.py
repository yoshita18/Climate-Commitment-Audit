"""
Evaluation metrics for the LLM auditing pipeline.

Metrics computed per report:
  - hallucination_rate       : fraction of LLM claims not grounded in retrieved context
  - factual_accuracy         : fraction of quantitative claims consistent with IPCC data
  - policy_alignment_score   : degree to which targets align with Paris/SBTi frameworks
  - greenwashing_index       : composite greenwashing risk score
  - overall_audit_score      : weighted composite of all above
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ..llm.base import LLMClient
from ..llm.json_utils import parse_llm_json
from ..audit.consistency_checker import ClaimAssessment
from ..audit.credibility_scorer import CredibilityReport
from ..audit.greenwashing_detector import GreenwashingReport
from ..adversarial.adversarial_prompter import AdversarialResult

logger = logging.getLogger(__name__)

HALLUCINATION_CHECK_PROMPT = """\
You are evaluating whether an LLM-generated claim is grounded in the provided source context.

LLM claim: "{claim}"

Retrieved source context:
\"\"\"
{context}
\"\"\"

Is the claim fully supported by the context above?
- "supported": claim is clearly backed by the context
- "partially_supported": claim is related but context doesn't fully confirm it
- "unsupported": claim makes specific assertions not present in the context (potential hallucination)
- "contradicted": context explicitly contradicts the claim

Respond as JSON: {{"grounding": "<supported|partially_supported|unsupported|contradicted>", "confidence": <0.0-1.0>}}"""

POLICY_ALIGNMENT_PROMPT = """\
You are evaluating corporate climate policy alignment with international frameworks.

Company: {company}
Claims assessed: {claims_summary}

Score the company's policy alignment (0.0 = none, 1.0 = fully aligned) on:
1. paris_agreement_alignment: Alignment with Paris Agreement goals
2. sbti_alignment: Alignment with Science Based Targets initiative criteria
3. ipcc_1_5c_alignment: Alignment with IPCC 1.5°C pathway
4. ndc_consistency: Consistency with national climate policies

Respond as JSON:
{{
  "paris_agreement_alignment": 0.0,
  "sbti_alignment": 0.0,
  "ipcc_1_5c_alignment": 0.0,
  "ndc_consistency": 0.0,
  "overall_policy_alignment": 0.0,
  "alignment_gaps": []
}}"""


@dataclass
class ReportMetrics:
    company_name: str
    hallucination_rate: float
    factual_accuracy: float
    policy_alignment_score: float
    greenwashing_index: float
    robustness_score: float
    overall_audit_score: float
    grade: str
    breakdown: dict = field(default_factory=dict)
    policy_alignment_detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "company_name": self.company_name,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "factual_accuracy": round(self.factual_accuracy, 4),
            "policy_alignment_score": round(self.policy_alignment_score, 4),
            "greenwashing_index": round(self.greenwashing_index, 4),
            "robustness_score": round(self.robustness_score, 4),
            "overall_audit_score": round(self.overall_audit_score, 4),
            "grade": self.grade,
            "breakdown": self.breakdown,
            "policy_alignment_detail": self.policy_alignment_detail,
        }

    @staticmethod
    def grade_from_score(score: float) -> str:
        if score >= 0.85:
            return "A"
        if score >= 0.70:
            return "B"
        if score >= 0.55:
            return "C"
        if score >= 0.40:
            return "D"
        return "F"


class EvaluationMetrics:
    """Computes all evaluation metrics for one audit run."""

    METRIC_WEIGHTS = {
        "factual_accuracy": 0.30,
        "policy_alignment_score": 0.25,
        "anti_greenwashing": 0.25,
        "robustness_score": 0.20,
    }

    SYSTEM_PROMPT = "You are a climate policy expert. Respond only with valid JSON."

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def compute(
        self,
        company_name: str,
        claim_assessments: list[ClaimAssessment],
        credibility_report: CredibilityReport,
        greenwashing_report: GreenwashingReport,
        adversarial_result: AdversarialResult,
    ) -> ReportMetrics:
        hallucination_rate = self._compute_hallucination_rate(claim_assessments)
        factual_accuracy = self._compute_factual_accuracy(claim_assessments)
        policy_alignment, policy_detail = self._compute_policy_alignment(
            company_name, claim_assessments
        )
        greenwashing_index = greenwashing_report.overall_greenwashing_score
        robustness = adversarial_result.robustness_score

        anti_greenwashing = 1.0 - greenwashing_index
        overall = (
            self.METRIC_WEIGHTS["factual_accuracy"] * factual_accuracy
            + self.METRIC_WEIGHTS["policy_alignment_score"] * policy_alignment
            + self.METRIC_WEIGHTS["anti_greenwashing"] * anti_greenwashing
            + self.METRIC_WEIGHTS["robustness_score"] * robustness
        )

        grade = ReportMetrics.grade_from_score(overall)

        return ReportMetrics(
            company_name=company_name,
            hallucination_rate=hallucination_rate,
            factual_accuracy=factual_accuracy,
            policy_alignment_score=policy_alignment,
            greenwashing_index=greenwashing_index,
            robustness_score=robustness,
            overall_audit_score=overall,
            grade=grade,
            breakdown={
                "num_claims": len(claim_assessments),
                "claims_consistent": sum(
                    1 for a in claim_assessments if a.consistent_with_ipcc is True
                ),
                "claims_inconsistent": sum(
                    1 for a in claim_assessments if a.consistent_with_ipcc is False
                ),
                "greenwashing_indicators": greenwashing_report.detected_indicators,
                "credibility_label": credibility_report.credibility_label,
                "adversarial_contradictions": adversarial_result.contradictions_found,
            },
            policy_alignment_detail=policy_detail,
        )

    def _compute_hallucination_rate(self, assessments: list[ClaimAssessment]) -> float:
        if not assessments:
            return 0.0
        hallucinated = sum(
            1 for a in assessments
            if a.consistent_with_ipcc is False and a.accuracy_score < 0.3
        )
        return hallucinated / len(assessments)

    def _compute_factual_accuracy(self, assessments: list[ClaimAssessment]) -> float:
        if not assessments:
            return 0.5
        scores = [a.accuracy_score for a in assessments]
        return sum(scores) / len(scores)

    def _compute_policy_alignment(
        self, company: str, assessments: list[ClaimAssessment]
    ) -> tuple[float, dict]:
        if not assessments:
            return 0.5, {}

        claims_summary = "; ".join(
            f"{a.claim_text[:80]} [{a.alignment_level}]" for a in assessments[:8]
        )
        prompt = POLICY_ALIGNMENT_PROMPT.format(
            company=company, claims_summary=claims_summary
        )
        response = self._call_llm(prompt, max_tokens=600)
        data = parse_llm_json(response, fallback={}, label="policy_alignment")
        return float(data.get("overall_policy_alignment", 0.5)), data

    def _call_llm(self, prompt: str, max_tokens: int = 600) -> str:
        return self.llm_client.complete(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
