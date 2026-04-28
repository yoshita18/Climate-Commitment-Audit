from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field

from ..llm.base import LLMClient
from ..llm.json_utils import parse_llm_json
from .consistency_checker import ClaimAssessment

logger = logging.getLogger(__name__)

CREDIBILITY_PROMPT = """\
You are a climate disclosure credibility expert. Evaluate the overall credibility of the following corporate climate report based on the claim assessments provided.

Company: {company}
Number of claims assessed: {num_claims}

Claim Assessment Summary:
{claim_summary}

Full report excerpt (first 3000 chars):
\"\"\"
{report_excerpt}
\"\"\"

Score the report across these dimensions (0.0 = lowest, 1.0 = highest):
1. **specificity_score**: Are targets specific with numbers, timelines, and baselines?
2. **science_alignment_score**: How well do targets align with IPCC 1.5°C pathways?
3. **completeness_score**: Are all emission scopes (1, 2, 3) covered? Are interim targets included?
4. **verifiability_score**: Are claims backed by third-party verification or auditable data?
5. **ambition_score**: Is the overall level of ambition adequate given climate science?

Also provide:
- overall_credibility_score: weighted average of above (0.0-1.0)
- key_strengths: list of up to 3 strong points
- key_weaknesses: list of up to 3 major gaps
- credibility_label: "High" | "Moderate" | "Low" | "Very Low"
- recommendation: one-sentence action for improvement

Respond ONLY with valid JSON matching this structure:
{{
  "specificity_score": 0.0,
  "science_alignment_score": 0.0,
  "completeness_score": 0.0,
  "verifiability_score": 0.0,
  "ambition_score": 0.0,
  "overall_credibility_score": 0.0,
  "credibility_label": "",
  "key_strengths": [],
  "key_weaknesses": [],
  "recommendation": ""
}}"""


@dataclass
class CredibilityReport:
    company_name: str
    specificity_score: float
    science_alignment_score: float
    completeness_score: float
    verifiability_score: float
    ambition_score: float
    overall_credibility_score: float
    credibility_label: str
    key_strengths: list[str] = field(default_factory=list)
    key_weaknesses: list[str] = field(default_factory=list)
    recommendation: str = ""
    num_claims_assessed: int = 0

    @property
    def scores_dict(self) -> dict:
        return {
            "specificity": self.specificity_score,
            "science_alignment": self.science_alignment_score,
            "completeness": self.completeness_score,
            "verifiability": self.verifiability_score,
            "ambition": self.ambition_score,
            "overall": self.overall_credibility_score,
        }


class CredibilityScorer:
    """Produces a multi-dimensional credibility score for climate reports."""

    SYSTEM_PROMPT = "You are a climate disclosure auditor. Respond only with valid JSON."

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def score(
        self,
        company_name: str,
        report_text: str,
        claim_assessments: list[ClaimAssessment],
    ) -> CredibilityReport:
        claim_summary = self._summarize_assessments(claim_assessments)

        prompt = CREDIBILITY_PROMPT.format(
            company=company_name,
            num_claims=len(claim_assessments),
            claim_summary=claim_summary,
            report_excerpt=report_text[:3000],
        )

        response = self._call_llm(prompt)
        data = parse_llm_json(response, fallback=None, label="credibility_scorer") or self._default_scores()

        return CredibilityReport(
            company_name=company_name,
            specificity_score=float(data.get("specificity_score", 0.5)),
            science_alignment_score=float(data.get("science_alignment_score", 0.5)),
            completeness_score=float(data.get("completeness_score", 0.5)),
            verifiability_score=float(data.get("verifiability_score", 0.5)),
            ambition_score=float(data.get("ambition_score", 0.5)),
            overall_credibility_score=float(data.get("overall_credibility_score", 0.5)),
            credibility_label=data.get("credibility_label", "Moderate"),
            key_strengths=data.get("key_strengths", []),
            key_weaknesses=data.get("key_weaknesses", []),
            recommendation=data.get("recommendation", ""),
            num_claims_assessed=len(claim_assessments),
        )

    def _summarize_assessments(self, assessments: list[ClaimAssessment]) -> str:
        if not assessments:
            return "No claims extracted."
        lines = []
        for a in assessments:
            consistent = "Yes" if a.consistent_with_ipcc else ("No" if a.consistent_with_ipcc is False else "Unclear")
            lines.append(
                f"- [{a.claim_type.upper()}] {a.claim_text[:120]}... "
                f"| IPCC Consistent: {consistent} | Alignment: {a.alignment_level} "
                f"| Accuracy: {a.accuracy_score:.2f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _default_scores() -> dict:
        return {
            "specificity_score": 0.5,
            "science_alignment_score": 0.5,
            "completeness_score": 0.5,
            "verifiability_score": 0.5,
            "ambition_score": 0.5,
            "overall_credibility_score": 0.5,
            "credibility_label": "Moderate",
            "key_strengths": [],
            "key_weaknesses": ["Could not assess automatically"],
            "recommendation": "Manual review required.",
        }

    def _call_llm(self, prompt: str) -> str:
        return self.llm_client.complete(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            max_tokens=1000,
        )
