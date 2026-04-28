from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field

from ..llm.base import LLMClient
from ..llm.json_utils import parse_llm_json

logger = logging.getLogger(__name__)

GREENWASHING_PROMPT = """\
You are an expert in detecting corporate greenwashing in climate disclosures. Analyze the following report for greenwashing signals.

Company: {company}

Report text:
\"\"\"
{text}
\"\"\"

Check for these specific greenwashing indicators:

1. **offset_dependency**: Does the company rely primarily on carbon offsets rather than actual emission reductions?
2. **scope3_omission**: Are Scope 3 (value chain) emissions excluded or understated?
3. **vague_language**: Are commitments vague without specific numbers or timelines?
4. **cherry_picked_baseline**: Is there evidence of choosing a favorable baseline year?
5. **distant_targets**: Only long-term targets (2050) with no meaningful 2030 interim milestones?
6. **unproven_technology**: Heavy reliance on CCS, DAC, or other unproven technologies at scale?
7. **misleading_framing**: Use of misleading comparisons, relative vs absolute targets, etc.?
8. **marketing_language**: Excessive aspirational marketing language vs. concrete commitments?

For each indicator, provide:
- detected: true/false
- severity: "high" | "medium" | "low" | "none"
- evidence: specific quote or pattern from the text (max 150 chars)

Also provide:
- overall_greenwashing_score: 0.0 (clean) to 1.0 (severe greenwashing)
- greenwashing_label: "Minimal" | "Moderate" | "Significant" | "Severe"
- summary: 2-sentence overall assessment
- red_flags: list of the most serious concerns

Respond ONLY with valid JSON:
{{
  "indicators": {{
    "offset_dependency": {{"detected": false, "severity": "none", "evidence": ""}},
    "scope3_omission": {{"detected": false, "severity": "none", "evidence": ""}},
    "vague_language": {{"detected": false, "severity": "none", "evidence": ""}},
    "cherry_picked_baseline": {{"detected": false, "severity": "none", "evidence": ""}},
    "distant_targets": {{"detected": false, "severity": "none", "evidence": ""}},
    "unproven_technology": {{"detected": false, "severity": "none", "evidence": ""}},
    "misleading_framing": {{"detected": false, "severity": "none", "evidence": ""}},
    "marketing_language": {{"detected": false, "severity": "none", "evidence": ""}}
  }},
  "overall_greenwashing_score": 0.0,
  "greenwashing_label": "",
  "summary": "",
  "red_flags": []
}}"""


@dataclass
class GreenwashingReport:
    company_name: str
    indicators: dict[str, dict]
    overall_greenwashing_score: float
    greenwashing_label: str
    summary: str
    red_flags: list[str] = field(default_factory=list)

    @property
    def detected_indicators(self) -> list[str]:
        return [k for k, v in self.indicators.items() if v.get("detected", False)]

    @property
    def high_severity_indicators(self) -> list[str]:
        return [k for k, v in self.indicators.items() if v.get("severity") == "high"]

    def is_greenwashing(self, threshold: float = 0.5) -> bool:
        return self.overall_greenwashing_score >= threshold


class GreenwashingDetector:
    """Detects greenwashing patterns in corporate climate disclosures."""

    SYSTEM_PROMPT = (
        "You are an expert in ESG disclosure analysis and greenwashing detection. "
        "You are objective, rigorous, and always respond with valid JSON."
    )

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def detect(self, company_name: str, report_text: str) -> GreenwashingReport:
        prompt = GREENWASHING_PROMPT.format(
            company=company_name,
            text=report_text[:5000],
        )
        response = self._call_llm(prompt)
        data = parse_llm_json(response, fallback=None, label="greenwashing_detector") or self._default_response()

        return GreenwashingReport(
            company_name=company_name,
            indicators=data.get("indicators", {}),
            overall_greenwashing_score=float(data.get("overall_greenwashing_score", 0.5)),
            greenwashing_label=data.get("greenwashing_label", "Moderate"),
            summary=data.get("summary", ""),
            red_flags=data.get("red_flags", []),
        )

    @staticmethod
    def _default_response() -> dict:
        return {
            "indicators": {},
            "overall_greenwashing_score": 0.5,
            "greenwashing_label": "Moderate",
            "summary": "Assessment unavailable.",
            "red_flags": [],
        }

    def _call_llm(self, prompt: str) -> str:
        return self.llm_client.complete(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            max_tokens=1500,
        )
