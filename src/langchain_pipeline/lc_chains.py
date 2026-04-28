"""
LangChain audit chains.

Builds LangChain chains for claim extraction, consistency checking,
greenwashing detection, and credibility scoring using:
  - ChatGroq (free) or HuggingFacePipeline for the LLM
  - ChatPromptTemplate for structured prompts
  - JsonOutputParser for structured JSON extraction

These chains wrap the same prompt logic as the original audit module
but express it as composable LangChain Runnables (LCEL).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ────────────────────────────── prompt templates ──────────────────────────

CLAIM_EXTRACTION_TEMPLATE = """You are a climate disclosure analyst. Extract all specific climate commitments and targets from this corporate report excerpt.

Return a JSON array only. Each item must have:
- claim_id: short unique identifier
- claim_text: the commitment (paraphrased or exact)
- claim_type: quantitative | qualitative | commitment | target
- sector: energy | transport | buildings | industry | overall | other
- numbers_mentioned: list of specific numbers/dates/percentages

Report excerpt:
{text}

Return ONLY valid JSON array, no explanation."""

CONSISTENCY_CHECK_TEMPLATE = """You are an IPCC AR6 expert auditing a corporate climate claim.

Claim: "{claim}"

IPCC AR6 benchmarks context:
{benchmark_context}

Evaluate this claim. Return JSON only:
{{
  "consistent_with_ipcc": true or false or null,
  "alignment_level": "1.5C_aligned" or "2C_aligned" or "insufficient" or "undetermined",
  "accuracy_score": 0.0 to 1.0,
  "explanation": "2-3 sentence assessment",
  "issues": ["list of concerns"],
  "benchmark_references": ["relevant IPCC targets"]
}}"""

GREENWASHING_TEMPLATE = """You are a greenwashing detection expert. Analyze this corporate climate report.

Company: {company}

Report:
{text}

Check for 8 greenwashing indicators and return JSON only:
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
  "greenwashing_label": "Minimal or Moderate or Significant or Severe",
  "summary": "2-sentence assessment",
  "red_flags": []
}}"""

CREDIBILITY_TEMPLATE = """You are a climate disclosure credibility expert.

Company: {company}
Claims summary: {claims_summary}

Report excerpt:
{report_excerpt}

Score credibility on 5 dimensions (0.0-1.0) and return JSON only:
{{
  "specificity_score": 0.0,
  "science_alignment_score": 0.0,
  "completeness_score": 0.0,
  "verifiability_score": 0.0,
  "ambition_score": 0.0,
  "overall_credibility_score": 0.0,
  "credibility_label": "High or Moderate or Low or Very Low",
  "key_strengths": [],
  "key_weaknesses": [],
  "recommendation": ""
}}"""


class LangChainAuditChain:
    """
    LangChain LCEL chains for climate audit tasks.

    Supports Groq (free) and HuggingFace as LLM backends.
    """

    def __init__(self, provider: str = "groq", api_key: str = ""):
        self.provider = provider
        self.api_key = api_key
        self._llm = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        if self.provider == "groq":
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=self.api_key,
                temperature=0.1,
            )
        elif self.provider == "huggingface":
            from langchain_huggingface import HuggingFaceEndpoint
            self._llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                huggingfacehub_api_token=self.api_key,
                temperature=0.1,
                max_new_tokens=1000,
            )
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=self.api_key,
                temperature=0.1,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        return self._llm

    def _build_chain(self, template: str):
        """Build a simple prompt → LLM → string chain."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self._get_llm() | StrOutputParser()

    def _parse_json(self, text: str) -> dict | list:
        """Extract JSON from LLM output (handles markdown code fences)."""
        text = text.strip()
        # Strip code fences
        for fence in ["```json", "```"]:
            if fence in text:
                text = text.split(fence, 1)[-1]
                if "```" in text:
                    text = text.rsplit("```", 1)[0]
                break
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {}

    # ─────────────────────────────── chain methods ────────────────────────

    def extract_claims(self, text: str) -> list[dict]:
        chain = self._build_chain(CLAIM_EXTRACTION_TEMPLATE)
        raw = chain.invoke({"text": text[:4000]})
        result = self._parse_json(raw)
        return result if isinstance(result, list) else []

    def check_consistency(self, claim: str, benchmark_context: str) -> dict:
        chain = self._build_chain(CONSISTENCY_CHECK_TEMPLATE)
        raw = chain.invoke({"claim": claim, "benchmark_context": benchmark_context})
        return self._parse_json(raw) or {
            "consistent_with_ipcc": None,
            "alignment_level": "undetermined",
            "accuracy_score": 0.5,
            "explanation": "Parse error.",
            "issues": [],
            "benchmark_references": [],
        }

    def detect_greenwashing(self, company: str, text: str) -> dict:
        chain = self._build_chain(GREENWASHING_TEMPLATE)
        raw = chain.invoke({"company": company, "text": text[:5000]})
        return self._parse_json(raw) or {}

    def score_credibility(self, company: str, claims_summary: str, report_excerpt: str) -> dict:
        chain = self._build_chain(CREDIBILITY_TEMPLATE)
        raw = chain.invoke({
            "company": company,
            "claims_summary": claims_summary,
            "report_excerpt": report_excerpt[:3000],
        })
        return self._parse_json(raw) or {}
