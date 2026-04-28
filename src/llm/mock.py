"""
Mock LLM client for demos and offline testing — no API key required.

Returns realistic-looking JSON stubs for every prompt type used in the
audit pipeline.  The response is selected by sniffing keywords in the
prompt text, so the full pipeline runs end-to-end without network access.

Usage:
    from src.llm.mock import MockLLMClient
    llm = MockLLMClient()
"""
from __future__ import annotations

import json
import random
from typing import Optional

from .base import LLMClient


# ── Realistic stub payloads ────────────────────────────────────────────────

_CLAIMS_STUB = json.dumps([
    {
        "claim_id": "c1",
        "claim_text": "We will reduce Scope 1 and 2 emissions by 50% by 2030 from a 2019 baseline.",
        "claim_type": "quantitative",
        "sector": "overall",
        "numbers_mentioned": ["50%", "2030"],
    },
    {
        "claim_id": "c2",
        "claim_text": "We commit to achieving net-zero across our full value chain by 2050.",
        "claim_type": "quantitative",
        "sector": "overall",
        "numbers_mentioned": ["2050"],
    },
    {
        "claim_id": "c3",
        "claim_text": "100% of our electricity will come from renewable sources by 2025.",
        "claim_type": "quantitative",
        "sector": "energy",
        "numbers_mentioned": ["100%", "2025"],
    },
])

_CONSISTENCY_STUB = json.dumps({
    "consistent_with_ipcc": True,
    "alignment_level": "1.5C_aligned",
    "accuracy_score": 0.83,
    "explanation": (
        "The 50% Scope 1+2 reduction by 2030 is broadly consistent with IPCC AR6 "
        "guidance requiring ~43-48% CO2 reduction by 2030 for a 1.5°C pathway. "
        "The target is specific, time-bound, and uses a reasonable baseline year."
    ),
    "issues": [],
    "benchmark_references": ["global_1.5c_co2_2030", "sbti_nz_standard"],
})

_CREDIBILITY_STUB = json.dumps({
    "specificity_score": 0.82,
    "science_alignment_score": 0.79,
    "completeness_score": 0.74,
    "verifiability_score": 0.80,
    "ambition_score": 0.85,
    "overall_credibility_score": 0.80,
    "credibility_label": "High",
    "key_strengths": [
        "Science-aligned interim targets",
        "Baseline year clearly defined (2019)",
        "Third-party verification mentioned",
    ],
    "key_weaknesses": [
        "Scope 3 detail limited to upstream only",
        "Interim 2025 milestone not specified",
    ],
    "recommendation": (
        "Strengthen Scope 3 coverage and add year-by-year interim milestones "
        "to improve credibility and trackability."
    ),
})

_GREENWASHING_STUB = json.dumps({
    "indicators": {
        "offset_dependency": {
            "detected": False,
            "severity": "none",
            "evidence": "",
        },
        "scope3_omission": {
            "detected": True,
            "severity": "medium",
            "evidence": "Scope 3 categories 4-15 not addressed in the report.",
        },
        "vague_language": {
            "detected": False,
            "severity": "none",
            "evidence": "",
        },
        "cherry_picked_baseline": {
            "detected": False,
            "severity": "none",
            "evidence": "",
        },
        "distant_targets": {
            "detected": False,
            "severity": "none",
            "evidence": "",
        },
        "unproven_technology": {
            "detected": False,
            "severity": "none",
            "evidence": "",
        },
        "misleading_framing": {
            "detected": False,
            "severity": "none",
            "evidence": "",
        },
        "marketing_language": {
            "detected": True,
            "severity": "low",
            "evidence": "'Leading the way to a sustainable future' — standard marketing language.",
        },
    },
    "overall_greenwashing_score": 0.22,
    "greenwashing_label": "Minimal",
    "summary": (
        "The report shows minimal greenwashing. Targets are specific and "
        "science-aligned. Minor concerns around incomplete Scope 3 coverage "
        "and generic marketing language."
    ),
    "red_flags": [],
})

_POLICY_STUB = json.dumps({
    "paris_agreement_alignment": 0.82,
    "sbti_alignment": 0.78,
    "ipcc_1_5c_alignment": 0.84,
    "ndc_consistency": 0.71,
    "overall_policy_alignment": 0.79,
    "alignment_gaps": [
        "Scope 3 categories not fully covered",
        "No explicit reference to Just Transition principles",
    ],
})

_CONTRADICTION_STUB = json.dumps({
    "contradictions_found": 0,
    "contradictions": [],
    "internal_consistency_score": 0.88,
})

_PERTURBATION_STUB = json.dumps({
    "error_detected": True,
    "error_type": "deflation",
    "explanation": "The perturbed value of 17% is inconsistent with the IPCC-required ~43% reduction.",
    "detection_confidence": 0.91,
})

_DEVILS_ADVOCATE_STUB = json.dumps({
    "counter_arguments": [
        {
            "argument": "The 2030 target covers only Scope 1+2 — Scope 3 (typically 70-90% of total footprint) is excluded.",
            "strength": "strong",
            "evidence_from_text": "Our net-zero pathway covers direct and purchased energy emissions.",
        },
        {
            "argument": "No interim annual milestones are provided, making progress impossible to track until 2030.",
            "strength": "medium",
            "evidence_from_text": "",
        },
    ],
    "devil_advocate_score": 0.31,
})

_MISINFORMATION_STUB = json.dumps({
    "error_found": True,
    "error_description": "The claim states 10% reduction is sufficient; IPCC AR6 requires 43-48% by 2030.",
    "correct_value": "43-48% CO2 reduction by 2030 (vs 2019 baseline)",
    "hallucination_detected": False,
})


# ── Keyword-based dispatch ─────────────────────────────────────────────────

_DISPATCH = [
    # (keywords_in_prompt,           stub_response)
    (["extract",  "claim", "json"],           _CLAIMS_STUB),
    (["consistent_with_ipcc", "assess"],      _CONSISTENCY_STUB),
    (["specificity_score", "credib"],         _CREDIBILITY_STUB),
    (["greenwashing", "indicator"],           _GREENWASHING_STUB),
    (["paris_agreement_alignment", "policy"], _POLICY_STUB),
    (["contradiction", "internal_consisten"], _CONTRADICTION_STUB),
    (["perturbed", "error_detected"],         _PERTURBATION_STUB),
    (["devil", "counter_argument"],           _DEVILS_ADVOCATE_STUB),
    (["injected_claim", "error_found"],       _MISINFORMATION_STUB),
]

_FALLBACK = json.dumps({
    "consistent_with_ipcc": True,
    "alignment_level": "1.5C_aligned",
    "accuracy_score": 0.75,
    "explanation": "[DEMO] Mock response — no API key required.",
    "issues": [],
    "benchmark_references": [],
})


class MockLLMClient(LLMClient):
    """
    Drop-in LLM client that returns pre-built JSON stubs.

    No API key, no network call, no latency.  Useful for:
      • Live demos without credentials
      • CI smoke tests
      • Offline development

    The response is selected by matching keywords in the prompt, so every
    stage of the pipeline (claim extraction → consistency → credibility →
    greenwashing → adversarial → policy) gets an appropriate stub.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def complete(self, prompt: str, system: str = "", max_tokens: int = 1000) -> str:
        combined = (prompt + " " + system).lower()
        for keywords, stub in _DISPATCH:
            if all(kw in combined for kw in keywords):
                return stub
        return _FALLBACK

    def get_provider_name(self) -> str:
        return "mock"

    def get_model_name(self) -> str:
        return "mock-demo-v1"

    def __repr__(self) -> str:
        return "MockLLMClient(demo — no API key needed)"
