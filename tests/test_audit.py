"""Tests for audit components using mocked LLM responses."""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.audit.consistency_checker import ConsistencyChecker, ClaimAssessment
from src.audit.credibility_scorer import CredibilityScorer, CredibilityReport
from src.audit.greenwashing_detector import GreenwashingDetector, GreenwashingReport


def make_mock_client(response_text: str):
    """Create a mock LLMClient whose complete() returns a fixed string."""
    from src.llm.base import LLMClient

    class _MockLLM(LLMClient):
        def complete(self, prompt, system="", max_tokens=1000):
            return response_text
        def get_provider_name(self):
            return "mock"
        def get_model_name(self):
            return "mock-model"

    return _MockLLM()


def make_mock_retriever(hits=None):
    mock = MagicMock()
    mock.retrieve_benchmarks.return_value = hits or [
        {"content": "IPCC AR6: CO2 must fall 43% by 2030.", "similarity": 0.9, "metadata": {}}
    ]
    mock.format_context.return_value = "IPCC AR6: CO2 must fall 43% by 2030."
    return mock


class TestConsistencyChecker:
    def test_extract_claims_valid_json(self):
        claims_json = json.dumps([
            {
                "claim_id": "c1",
                "claim_text": "We will reduce CO2 by 50% by 2030.",
                "claim_type": "quantitative",
                "sector": "overall",
                "numbers_mentioned": ["50%", "2030"],
            }
        ])
        client = make_mock_client(claims_json)
        retriever = make_mock_retriever()
        checker = ConsistencyChecker(client, retriever)
        claims = checker.extract_claims("We will reduce CO2 by 50% by 2030.")
        assert len(claims) == 1
        assert claims[0]["claim_id"] == "c1"

    def test_extract_claims_handles_bad_json(self):
        client = make_mock_client("not valid json at all")
        retriever = make_mock_retriever()
        checker = ConsistencyChecker(client, retriever)
        claims = checker.extract_claims("some text")
        assert claims == []

    def test_assess_claim_returns_assessment(self):
        assessment_json = json.dumps({
            "consistent_with_ipcc": True,
            "alignment_level": "1.5C_aligned",
            "accuracy_score": 0.85,
            "explanation": "This target aligns well with IPCC guidance.",
            "issues": [],
            "benchmark_references": ["global_1.5c_co2_2030"],
        })
        client = make_mock_client(assessment_json)
        retriever = make_mock_retriever()
        checker = ConsistencyChecker(client, retriever)

        claim = {
            "claim_id": "c1",
            "claim_text": "Reduce Scope 1+2 by 50% by 2030.",
            "claim_type": "quantitative",
            "sector": "overall",
        }
        result = checker.assess_claim(claim)
        assert isinstance(result, ClaimAssessment)
        assert result.consistent_with_ipcc is True
        assert result.alignment_level == "1.5C_aligned"
        assert result.accuracy_score == 0.85

    def test_assess_claim_handles_bad_json(self):
        client = make_mock_client("{ broken json")
        retriever = make_mock_retriever()
        checker = ConsistencyChecker(client, retriever)
        claim = {"claim_id": "c1", "claim_text": "some claim", "claim_type": "qualitative", "sector": "overall"}
        result = checker.assess_claim(claim)
        assert isinstance(result, ClaimAssessment)
        assert result.accuracy_score == 0.5  # default


class TestCredibilityScorer:
    def test_score_returns_report(self):
        score_json = json.dumps({
            "specificity_score": 0.8,
            "science_alignment_score": 0.75,
            "completeness_score": 0.7,
            "verifiability_score": 0.85,
            "ambition_score": 0.9,
            "overall_credibility_score": 0.8,
            "credibility_label": "High",
            "key_strengths": ["Science-aligned targets", "Third-party verified"],
            "key_weaknesses": ["Scope 3 detail limited"],
            "recommendation": "Strengthen Scope 3 interim milestones.",
        })
        client = make_mock_client(score_json)
        scorer = CredibilityScorer(client)

        mock_assessment = MagicMock()
        mock_assessment.claim_text = "Reduce emissions 50% by 2030."
        mock_assessment.claim_type = "quantitative"
        mock_assessment.consistent_with_ipcc = True
        mock_assessment.alignment_level = "1.5C_aligned"
        mock_assessment.accuracy_score = 0.85

        report = scorer.score("Test Corp", "some report text", [mock_assessment])
        assert isinstance(report, CredibilityReport)
        assert report.credibility_label == "High"
        assert report.overall_credibility_score == 0.8
        assert len(report.key_strengths) == 2

    def test_scores_dict_property(self):
        client = make_mock_client(json.dumps({
            "specificity_score": 0.5, "science_alignment_score": 0.6,
            "completeness_score": 0.7, "verifiability_score": 0.8,
            "ambition_score": 0.9, "overall_credibility_score": 0.7,
            "credibility_label": "Moderate", "key_strengths": [], "key_weaknesses": [],
            "recommendation": "",
        }))
        scorer = CredibilityScorer(client)
        report = scorer.score("Corp", "text", [])
        scores = report.scores_dict
        assert "overall" in scores
        assert len(scores) == 6


class TestGreenwashingDetector:
    def test_detect_greenwashing(self):
        gw_json = json.dumps({
            "indicators": {
                "offset_dependency": {"detected": True, "severity": "high", "evidence": "100% offset for operations"},
                "scope3_omission": {"detected": True, "severity": "high", "evidence": "No Scope 3 reported"},
                "vague_language": {"detected": False, "severity": "none", "evidence": ""},
                "cherry_picked_baseline": {"detected": False, "severity": "none", "evidence": ""},
                "distant_targets": {"detected": False, "severity": "none", "evidence": ""},
                "unproven_technology": {"detected": False, "severity": "none", "evidence": ""},
                "misleading_framing": {"detected": False, "severity": "none", "evidence": ""},
                "marketing_language": {"detected": True, "severity": "medium", "evidence": "force for good"},
            },
            "overall_greenwashing_score": 0.72,
            "greenwashing_label": "Significant",
            "summary": "Heavy offset reliance without addressing supply chain.",
            "red_flags": ["offset_dependency", "scope3_omission"],
        })
        client = make_mock_client(gw_json)
        detector = GreenwashingDetector(client)
        report = detector.detect("FashionCo", "some report text")

        assert isinstance(report, GreenwashingReport)
        assert report.overall_greenwashing_score == 0.72
        assert report.greenwashing_label == "Significant"
        assert "offset_dependency" in report.detected_indicators
        assert report.is_greenwashing(threshold=0.5) is True

    def test_not_greenwashing(self):
        clean_json = json.dumps({
            "indicators": {k: {"detected": False, "severity": "none", "evidence": ""} for k in [
                "offset_dependency", "scope3_omission", "vague_language",
                "cherry_picked_baseline", "distant_targets", "unproven_technology",
                "misleading_framing", "marketing_language",
            ]},
            "overall_greenwashing_score": 0.1,
            "greenwashing_label": "Minimal",
            "summary": "Strong, credible climate commitments.",
            "red_flags": [],
        })
        client = make_mock_client(clean_json)
        detector = GreenwashingDetector(client)
        report = detector.detect("GoodCorp", "credible report")
        assert report.is_greenwashing(threshold=0.5) is False
        assert report.detected_indicators == []
