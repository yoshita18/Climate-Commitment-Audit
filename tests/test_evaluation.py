"""Tests for evaluation metrics and benchmarking methodology."""
from __future__ import annotations

import json
import pytest
from typing import Optional
from unittest.mock import MagicMock

from src.evaluation.metrics import EvaluationMetrics, ReportMetrics
from src.audit.consistency_checker import ClaimAssessment
from src.audit.credibility_scorer import CredibilityReport
from src.audit.greenwashing_detector import GreenwashingReport
from src.adversarial.adversarial_prompter import AdversarialResult


def make_claim_assessment(
    consistent: Optional[bool] = True,
    accuracy: float = 0.8,
    alignment: str = "1.5C_aligned",
) -> ClaimAssessment:
    return ClaimAssessment(
        claim_id="test_claim",
        claim_text="We reduce emissions 50% by 2030.",
        claim_type="quantitative",
        sector="overall",
        consistent_with_ipcc=consistent,
        alignment_level=alignment,
        accuracy_score=accuracy,
        explanation="Test explanation.",
        issues=[],
        benchmark_references=[],
    )


def make_credibility_report(label: str = "High", score: float = 0.8) -> CredibilityReport:
    return CredibilityReport(
        company_name="TestCorp",
        specificity_score=0.8,
        science_alignment_score=0.8,
        completeness_score=0.8,
        verifiability_score=0.8,
        ambition_score=0.8,
        overall_credibility_score=score,
        credibility_label=label,
        key_strengths=["Good targets"],
        key_weaknesses=[],
        recommendation="Continue.",
        num_claims_assessed=5,
    )


def make_greenwashing_report(score: float = 0.2, label: str = "Minimal") -> GreenwashingReport:
    return GreenwashingReport(
        company_name="TestCorp",
        indicators={},
        overall_greenwashing_score=score,
        greenwashing_label=label,
        summary="Clean report.",
        red_flags=[],
    )


def make_adversarial_result(robustness: float = 0.75) -> AdversarialResult:
    result = AdversarialResult(
        company_name="TestCorp",
        contradictions_found=0,
        contradictions=[],
        internal_consistency_score=0.9,
        perturbed_claims=[],
        perturbation_detection_rate=0.8,
        devil_advocate_score=0.3,
        counter_arguments=[],
        misinformation_detection_rate=0.9,
    )
    return result


class TestReportMetrics:
    def test_grade_from_score(self):
        assert ReportMetrics.grade_from_score(0.90) == "A"
        assert ReportMetrics.grade_from_score(0.75) == "B"
        assert ReportMetrics.grade_from_score(0.60) == "C"
        assert ReportMetrics.grade_from_score(0.45) == "D"
        assert ReportMetrics.grade_from_score(0.30) == "F"

    def test_to_dict_keys(self):
        m = ReportMetrics(
            company_name="Corp",
            hallucination_rate=0.05,
            factual_accuracy=0.85,
            policy_alignment_score=0.75,
            greenwashing_index=0.2,
            robustness_score=0.8,
            overall_audit_score=0.78,
            grade="B",
        )
        d = m.to_dict()
        for key in ["company_name", "hallucination_rate", "factual_accuracy",
                    "policy_alignment_score", "greenwashing_index", "overall_audit_score", "grade"]:
            assert key in d


class TestEvaluationMetrics:
    def setup_method(self):
        from src.llm.base import LLMClient

        policy_json = json.dumps({
            "paris_agreement_alignment": 0.8,
            "sbti_alignment": 0.75,
            "ipcc_1_5c_alignment": 0.85,
            "ndc_consistency": 0.7,
            "overall_policy_alignment": 0.78,
            "alignment_gaps": [],
        })

        class _MockLLM(LLMClient):
            def complete(self, prompt, system="", max_tokens=1000):
                return policy_json
            def get_provider_name(self): return "mock"
            def get_model_name(self): return "mock"

        self.evaluator = EvaluationMetrics(_MockLLM())

    def test_compute_returns_metrics(self):
        assessments = [make_claim_assessment() for _ in range(5)]
        cred = make_credibility_report()
        gw = make_greenwashing_report()
        adv = make_adversarial_result()

        metrics = self.evaluator.compute("TestCorp", assessments, cred, gw, adv)
        assert isinstance(metrics, ReportMetrics)
        assert 0.0 <= metrics.overall_audit_score <= 1.0
        assert metrics.grade in {"A", "B", "C", "D", "F"}

    def test_hallucination_rate_high_for_inconsistent_claims(self):
        assessments = [
            make_claim_assessment(consistent=False, accuracy=0.1) for _ in range(4)
        ] + [make_claim_assessment(consistent=True, accuracy=0.9)]
        rate = self.evaluator._compute_hallucination_rate(assessments)
        assert rate == 0.8

    def test_hallucination_rate_zero_for_consistent_claims(self):
        assessments = [make_claim_assessment(consistent=True, accuracy=0.9) for _ in range(5)]
        rate = self.evaluator._compute_hallucination_rate(assessments)
        assert rate == 0.0

    def test_factual_accuracy_average(self):
        assessments = [make_claim_assessment(accuracy=a) for a in [0.6, 0.8, 1.0]]
        accuracy = self.evaluator._compute_factual_accuracy(assessments)
        assert abs(accuracy - 0.8) < 1e-6

    def test_factual_accuracy_empty_list(self):
        accuracy = self.evaluator._compute_factual_accuracy([])
        assert accuracy == 0.5

    def test_greenwashing_report_detected_indicators(self):
        gw = GreenwashingReport(
            company_name="Corp",
            indicators={
                "offset_dependency": {"detected": True, "severity": "high", "evidence": ""},
                "vague_language": {"detected": False, "severity": "none", "evidence": ""},
            },
            overall_greenwashing_score=0.6,
            greenwashing_label="Significant",
            summary="",
            red_flags=[],
        )
        assert "offset_dependency" in gw.detected_indicators
        assert "vague_language" not in gw.detected_indicators

    def test_overall_score_bounded(self):
        assessments = [make_claim_assessment(consistent=True, accuracy=1.0) for _ in range(10)]
        cred = make_credibility_report(score=1.0)
        gw = make_greenwashing_report(score=0.0)
        adv = make_adversarial_result(robustness=1.0)
        metrics = self.evaluator.compute("BestCorp", assessments, cred, gw, adv)
        assert metrics.overall_audit_score <= 1.0
        assert metrics.overall_audit_score >= 0.0
