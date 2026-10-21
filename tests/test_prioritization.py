"""Unit tests for prioritization and triage engine."""

import pytest
from src.monitoring.prioritization import AlertPrioritizationEngine


def test_prioritization_critical_vs_benign():
    """Verify that high drift on high importance leads to P1 Critical, while low importance leads to P3 Benign."""
    importances = {
        "top_feature": 0.25,
        "low_feature": 0.02,
    }
    engine = AlertPrioritizationEngine(importances)

    # 1. Critical scenario
    feature_drift_crit = {
        "top_feature": {"psi": 0.45, "severity": "HIGH"},
        "low_feature": {"psi": 0.02, "severity": "STABLE"},
    }
    triage_crit = engine.triage_batch(feature_drift_crit, [], delta_f1=0.12, delta_roc_auc=0.10)
    assert triage_crit["system_priority"] == "P1_CRITICAL"
    assert triage_crit["suppressed_false_alarm"] is False

    # 2. Benign scenario
    feature_drift_benign = {
        "top_feature": {"psi": 0.02, "severity": "STABLE"},
        "low_feature": {"psi": 0.45, "severity": "HIGH"},
    }
    triage_benign = engine.triage_batch(feature_drift_benign, [], delta_f1=0.005, delta_roc_auc=0.003)
    assert triage_benign["system_priority"] == "P3_BENIGN_DRIFT"
    assert triage_benign["suppressed_false_alarm"] is True
