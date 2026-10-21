"""Unit tests for statistical drift suite."""

import numpy as np
import pytest
from src.monitoring.drift_suite import (
    calculate_psi,
    calculate_continuous_jsd,
    calculate_categorical_psi,
    calculate_categorical_jsd,
)


def test_psi_identical_distributions():
    """Identical distributions should have near-zero PSI."""
    rng = np.random.RandomState(42)
    sample1 = rng.normal(0, 1, 1000)
    sample2 = rng.normal(0, 1, 1000)

    psi_val = calculate_psi(sample1, sample2)
    assert psi_val < 0.10, f"Expected PSI < 0.10 for identical distributions, got {psi_val}"


def test_psi_shifted_distribution():
    """Significantly shifted distribution should trigger high PSI (> 0.25)."""
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 1000)
    shifted = rng.normal(3, 1, 1000)

    psi_val = calculate_psi(ref, shifted)
    assert psi_val >= 0.25, f"Expected PSI >= 0.25 for shifted distribution, got {psi_val}"


def test_continuous_jsd_bounded():
    """Jensen-Shannon Divergence must be strictly bounded in [0, 1]."""
    rng = np.random.RandomState(42)
    sample1 = rng.normal(10, 2, 500)
    sample2 = rng.normal(25, 5, 500)

    jsd_val = calculate_continuous_jsd(sample1, sample2)
    assert 0.0 <= jsd_val <= 1.0, f"JSD must be in [0, 1], got {jsd_val}"
