"""Unit tests for controlled perturbation module."""

import numpy as np
import pandas as pd
import pytest
from src.simulation.perturbation import DataPerturber


def test_mcar_missingness_rate():
    """Verify that MCAR missingness injection matches target rate within tolerance."""
    df = pd.DataFrame({
        "num1": np.arange(1000, dtype=float),
        "num2": np.arange(1000, dtype=float),
    })

    perturber = DataPerturber(random_state=42)
    corrupted = perturber.inject_mcar(df, ["num1"], missing_rate=0.20)

    actual_rate = corrupted["num1"].isnull().mean()
    assert abs(actual_rate - 0.20) < 0.03, f"Expected ~0.20 missing rate, got {actual_rate}"
    assert corrupted["num2"].isnull().mean() == 0.0, "Untargeted column should not have missing values"


def test_covariate_shift():
    """Verify that covariate shift shifts feature mean by approximately k * sigma."""
    df = pd.DataFrame({
        "feature_x": np.random.RandomState(42).normal(10.0, 2.0, 2000),
    })

    perturber = DataPerturber(random_state=42)
    orig_mean = df["feature_x"].mean()
    orig_std = df["feature_x"].std()

    shifted = perturber.inject_covariate_shift(df, ["feature_x"], shift_sigmas=1.5)
    new_mean = shifted["feature_x"].mean()

    expected_shift = 1.5 * orig_std
    actual_shift = new_mean - orig_mean
    assert abs(actual_shift - expected_shift) < 0.1, f"Expected shift ~{expected_shift}, got {actual_shift}"
