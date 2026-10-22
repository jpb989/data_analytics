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


def test_mar_conditional_missingness():
    """Verify that MAR missingness only affects records exceeding the condition threshold."""
    df = pd.DataFrame({
        "target": np.arange(1000, dtype=float),
        "condition": np.linspace(0, 100, 1000),
    })

    perturber = DataPerturber(random_state=42)
    corrupted = perturber.inject_mar(
        df,
        target_col="target",
        condition_col="condition",
        condition_quantile=0.5,
        missing_rate=0.40,
    )

    threshold = df["condition"].quantile(0.5)
    below_mask = df["condition"] < threshold
    above_mask = df["condition"] >= threshold

    # Rows below median must have zero missingness
    assert corrupted.loc[below_mask, "target"].isnull().sum() == 0, "Below threshold rows must not be missing"

    # Rows above median should have ~40% missingness
    above_missing_rate = corrupted.loc[above_mask, "target"].isnull().mean()
    assert abs(above_missing_rate - 0.40) < 0.05, f"Expected ~0.40 missing rate above threshold, got {above_missing_rate}"

