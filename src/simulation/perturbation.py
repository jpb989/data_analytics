"""Controlled data quality degradation and distribution drift perturber.

Systematically mutates tabular datasets across two primary axes:
1. Data Quality Anomalies:
   - MCAR (Missing Completely at Random)
   - MAR (Missing at Random)
   - Outliers (Gaussian noise, Cauchy heavy tails, scale multipliers)
   - Categorical Corruptions (unseen novel categories, typos)
2. Distributional Drift:
   - Sudden Covariate Shift (mean shifts by k * sigma)
   - Variance / Scale Inflation
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd


class DataPerturber:
    """Applies controlled, reproducible quality anomalies and statistical drift."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    # -------------------------------------------------------------
    # 1. DATA QUALITY PERTURBATIONS
    # -------------------------------------------------------------

    def inject_mcar(
        self,
        df: pd.DataFrame,
        columns: List[str],
        missing_rate: float,
    ) -> pd.DataFrame:
        """Inject Missing Completely At Random (MCAR) uniformly across columns."""
        corrupted = df.copy()
        n_rows = len(corrupted)

        for col in columns:
            if col not in corrupted.columns:
                continue
            mask = self.rng.rand(n_rows) < missing_rate
            corrupted.loc[mask, col] = np.nan

        return corrupted

    def inject_mar(
        self,
        df: pd.DataFrame,
        target_col: str,
        condition_col: str,
        condition_quantile: float = 0.5,
        missing_rate: float = 0.3,
    ) -> pd.DataFrame:
        """Inject Missing At Random (MAR) conditioned on another feature's values.

        Missingness occurs in `target_col` primarily when `condition_col`
        exceeds the given quantile threshold.
        """
        corrupted = df.copy()
        if target_col not in corrupted.columns or condition_col not in corrupted.columns:
            return corrupted

        threshold = corrupted[condition_col].quantile(condition_quantile)
        eligible_mask = corrupted[condition_col] >= threshold
        random_submask = self.rng.rand(len(corrupted)) < missing_rate

        final_mask = eligible_mask & random_submask
        corrupted.loc[final_mask, target_col] = np.nan
        return corrupted

    def inject_numerical_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        mode: str = "scale_multiplier",
        intensity: float = 3.0,
        fraction: float = 0.10,
    ) -> pd.DataFrame:
        """Inject extreme outliers into specified continuous columns.

        Modes:
            - 'scale_multiplier': Multiply selected values by factor (e.g. 3x or 5x).
            - 'gaussian_noise': Add N(intensity * sigma, sigma^2).
            - 'cauchy_noise': Add heavy-tailed Cauchy noise.
        """
        corrupted = df.copy()
        n_rows = len(corrupted)

        for col in columns:
            if col not in corrupted.columns:
                continue
            series = pd.to_numeric(corrupted[col], errors="coerce")
            std = float(series.std()) if series.std() > 0 else 1.0
            mean = float(series.mean())

            mask = self.rng.rand(n_rows) < fraction
            num_outliers = int(mask.sum())
            if num_outliers == 0:
                continue

            if mode == "scale_multiplier":
                corrupted.loc[mask, col] = series.loc[mask] * intensity
            elif mode == "gaussian_noise":
                noise = self.rng.normal(loc=intensity * std, scale=std, size=num_outliers)
                corrupted.loc[mask, col] = series.loc[mask] + noise
            elif mode == "cauchy_noise":
                noise = self.rng.standard_cauchy(size=num_outliers) * (intensity * std)
                corrupted.loc[mask, col] = series.loc[mask] + noise

        return corrupted

    def inject_unseen_categories(
        self,
        df: pd.DataFrame,
        columns: List[str],
        corruption_rate: float = 0.15,
        synthetic_label: str = "NOVEL_CATEGORY_X",
    ) -> pd.DataFrame:
        """Replace existing categorical levels with novel, previously unseen category tokens."""
        corrupted = df.copy()
        n_rows = len(corrupted)

        for col in columns:
            if col not in corrupted.columns:
                continue
            mask = self.rng.rand(n_rows) < corruption_rate
            corrupted.loc[mask, col] = synthetic_label

        return corrupted

    # -------------------------------------------------------------
    # 2. DISTRIBUTIONAL DRIFT PERTURBATIONS
    # -------------------------------------------------------------

    def inject_covariate_shift(
        self,
        df: pd.DataFrame,
        columns: List[str],
        shift_sigmas: float = 1.5,
    ) -> pd.DataFrame:
        """Shift the mean of continuous features by k standard deviations (Sudden / Mean Drift)."""
        corrupted = df.copy()
        for col in columns:
            if col not in corrupted.columns:
                continue
            series = pd.to_numeric(corrupted[col], errors="coerce")
            std = float(series.std()) if series.std() > 0 else 1.0
            shift_amount = shift_sigmas * std
            corrupted[col] = series + shift_amount
        return corrupted

    def inject_variance_inflation(
        self,
        df: pd.DataFrame,
        columns: List[str],
        scale_factor: float = 2.0,
    ) -> pd.DataFrame:
        """Inflate the variance/spread of continuous features without shifting the mean."""
        corrupted = df.copy()
        for col in columns:
            if col not in corrupted.columns:
                continue
            series = pd.to_numeric(corrupted[col], errors="coerce")
            mean = float(series.mean())
            corrupted[col] = mean + (series - mean) * scale_factor
        return corrupted
