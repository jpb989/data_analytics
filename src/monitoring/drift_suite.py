"""Statistical drift metric suite for tabular machine learning pipelines.

Implements industry-standard univariate drift tests:
- Continuous: Kolmogorov-Smirnov (KS) test, Population Stability Index (PSI),
  Jensen-Shannon Divergence (JSD), Wasserstein distance.
- Categorical: Chi-Square Goodness-of-Fit, Categorical PSI, Jensen-Shannon Divergence.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, ks_2samp, wasserstein_distance


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    bin_edges: Optional[List[float]] = None,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current samples.

    PSI = sum((P_b - Q_b) * ln((P_b + eps) / (Q_b + eps)))
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    if bin_edges is None:
        # Compute quantile bins from reference
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(ref_clean, quantiles)
        bin_edges = sorted(list(set(bin_edges)))
        if len(bin_edges) < 2:
            return 0.0

    # Ensure outermost edges cover the full support
    edges = np.array(bin_edges, dtype=float)
    edges[0] = min(edges[0], float(np.min(ref_clean)), float(np.min(curr_clean))) - 1e-5
    edges[-1] = max(edges[-1], float(np.max(ref_clean)), float(np.max(curr_clean))) + 1e-5

    ref_counts, _ = np.histogram(ref_clean, bins=edges)
    curr_counts, _ = np.histogram(curr_clean, bins=edges)

    ref_pct = ref_counts / len(ref_clean)
    curr_pct = curr_counts / len(curr_clean)

    # Smooth zero probabilities
    p = np.where(ref_pct == 0, epsilon, ref_pct)
    q = np.where(curr_pct == 0, epsilon, curr_pct)

    psi_val = np.sum((p - q) * np.log(p / q))
    return float(round(psi_val, 4))


def calculate_categorical_psi(
    ref_freqs: Dict[str, float],
    curr_series: pd.Series,
    epsilon: float = 1e-4,
) -> float:
    """Compute PSI for categorical distributions using reference frequency dict."""
    total_curr = len(curr_series)
    if total_curr == 0:
        return 0.0

    curr_counts = curr_series.value_counts(dropna=False).to_dict()
    curr_freqs = {str(k): float(v / total_curr) for k, v in curr_counts.items()}

    all_categories = set(ref_freqs.keys()) | set(curr_freqs.keys())
    psi_sum = 0.0

    for cat in all_categories:
        p = ref_freqs.get(cat, epsilon)
        q = curr_freqs.get(cat, epsilon)
        p = max(p, epsilon)
        q = max(q, epsilon)
        psi_sum += (p - q) * np.log(p / q)

    return float(round(psi_sum, 4))


def calculate_continuous_jsd(reference: np.ndarray, current: np.ndarray, num_bins: int = 15) -> float:
    """Calculate Jensen-Shannon Divergence [0, 1] for continuous features."""
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    global_min = min(float(np.min(ref_clean)), float(np.min(curr_clean)))
    global_max = max(float(np.max(ref_clean)), float(np.max(curr_clean)))
    if global_min == global_max:
        return 0.0

    bins = np.linspace(global_min, global_max, num_bins + 1)
    p, _ = np.histogram(ref_clean, bins=bins, density=True)
    q, _ = np.histogram(curr_clean, bins=bins, density=True)

    # Convert to probabilities
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q

    js_dist = jensenshannon(p, q, base=2)
    return float(round(js_dist, 4)) if not np.isnan(js_dist) else 0.0


def calculate_categorical_jsd(ref_freqs: Dict[str, float], curr_series: pd.Series) -> float:
    """Calculate Jensen-Shannon Divergence [0, 1] for categorical features."""
    total_curr = len(curr_series)
    if total_curr == 0:
        return 0.0

    curr_counts = curr_series.value_counts(dropna=False).to_dict()
    curr_freqs = {str(k): float(v / total_curr) for k, v in curr_counts.items()}
    all_categories = sorted(list(set(ref_freqs.keys()) | set(curr_freqs.keys())))

    p = np.array([ref_freqs.get(c, 1e-6) for c in all_categories])
    q = np.array([curr_freqs.get(c, 1e-6) for c in all_categories])

    p /= np.sum(p)
    q /= np.sum(q)

    js_dist = jensenshannon(p, q, base=2)
    return float(round(js_dist, 4)) if not np.isnan(js_dist) else 0.0


class StatisticalDriftSuite:
    """Orchestrates comprehensive drift evaluation across numerical and categorical features."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str],
        baseline_profile: Optional[Dict[str, Any]] = None,
    ):
        self.ref_df = reference_df
        self.num_features = numerical_features
        self.cat_features = categorical_features
        self.baseline_profile = baseline_profile or {}

    def compute_feature_drift(self, curr_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute statistical drift metrics for each individual feature."""
        results = {}

        # 1. Numerical Features
        for col in self.num_features:
            if col not in self.ref_df.columns or col not in curr_df.columns:
                continue

            ref_vals = pd.to_numeric(self.ref_df[col], errors="coerce").values
            curr_vals = pd.to_numeric(curr_df[col], errors="coerce").values

            # KS-Test
            ks_res = ks_2samp(ref_vals[~np.isnan(ref_vals)], curr_vals[~np.isnan(curr_vals)])
            ks_stat = float(round(ks_res.statistic, 4))
            ks_p = float(round(ks_res.pvalue, 6))

            # PSI
            cached_bins = self.baseline_profile.get("features", {}).get(col, {}).get("bin_edges")
            psi_val = calculate_psi(ref_vals, curr_vals, bin_edges=cached_bins)

            # JSD
            jsd_val = calculate_continuous_jsd(ref_vals, curr_vals)

            # Wasserstein Distance
            w_dist = float(
                round(wasserstein_distance(ref_vals[~np.isnan(ref_vals)], curr_vals[~np.isnan(curr_vals)]), 4)
            )

            # Drift Severity Classification
            is_drifted = (psi_val >= 0.25) or (ks_stat >= 0.15 and ks_p < 0.01)

            results[col] = {
                "type": "numerical",
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "psi": psi_val,
                "jsd": jsd_val,
                "wasserstein_distance": w_dist,
                "is_drifted": bool(is_drifted),
                "severity": "HIGH" if psi_val >= 0.25 else ("MODERATE" if psi_val >= 0.10 else "STABLE"),
            }

        # 2. Categorical Features
        for col in self.cat_features:
            if col not in self.ref_df.columns or col not in curr_df.columns:
                continue

            ref_freqs = self.baseline_profile.get("features", {}).get(col, {}).get("frequencies")
            if not ref_freqs:
                ref_series = self.ref_df[col].astype(str)
                ref_freqs = {str(k): float(v / len(ref_series)) for k, v in ref_series.value_counts().items()}

            curr_series = curr_df[col].astype(str)

            # Categorical PSI
            psi_val = calculate_categorical_psi(ref_freqs, curr_series)

            # Categorical JSD
            jsd_val = calculate_categorical_jsd(ref_freqs, curr_series)

            # Chi-Square Test
            all_cats = list(ref_freqs.keys())
            curr_counts = curr_series.value_counts().to_dict()
            total_curr = len(curr_series)
            observed = [curr_counts.get(c, 0) for c in all_cats]
            expected = [max(1.0, ref_freqs[c] * total_curr) for c in all_cats]

            # Normalize sums to match exactly
            sum_obs = sum(observed)
            sum_exp = sum(expected)
            if sum_exp > 0:
                expected = [e * (sum_obs / sum_exp) for e in expected]

            try:
                chi2_stat, chi2_p = chisquare(observed, f_exp=expected)
                chi2_stat = float(round(chi2_stat, 4))
                chi2_p = float(round(chi2_p, 6))
            except Exception:
                chi2_stat, chi2_p = 0.0, 1.0

            is_drifted = (psi_val >= 0.25) or (chi2_p < 0.01 and chi2_stat > 15.0)

            results[col] = {
                "type": "categorical",
                "psi": psi_val,
                "jsd": jsd_val,
                "chi2_statistic": chi2_stat,
                "chi2_pvalue": chi2_p,
                "is_drifted": bool(is_drifted),
                "severity": "HIGH" if psi_val >= 0.25 else ("MODERATE" if psi_val >= 0.10 else "STABLE"),
            }

        return results
