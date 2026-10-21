"""Data Quality Profiler and Baseline Reference Generator.

Computes completeness, validity, uniqueness, statistical summaries, and
generates immutable JSON baseline profiles for production drift comparison.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any, Dict, List, Optional
import json
import numpy as np
import pandas as pd


class DataQualityProfiler:
    """Computes comprehensive data quality metrics and baseline statistical profiles."""

    def __init__(self, numerical_features: List[str], categorical_features: List[str]):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute top-level and feature-level data quality statistics."""
        n_rows, n_cols = df.shape
        missing_counts = df.isnull().sum()
        total_missing = int(missing_counts.sum())
        total_cells = n_rows * n_cols if n_rows > 0 else 1
        duplicate_rows = int(df.duplicated().sum()) 

        feature_profiles: Dict[str, Dict[str, Any]] = {}

        # Profile numerical features
        for col in self.numerical_features:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) == 0:
                feature_profiles[col] = {
                    "type": "numerical",
                    "missing_count": int(df[col].isnull().sum()),
                    "missing_rate": float(df[col].isnull().mean()),
                    "count": 0,
                }
                continue

            q25, q50, q75 = float(np.percentile(series, 25)), float(np.percentile(series, 50)), float(np.percentile(series, 75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]

            # 10 equal-frequency quantile bin edges for downstream PSI
            quantiles = np.linspace(0, 1, 11)
            bin_edges = [float(x) for x in np.percentile(series, quantiles * 100)]
            # ensure monotonic bin edges
            bin_edges = sorted(list(set(bin_edges)))

            feature_profiles[col] = {
                "type": "numerical",
                "missing_count": int(df[col].isnull().sum()),
                "missing_rate": float(df[col].isnull().mean()),
                "count": int(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "median": q50,
                "q25": q25,
                "q75": q75,
                "iqr": float(iqr),
                "outlier_count": int(len(outliers)),
                "outlier_rate": float(len(outliers) / len(series)),
                "bin_edges": bin_edges,
            }

        # Profile categorical features
        for col in self.categorical_features:
            if col not in df.columns:
                continue
            series = df[col].astype(str)
            val_counts = series.value_counts(dropna=False)
            total = len(series)
            frequencies = {str(k): float(v / total) for k, v in val_counts.items()}

            feature_profiles[col] = {
                "type": "categorical",
                "missing_count": int(df[col].isnull().sum()),
                "missing_rate": float(df[col].isnull().mean()),
                "cardinality": int(series.nunique()),
                "unique_values": [str(x) for x in series.unique()],
                "frequencies": frequencies,
            }

        return {
            "row_count": int(n_rows),
            "column_count": int(n_cols),
            "total_missing_cells": total_missing,
            "overall_missing_rate": float(total_missing / total_cells),
            "duplicate_rows": duplicate_rows,
            "duplicate_rate": float(duplicate_rows / n_rows) if n_rows > 0 else 0.0,
            "features": feature_profiles,
        }

    def generate_reference_profile(self, train_df: pd.DataFrame, output_path: Path) -> Dict[str, Any]:
        """Generate and save baseline reference profile artifact."""
        profile = self.profile_dataset(train_df)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        print(f"[Profiler] Baseline reference profile saved to: {output_path}")
        return profile

    def compare_quality_against_baseline(
        self, current_df: pd.DataFrame, baseline_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect data quality defects in current batch compared to baseline."""
        current_profile = self.profile_dataset(current_df)
        defects = []
        feature_reports = {}

        base_features = baseline_profile.get("features", {})
        curr_features = current_profile.get("features", {})

        for col, base_meta in base_features.items():
            curr_meta = curr_features.get(col, {})
            f_report = {
                "missing_rate": curr_meta.get("missing_rate", 0.0),
                "base_missing_rate": base_meta.get("missing_rate", 0.0),
                "defects": [],
            }

            # Check missingness inflation (> 5% increase)
            delta_missing = f_report["missing_rate"] - f_report["base_missing_rate"]
            if delta_missing >= 0.05:
                defect_msg = f"Missingness rate spiked by {delta_missing*100:.1f}%"
                defects.append({"feature": col, "type": "MISSINGNESS_SPIKE", "message": defect_msg})
                f_report["defects"].append(defect_msg)

            # Check unseen categories for categorical features
            if base_meta.get("type") == "categorical":
                base_cats = set(base_meta.get("unique_values", []))
                curr_cats = set(curr_meta.get("unique_values", []))
                unseen = curr_cats - base_cats - {"nan", "None"}
                if unseen:
                    defect_msg = f"Detected {len(unseen)} unseen categories: {list(unseen)[:3]}"
                    defects.append({"feature": col, "type": "UNSEEN_CATEGORIES", "message": defect_msg})
                    f_report["defects"].append(defect_msg)

            # Check extreme outlier rate increase for numerical features
            if base_meta.get("type") == "numerical":
                base_outlier = base_meta.get("outlier_rate", 0.0)
                curr_outlier = curr_meta.get("outlier_rate", 0.0)
                if curr_outlier - base_outlier >= 0.10:
                    defect_msg = f"Outlier rate increased to {curr_outlier*100:.1f}% (baseline: {base_outlier*100:.1f}%)"
                    defects.append({"feature": col, "type": "OUTLIER_SURGE", "message": defect_msg})
                    f_report["defects"].append(defect_msg)

            feature_reports[col] = f_report

        return {
            "current_rows": current_profile["row_count"],
            "total_defects": len(defects),
            "defect_list": defects,
            "feature_reports": feature_reports,
            "overall_missing_rate": current_profile["overall_missing_rate"],
        }


if __name__ == "__main__":
    from src.data.ingestion import load_dataset_config, get_project_root

    root = get_project_root()
    cfg = load_dataset_config()
    train_df = pd.read_csv(root / "data" / "processed" / "train.csv")

    profiler = DataQualityProfiler(
        numerical_features=cfg["numerical_features"],
        categorical_features=cfg["categorical_features"],
    )
    ref_path = root / "data" / "reference" / "baseline_profile.json"
    profiler.generate_reference_profile(train_df, ref_path)

