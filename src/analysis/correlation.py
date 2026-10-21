"""Statistical correlation and sensitivity analysis module.

Evaluates how strongly statistical drift metrics (PSI, KS D, JSD) and data quality defects
correlate with actual downstream model degradation (Delta F1, Delta ROC-AUC),
and quantifies the False Positive Alarm Rate of univariate statistical monitoring.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Any, Dict, List
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from src.data.ingestion import get_project_root, load_dataset_config
from src.data.preprocessor import DataPreprocessor
from src.models.evaluate import compute_metric_degradation, evaluate_binary_classifier
from src.monitoring.drift_suite import StatisticalDriftSuite
from src.quality.profiler import DataQualityProfiler


def run_correlation_study() -> Dict[str, Any]:
    """Execute end-to-end sensitivity and correlation experiments across all scenario batches."""
    root = get_project_root()
    dataset_cfg = load_dataset_config()

    num_cols = dataset_cfg["numerical_features"]
    cat_cols = dataset_cfg["categorical_features"]
    target_col = dataset_cfg["target_column"]

    # 1. Load reference assets
    train_df = pd.read_csv(root / "data" / "processed" / "train.csv")
    test_df = pd.read_csv(root / "data" / "processed" / "test.csv")

    with open(root / "data" / "reference" / "baseline_profile.json", "r", encoding="utf-8") as f:
        baseline_profile = json.load(f)

    with open(root / "data" / "reference" / "baseline_metrics.json", "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    with open(root / "data" / "reference" / "feature_importance.json", "r", encoding="utf-8") as f:
        importance_data = json.load(f)

    preprocessor = DataPreprocessor.load(root / "data" / "reference" / "preprocessor.joblib")

    # Load models
    models = {
        "logistic_regression": joblib.load(root / "models" / "logistic_regression.joblib"),
        "random_forest": joblib.load(root / "models" / "random_forest.joblib"),
        "xgboost": joblib.load(root / "models" / "xgboost.joblib"),
    }

    # Profilers and Drift Suite
    dq_profiler = DataQualityProfiler(num_cols, cat_cols)
    drift_suite = StatisticalDriftSuite(train_df, num_cols, cat_cols, baseline_profile)

    # 2. Iterate through all simulation scenario batches
    scenarios_dir = root / "data" / "simulations"
    batch_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 60)
    print("RUNNING SENSITIVITY & CORRELATION STUDY ACROSS BATCHES")
    print("=" * 60)

    for scenario_dir in sorted(scenarios_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        scenario_name = scenario_dir.name
        for batch_file in sorted(scenario_dir.glob("*.csv")):
            batch_df = pd.read_csv(batch_file)
            batch_label = batch_file.stem
            y_batch = batch_df[target_col].values

            # Compute Drift & Quality
            feature_drift = drift_suite.compute_feature_drift(batch_df)
            quality_report = dq_profiler.compare_quality_against_baseline(batch_df, baseline_profile)

            # Max & Average PSI, KS, JSD
            psi_values = [v.get("psi", 0.0) for v in feature_drift.values()]
            max_psi = float(np.max(psi_values)) if psi_values else 0.0
            avg_psi = float(np.mean(psi_values)) if psi_values else 0.0

            # Transform and evaluate with all 3 models
            X_batch_proc = preprocessor.transform(batch_df.drop(columns=[target_col], errors="ignore"))

            for model_name, model in models.items():
                proba = model.predict_proba(X_batch_proc)[:, 1]
                pred = (proba >= 0.5).astype(int)
                metrics = evaluate_binary_classifier(y_batch, pred, proba)

                base_test_metrics = baseline_metrics[model_name]["test"]
                deg = compute_metric_degradation(base_test_metrics, metrics, primary_metric="f1_macro")

                delta_f1 = deg["primary_absolute_drop"]
                delta_auc = deg["metrics"].get("roc_auc", {}).get("absolute_drop", 0.0)

                record = {
                    "scenario": scenario_name,
                    "batch": batch_label,
                    "model": model_name,
                    "max_psi": max_psi,
                    "avg_psi": avg_psi,
                    "missing_defects": quality_report["total_defects"],
                    "f1_current": metrics["f1_macro"],
                    "roc_auc_current": metrics["roc_auc"],
                    "delta_f1": delta_f1,
                    "delta_roc_auc": delta_auc,
                    "is_false_alarm": bool(max_psi >= 0.25 and delta_f1 < 0.02),
                }
                batch_records.append(record)

    df_records = pd.DataFrame(batch_records)

    # 3. Calculate Correlations
    correlations = {}
    for model_name in models.keys():
        m_df = df_records[df_records["model"] == model_name]
        sp_rho, sp_p = spearmanr(m_df["max_psi"], m_df["delta_f1"])
        kt_tau, kt_p = kendalltau(m_df["max_psi"], m_df["delta_f1"])

        false_alarms = m_df["is_false_alarm"].sum()
        total_drifted_batches = (m_df["max_psi"] >= 0.25).sum()
        false_alarm_rate = float(false_alarms / total_drifted_batches) if total_drifted_batches > 0 else 0.0

        correlations[model_name] = {
            "spearman_rho": float(round(sp_rho, 4)) if not np.isnan(sp_rho) else 0.0,
            "spearman_pvalue": float(round(sp_p, 6)) if not np.isnan(sp_p) else 1.0,
            "kendall_tau": float(round(kt_tau, 4)) if not np.isnan(kt_tau) else 0.0,
            "kendall_pvalue": float(round(kt_p, 6)) if not np.isnan(kt_p) else 1.0,
            "total_drift_alerts": int(total_drifted_batches),
            "benign_false_alarms": int(false_alarms),
            "false_alarm_rate_pct": float(round(false_alarm_rate * 100, 1)),
        }

    # Summary outputs
    summary = {
        "total_evaluated_batches": len(df_records),
        "correlations": correlations,
        "records": batch_records,
    }

    output_path = root / "data" / "reference" / "correlation_study.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EMPIRICAL CORRELATION & FALSE ALARM RATE SUMMARY:")
    print("=" * 60)
    for m_name, stats in correlations.items():
        print(f"\nModel: {m_name.upper()}")
        print(f"  - Spearman Rank Correlation (Max PSI vs Delta F1): rho = {stats['spearman_rho']:.4f} (p = {stats['spearman_pvalue']:.4e})")
        print(f"  - Kendall's Tau: tau = {stats['kendall_tau']:.4f}")
        print(f"  - Univariate Drift Alerts Triggered: {stats['total_drift_alerts']}")
        print(f"  - False Alarms (High Drift, but Delta F1 < 2%): {stats['benign_false_alarms']} ({stats['false_alarm_rate_pct']:.1f}%)")

    print(f"\n[Analysis] Full correlation study persisted to: {output_path}")
    return summary


if __name__ == "__main__":
    run_correlation_study()
