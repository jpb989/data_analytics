"""Feature importance and SHAP attribution extraction.

Calculates model-agnostic and architecture-specific feature attributions,
aggregating one-hot encoded columns back to raw input features to establish
the foundational weighting for the Intelligent Alert Prioritization Engine.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Tuple
import json
import joblib
import numpy as np
import pandas as pd
import shap

from src.data.ingestion import get_project_root, load_dataset_config
from src.data.preprocessor import DataPreprocessor


def aggregate_encoded_importance(
    importance_per_feature: Dict[str, float],
    raw_numerical: List[str],
    raw_categorical: List[str],
) -> Dict[str, float]:
    """Aggregate one-hot encoded feature importances back to raw feature names."""
    aggregated: Dict[str, float] = {feat: 0.0 for feat in raw_numerical + raw_categorical}

    for col_name, score in importance_per_feature.items():
        matched = False
        # Exact match for numericals
        if col_name in raw_numerical:
            aggregated[col_name] += abs(score)
            matched = True
        else:
            # Check prefixes for categorical one-hot encoded features
            for cat_col in raw_categorical:
                if col_name.startswith(f"{cat_col}_") or col_name == cat_col:
                    aggregated[cat_col] += abs(score)
                    matched = True
                    break
        if not matched:
            # Fallback if any unmatched
            aggregated[col_name] = aggregated.get(col_name, 0.0) + abs(score)

    # Normalize to sum to 1.0
    total = sum(aggregated.values())
    if total > 0:
        normalized = {k: round(v / total, 4) for k, v in aggregated.items()}
    else:
        normalized = aggregated

    # Sort descending
    return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))


def extract_feature_importance() -> Dict[str, dict]:
    """Extract global importance and SHAP values across models, saving reference rankings."""
    root = get_project_root()
    dataset_cfg = load_dataset_config()

    num_cols = dataset_cfg["numerical_features"]
    cat_cols = dataset_cfg["categorical_features"]
    target_col = dataset_cfg["target_column"]

    # Load preprocessor
    prep_path = root / "data" / "reference" / "preprocessor.joblib"
    preprocessor = DataPreprocessor.load(prep_path)
    encoded_features = preprocessor.get_feature_names()

    # Load test split for SHAP evaluation
    test_df = pd.read_csv(root / "data" / "processed" / "test.csv")
    X_test_proc = preprocessor.transform(test_df.drop(columns=[target_col], errors="ignore"))

    # Sample subset for fast SHAP computation
    sample_indices = np.random.RandomState(42).choice(len(X_test_proc), size=min(300, len(X_test_proc)), replace=False)
    X_shap_sample = X_test_proc[sample_indices]

    models_dir = root / "models"
    importance_summary: Dict[str, dict] = {}

    # 1. XGBoost Feature Importance & SHAP
    xgb_path = models_dir / "xgboost.joblib"
    if xgb_path.exists():
        xgb_model = joblib.load(xgb_path)
        # Tree Gain
        gains = xgb_model.feature_importances_
        gain_dict = {feat: float(gain) for feat, gain in zip(encoded_features, gains)}
        xgb_gain_agg = aggregate_encoded_importance(gain_dict, num_cols, cat_cols)

        # SHAP
        try:
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_shap_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            shap_dict = {feat: float(s) for feat, s in zip(encoded_features, mean_abs_shap)}
            xgb_shap_agg = aggregate_encoded_importance(shap_dict, num_cols, cat_cols)
        except Exception as e:
            print(f"[SHAP] Fallback for XGBoost: {e}")
            xgb_shap_agg = xgb_gain_agg

        importance_summary["xgboost"] = {
            "native_importance": xgb_gain_agg,
            "shap_importance": xgb_shap_agg,
        }

    # 2. Random Forest Importance & SHAP
    rf_path = models_dir / "random_forest.joblib"
    if rf_path.exists():
        rf_model = joblib.load(rf_path)
        rf_importances = rf_model.feature_importances_
        rf_dict = {feat: float(score) for feat, score in zip(encoded_features, rf_importances)}
        rf_gain_agg = aggregate_encoded_importance(rf_dict, num_cols, cat_cols)

        importance_summary["random_forest"] = {
            "native_importance": rf_gain_agg,
            "shap_importance": rf_gain_agg,  # Gini importance proxy
        }

    # 3. Logistic Regression Coefficients
    lr_path = models_dir / "logistic_regression.joblib"
    if lr_path.exists():
        lr_model = joblib.load(lr_path)
        coefs = np.abs(lr_model.coef_[0])
        lr_dict = {feat: float(c) for feat, c in zip(encoded_features, coefs)}
        lr_agg = aggregate_encoded_importance(lr_dict, num_cols, cat_cols)

        importance_summary["logistic_regression"] = {
            "native_importance": lr_agg,
            "shap_importance": lr_agg,
        }

    # Compute Ensemble Consensus Importance (Weighted average across models)
    # Weights reflect model benchmark accuracy and attribution rigor:
    # - XGBoost (50%): Gradient boosting with axiomatic TreeSHAP attributions
    # - Random Forest (30%): Bagged ensemble with non-linear Gini impurity
    # - Logistic Regression (20%): Standardized linear main-effect coefficients
    model_weights = {
        "xgboost": 0.50,
        "random_forest": 0.30,
        "logistic_regression": 0.20,
    }

    all_raw_features = num_cols + cat_cols
    consensus: Dict[str, float] = {}
    for feat in all_raw_features:
        weighted_score = 0.0
        total_weight = 0.0
        for m_name, w in model_weights.items():
            if m_name in importance_summary:
                score = importance_summary[m_name].get("shap_importance", {}).get(feat, 0.0)
                weighted_score += w * score
                total_weight += w
        consensus[feat] = round(float(weighted_score / total_weight), 4) if total_weight > 0 else 0.0

    consensus_sorted = dict(sorted(consensus.items(), key=lambda item: item[1], reverse=True))
    importance_summary["consensus"] = consensus_sorted

    # Save to JSON
    output_path = root / "data" / "reference" / "feature_importance.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(importance_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("GLOBAL FEATURE IMPORTANCE CONSENSUS (Top 10):")
    print("=" * 60)
    for rank, (feat, score) in enumerate(list(consensus_sorted.items())[:10], 1):
        print(f"  #{rank:02d} {feat:<20} : {score:.4f} ({score*100:.1f}%)")
    print(f"\n[Importance] Saved rankings to: {output_path}")

    return importance_summary


if __name__ == "__main__":
    extract_feature_importance()
