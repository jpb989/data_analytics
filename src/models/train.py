"""Model training and baseline benchmark establishment.

Trains and validates three diverse model architectures:
1. Logistic Regression (Linear baseline, vulnerable to scaling/outliers)
2. Random Forest (Non-linear bagging ensemble)
3. XGBoost (Gradient boosted decision trees)
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Tuple
import json
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data.ingestion import get_project_root, load_dataset_config
from src.data.preprocessor import DataPreprocessor
from src.models.evaluate import evaluate_binary_classifier


def load_model_config(config_path: Path = None) -> dict:
    """Load model training configuration YAML."""
    if config_path is None:
        config_path = get_project_root() / "configs" / "model_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_baseline_models() -> Dict[str, dict]:
    """Train all 3 baseline architectures, evaluate, and persist artifacts."""
    root = get_project_root()
    dataset_cfg = load_dataset_config()
    model_cfg = load_model_config()

    # 1. Load data
    train_df = pd.read_csv(root / "data" / "processed" / "train.csv")
    val_df = pd.read_csv(root / "data" / "processed" / "val.csv")
    test_df = pd.read_csv(root / "data" / "processed" / "test.csv")

    target_col = dataset_cfg["target_column"]
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    # 2. Load preprocessor and transform data
    prep_path = root / "data" / "reference" / "preprocessor.joblib"
    preprocessor = DataPreprocessor.load(prep_path)

    X_train_proc = preprocessor.transform(train_df.drop(columns=[target_col], errors="ignore"))
    X_val_proc = preprocessor.transform(val_df.drop(columns=[target_col], errors="ignore"))
    X_test_proc = preprocessor.transform(test_df.drop(columns=[target_col], errors="ignore"))

    # 3. Instantiate models with configured hyperparameters
    lr_cfg = model_cfg["models"]["logistic_regression"]
    rf_cfg = model_cfg["models"]["random_forest"]
    xgb_cfg = model_cfg["models"]["xgboost"]

    models = {
        "logistic_regression": LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            solver=lr_cfg.get("solver", "lbfgs"),
            random_state=lr_cfg.get("random_state", 42),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 150),
            max_depth=rf_cfg.get("max_depth", 8),
            min_samples_split=rf_cfg.get("min_samples_split", 5),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            random_state=rf_cfg.get("random_state", 42),
            n_jobs=rf_cfg.get("n_jobs", -1),
        ),
        "xgboost": XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 150),
            max_depth=xgb_cfg.get("max_depth", 5),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            random_state=xgb_cfg.get("random_state", 42),
            eval_metric=xgb_cfg.get("eval_metric", "logloss"),
        ),
    }

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    benchmark_results: Dict[str, dict] = {}

    print("\n" + "=" * 60)
    print("BASELINE MODEL TRAINING & EVALUATION BENCHMARKS")
    print("=" * 60)

    for model_name, model in models.items():
        print(f"\n[Training] {model_name.upper()}...")
        model.fit(X_train_proc, y_train)

        # Predict Val & Test
        val_proba = model.predict_proba(X_val_proc)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)
        val_metrics = evaluate_binary_classifier(y_val, val_pred, val_proba)

        test_proba = model.predict_proba(X_test_proc)[:, 1]
        test_pred = (test_proba >= 0.5).astype(int)
        test_metrics = evaluate_binary_classifier(y_test, test_pred, test_proba)

        benchmark_results[model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

        # Persist trained model
        save_path = models_dir / f"{model_name}.joblib"
        joblib.dump(model, save_path)
        print(f"  -> Saved model to: {save_path}")
        print(f"  -> Test ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f} | F1-Macro: {test_metrics['f1_macro']:.4f} | Accuracy: {test_metrics['accuracy']:.4f}")

    # Persist benchmark metrics
    metrics_path = root / "data" / "reference" / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[Benchmarking] Saved baseline reference metrics to: {metrics_path}")

    return benchmark_results


if __name__ == "__main__":
    train_baseline_models()
