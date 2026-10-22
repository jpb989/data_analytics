"""Preprocessing and feature transformation pipeline.

Constructs, fits, and serializes Scikit-Learn ColumnTransformer pipelines
with median imputation, scaling, and robust one-hot encoding for categorical variables.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.ingestion import get_project_root, load_dataset_config


class DataPreprocessor:
    """Preprocesses tabular data using Scikit-Learn ColumnTransformer."""

    def __init__(self, numerical_features: List[str], categorical_features: List[str]):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []

    def build_pipeline(self) -> ColumnTransformer:
        """Construct ColumnTransformer for numerical and categorical pipelines."""
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numerical_features),
                ("cat", cat_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )
        self.pipeline = transformer
        return transformer

    def fit(self, X: pd.DataFrame) -> "DataPreprocessor":
        """Fit preprocessing pipeline on reference training DataFrame."""
        if self.pipeline is None:
            self.build_pipeline()

        self.pipeline.fit(X)

        # Extract transformed feature names
        num_names = self.numerical_features
        cat_encoder = self.pipeline.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(cat_encoder.get_feature_names_out(self.categorical_features))
        self.feature_names_ = num_names + cat_names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame into preprocessed numerical feature matrix."""
        if self.pipeline is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit on X and return transformed matrix."""
        return self.fit(X).transform(X)

    def get_feature_names(self) -> List[str]:
        """Return names of output features after one-hot encoding."""
        return self.feature_names_

    def save(self, filepath: Path) -> None:
        """Serialize fitted preprocessor to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "feature_names_": self.feature_names_,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
        }
        joblib.dump(payload, filepath)
        print(f"[Preprocessor] Saved preprocessor pipeline to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "DataPreprocessor":
        """Load serialized preprocessor pipeline."""
        payload = joblib.load(filepath)
        if isinstance(payload, dict):
            obj = cls(
                numerical_features=payload["numerical_features"],
                categorical_features=payload["categorical_features"],
            )
            obj.pipeline = payload["pipeline"]
            obj.feature_names_ = payload["feature_names_"]
            return obj
        return payload


def run_preprocessor_pipeline() -> Tuple[DataPreprocessor, np.ndarray, np.ndarray, np.ndarray]:
    """Fit preprocessor on train.csv and save artifact."""
    root = get_project_root()
    cfg = load_dataset_config()

    train_df = pd.read_csv(root / "data" / "processed" / "train.csv")
    val_df = pd.read_csv(root / "data" / "processed" / "val.csv")
    test_df = pd.read_csv(root / "data" / "processed" / "test.csv")

    target_col = cfg["target_column"]
    X_train = train_df.drop(columns=[target_col], errors="ignore")
    X_val = val_df.drop(columns=[target_col], errors="ignore")
    X_test = test_df.drop(columns=[target_col], errors="ignore")

    preprocessor = DataPreprocessor(
        numerical_features=cfg["numerical_features"],
        categorical_features=cfg["categorical_features"],
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    prep_path = root / "data" / "reference" / "preprocessor.joblib"
    preprocessor.save(prep_path)

    print(f"[Preprocessor] Transformation complete:")
    print(f"  - Features count after one-hot encoding: {len(preprocessor.get_feature_names())}")
    print(f"  - X_train_proc shape: {X_train_proc.shape}")
    print(f"  - X_val_proc shape:   {X_val_proc.shape}")
    print(f"  - X_test_proc shape:  {X_test_proc.shape}")

    return preprocessor, X_train_proc, X_val_proc, X_test_proc


if __name__ == "__main__":
    run_preprocessor_pipeline()
