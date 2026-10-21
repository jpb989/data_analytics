"""Data ingestion module for Telco Customer Churn dataset.

Handles dataset downloading, local caching, schema validation, type coercion,
and stratified train/val/test splitting.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import os
import urllib.request
import numpy as np
import pandas as pd
import yaml


def get_project_root() -> Path:
    """Return root path of the project."""
    return Path(__file__).resolve().parent.parent.parent


def load_dataset_config(config_path: Optional[str] = None) -> dict:
    """Load dataset configuration YAML."""
    if config_path is None:
        config_path = get_project_root() / "configs" / "dataset_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_raw_data(data_url: str, dest_path: Path) -> Path:
    """Download raw dataset from URL if not already present."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"[Ingestion] Cached dataset found at: {dest_path}")
        return dest_path

    print(f"[Ingestion] Downloading dataset from {data_url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(data_url, headers=headers)
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())
    print(f"[Ingestion] Downloaded successfully to: {dest_path}")
    return dest_path


def clean_telco_churn_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean and validate raw Telco Churn DataFrame.

    - Strips whitespace
    - Coerces TotalCharges to float (handles blank string entries)
    - Maps target column (e.g., 'Yes' -> 1, 'No' -> 0)
    - Validates expected columns
    """
    cleaned = df.copy()

    # Drop ID column if present
    id_col = config.get("id_column", "customerID")
    if id_col in cleaned.columns:
        cleaned = cleaned.drop(columns=[id_col])

    # Convert TotalCharges from string/object with spaces to float
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )
        # Impute missing TotalCharges with 0.0 (these correspond to tenure == 0)
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(0.0)

    # Standardize target column to binary int (1/0)
    target_col = config.get("target_column", "Churn")
    positive_val = config.get("target_positive_value", "Yes")
    if target_col in cleaned.columns:
        cleaned[target_col] = (cleaned[target_col].astype(str).str.strip() == positive_val).astype(int)

    # Convert SeniorCitizen to string category or int
    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(int)

    # Strip string whitespaces in categorical features
    for cat_col in config.get("categorical_features", []):
        if cat_col in cleaned.columns and cleaned[cat_col].dtype == "object":
            cleaned[cat_col] = cleaned[cat_col].astype(str).str.strip()

    return cleaned


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "Churn",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified split into Train (70%), Validation (15%), and Test (15%)."""
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # First split: train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=df[target_col] if target_col in df.columns else None,
    )

    # Second split: val vs test (equal shares of temp)
    val_share_of_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        random_state=random_state,
        stratify=temp_df[target_col] if target_col in temp_df.columns else None,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run_ingestion_pipeline(config_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Execute complete ingestion pipeline and save splits to disk."""
    config = load_dataset_config(config_path)
    root = get_project_root()

    raw_path = root / "data" / "raw" / "telco_churn.csv"
    data_url = config["data_url"]

    # 1. Download
    download_raw_data(data_url, raw_path)

    # 2. Load
    raw_df = pd.read_csv(raw_path)
    print(f"[Ingestion] Loaded raw shape: {raw_df.shape}")

    # 3. Clean
    cleaned_df = clean_telco_churn_data(raw_df, config)
    print(f"[Ingestion] Cleaned shape: {cleaned_df.shape}")

    # 4. Stratified Split
    splits_cfg = config.get("split_ratio", {})
    train_df, val_df, test_df = split_dataset(
        cleaned_df,
        target_col=config["target_column"],
        train_ratio=splits_cfg.get("train", 0.70),
        val_ratio=splits_cfg.get("val", 0.15),
        test_ratio=splits_cfg.get("test", 0.15),
        random_state=splits_cfg.get("random_state", 42),
    )

    # 5. Save processed splits
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    print(f"[Ingestion] Saved splits to {processed_dir}:")
    print(f"  - Train: {train_df.shape} (target rate: {train_df[config['target_column']].mean():.3f})")
    print(f"  - Val:   {val_df.shape} (target rate: {val_df[config['target_column']].mean():.3f})")
    print(f"  - Test:  {test_df.shape} (target rate: {test_df[config['target_column']].mean():.3f})")

    return {"train": train_df, "val": val_df, "test": test_df}


if __name__ == "__main__":
    run_ingestion_pipeline()
