# Data Module: `src/data/`

This directory implements raw data ingestion, schema cleaning, stratified splitting, and the feature preprocessing pipeline.

For the exhaustive mathematical specification, formulas, and preprocessor mechanics, see the complete guide:
👉 [**Comprehensive Data & Quality Documentation**](../../docs/folder_readmes/data.md)

---

### Key Files in this Folder

| File | Class / Functions | Description |
| :--- | :--- | :--- |
| [`ingestion.py`](ingestion.py) | `run_ingestion_pipeline`, `split_dataset`, `clean_telco_churn_data` | Downloads raw Telco dataset, cleans whitespace strings, coerces numeric types, and performs two-stage stratified 70/15/15 split. |
| [`preprocessor.py`](preprocessor.py) | `DataPreprocessor`, `run_preprocessor_pipeline` | Scikit-Learn `ColumnTransformer` (Median Imputation + Scaling + OneHotEncoding) transforming 19 raw features into 46 encoded dimensions. |

---

### How to Run

```powershell
# Run data ingestion and generate processed splits
uv run .\src\data\ingestion.py

# Fit and serialize preprocessor pipeline
uv run .\src\data\preprocessor.py
```
