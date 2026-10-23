# Models Module: `src/models/`

This directory implements baseline model training, performance benchmarking, and consensus feature attribution.

For the exhaustive mathematical specification, formulas, and SHAP attribution mechanics, see the complete guide:
👉 [**Comprehensive Models & Importance Documentation**](../../docs/folder_readmes/models.md)

---

### Key Files in this Folder

| File | Class / Functions | Description |
| :--- | :--- | :--- |
| [`train.py`](train.py) | `train_baseline_models` | Configures and trains Logistic Regression, Random Forest, and XGBoost; evaluates benchmark metrics and serializes binaries to `models/`. |
| [`evaluate.py`](evaluate.py) | `evaluate_binary_classifier`, `compute_metric_degradation` | Classification metrics suite (ROC-AUC, PR-AUC, Macro-F1, Brier Score, etc.) and longitudinal performance degradation tracking. |
| [`importance.py`](importance.py) | `extract_feature_importance`, `aggregate_encoded_importance` | Aggregates one-hot dummy columns back to raw features, extracts TreeSHAP and native attributions, and synthesizes multi-model consensus weights. |

---

### How to Run

```powershell
# Train all three baseline models and establish benchmark metrics
uv run .\src\models\train.py

# Extract and serialize consensus feature importance rankings
uv run .\src\models\importance.py
```
