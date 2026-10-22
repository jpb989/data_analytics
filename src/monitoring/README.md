# Monitoring Module: `src/monitoring/`

This directory implements univariate statistical drift detection and the intelligent alert prioritization triage engine.

For the exhaustive mathematical specification, formulas, and triage mechanics, see the complete guide:
👉 [**Comprehensive Monitoring Documentation**](../../docs/folder_readmes/monitoring.md)

---

### Key Files in this Folder

| File | Class / Functions | Description |
| :--- | :--- | :--- |
| [`drift_suite.py`](drift_suite.py) | `StatisticalDriftSuite`, `calculate_psi`, `calculate_continuous_jsd`, `calculate_categorical_psi`, `calculate_categorical_jsd` | Univariate drift tests across continuous (KS-test, PSI, JSD, Wasserstein) and categorical (Chi-Square, Categorical PSI, Categorical JSD) features. |
| [`prioritization.py`](prioritization.py) | `AlertPrioritizationEngine` | Intelligent triage engine calculating the Composite Impact Score (CIS) and mapping alerts into P1 Critical, P2 Quality Alert, P3 Benign Drift (suppressed), and P4 Nominal. |

---

### How to Run

```powershell
# Run monitoring unit tests
uv run pytest -v tests/test_drift_suite.py tests/test_prioritization.py
```
