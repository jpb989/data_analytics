# Analysis Module: `src/analysis/`

This directory conducts empirical sensitivity studies and rank correlation benchmarks across simulated operational scenarios.

For the exhaustive mathematical specification, formulas, and empirical findings, see the complete guide:
👉 [**Comprehensive Analysis Documentation**](../../docs/folder_readmes/analysis.md)

---

### Key Files in this Folder

| File | Class / Functions | Description |
| :--- | :--- | :--- |
| [`correlation.py`](correlation.py) | `run_correlation_study` | Executes multi-model degradation benchmarking across all 20 scenario batches, computes Spearman's rho and Kendall's tau, and quantifies false alarm rates. |

---

### How to Run

```powershell
# Run the complete correlation and sensitivity study across all 20 batches
uv run .\src\analysis\correlation.py
```
