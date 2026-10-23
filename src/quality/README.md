# Quality Module: `src/quality/`

This directory implements baseline dataset profiling, locked reference quantile deciles, and data quality defect detection.

For the exhaustive mathematical specification, formulas, and profiling mechanics, see the complete guide:
👉 [**Comprehensive Data & Quality Documentation**](../../docs/folder_readmes/data.md)

---

### Key Files in this Folder

| File | Class / Functions | Description |
| :--- | :--- | :--- |
| [`profiler.py`](profiler.py) | `DataQualityProfiler` | Computes reference profiles, locks 10 quantile decile bin boundaries for continuous PSI, and tracks missingness & IQR outliers. |

---

### How to Run

```powershell
# Generate and serialize baseline reference profile
uv run .\src\quality\profiler.py
```
