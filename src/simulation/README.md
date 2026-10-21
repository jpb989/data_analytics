# Simulation Module: `src/simulation/`

This directory implements the synthetic data quality perturbation and multi-batch operational scenario simulator.

For the exhaustive mathematical specification, formulas, and scenario mechanics, see the complete guide:
👉 [**Comprehensive Simulation Documentation**](../../docs/folder_readmes/simulation.md)

---

### Key Files in this Folder

| File | Class / Functions | Description |
| :--- | :--- | :--- |
| [`perturbation.py`](perturbation.py) | `DataPerturber` | Core stochastic mutator implementing MCAR, MAR, Cauchy/Gaussian outliers, scale multipliers, novel unseen category injections, covariate mean shifts ($k\sigma$), and variance inflation. |
| [`drift_generator.py`](drift_generator.py) | `ProductionDriftSimulator`, `generate_and_save_all_scenarios()` | Temporal scenario generator synthesizing 20 production batches across four operational scenarios (`nominal`, `critical_drift`, `benign_drift`, `quality_failure`). |

---

### How to Run

```bash
# Regenerate all 20 scenario batch CSVs to data/simulations/
uv run .\src\simulation\drift_generator.py

# Run unit tests
uv run pytest -v tests/test_perturbation.py
```
