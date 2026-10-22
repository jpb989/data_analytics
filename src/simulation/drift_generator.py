"""Production drift scenario generator.

Synthesizes multi-step temporal batches simulating four distinct operational scenarios:
1. Scenario A: Nominal Baseline (Clean, natural sampling variation)
2. Scenario B: Critical Drift (Severe drift on high-importance features: Contract, tenure)
3. Scenario C: Benign Drift (High statistical drift on negligible-importance features: gender, PhoneService)
4. Scenario D: Data Quality Failure (MCAR missingness surge and extreme outlier contamination)
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path when executed directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Tuple
import pandas as pd

from src.data.ingestion import get_project_root, load_dataset_config
from src.simulation.perturbation import DataPerturber


class ProductionDriftSimulator:
    """Generates parameterized multi-batch operational scenarios."""

    def __init__(self, base_df: pd.DataFrame, random_state: int = 42):
        self.base_df = base_df.copy()
        self.perturber = DataPerturber(random_state=random_state)
        self.random_state = random_state

    def generate_batches(self, num_batches: int = 5, batch_size: int = 500) -> List[pd.DataFrame]:
        """Bootstrap or sample temporal batches from the base test pool."""
        batches = []
        for b_idx in range(num_batches):
            sample = self.base_df.sample(
                n=min(batch_size, len(self.base_df)),
                replace=True,
                random_state=self.random_state + b_idx * 17,
            ).reset_index(drop=True)
            batches.append(sample)
        return batches

    def build_scenario_nominal(self, num_batches: int = 5) -> List[Tuple[str, pd.DataFrame]]:
        """Scenario A: Pristine operations without synthetic corruption."""
        batches = self.generate_batches(num_batches)
        return [(f"Batch_T{i+1}_Nominal", df) for i, df in enumerate(batches)]

    def build_scenario_critical_drift(self, num_batches: int = 5) -> List[Tuple[str, pd.DataFrame]]:
        """Scenario B: Progressive and sudden drift on top-ranked features (Contract, tenure, MonthlyCharges)."""
        batches = self.generate_batches(num_batches)
        scenario = []
        for i, df in enumerate(batches):
            b_df = df.copy()
            # Increasing drift intensity from Batch 2 onwards
            if i >= 1:
                intensity = i * 0.7  # 0.7, 1.4, 2.1, 2.8 sigmas
                b_df = self.perturber.inject_covariate_shift(b_df, ["tenure", "MonthlyCharges"], shift_sigmas=intensity)
            if i >= 2:
                # Corrupt Contract (top #1 feature)
                b_df = self.perturber.inject_unseen_categories(
                    b_df, ["Contract"], corruption_rate=0.10 * (i - 1), synthetic_label="NEW_FLEX_PLAN"
                )
            scenario.append((f"Batch_T{i+1}_CriticalDrift", b_df))
        return scenario

    def build_scenario_benign_drift(self, num_batches: int = 5) -> List[Tuple[str, pd.DataFrame]]:
        """Scenario C: High statistical drift on low-importance features (gender, PhoneService)."""
        batches = self.generate_batches(num_batches)
        scenario = []
        for i, df in enumerate(batches):
            b_df = df.copy()
            if i >= 1:
                # Inject novel categories and high shift into low-importance features
                b_df = self.perturber.inject_unseen_categories(
                    b_df, ["gender", "PhoneService"], corruption_rate=0.20 + 0.10 * i, synthetic_label="NON_BINARY_OTHER"
                )
            scenario.append((f"Batch_T{i+1}_BenignDrift", b_df))
        return scenario

    def build_scenario_quality_failure(self, num_batches: int = 5) -> List[Tuple[str, pd.DataFrame]]:
        """Scenario D: Pipeline data quality failure (MAR/MCAR missingness spike and outlier explosion)."""
        batches = self.generate_batches(num_batches)
        scenario = []
        for i, df in enumerate(batches):
            b_df = df.copy()
            if i >= 1:
                missing_rate = 0.10 * i  # 10%, 20%, 30%, 40%
                # Invalidate TotalCharges via MAR conditioned on MonthlyCharges >= median
                b_df = self.perturber.inject_mar(
                    b_df,
                    target_col="TotalCharges",
                    condition_col="MonthlyCharges",
                    condition_quantile=0.5,
                    missing_rate=min(0.85, missing_rate * 1.5),
                )
                # Invalidate tenure via MCAR
                b_df = self.perturber.inject_mcar(
                    b_df, ["tenure"], missing_rate=missing_rate
                )
            if i >= 2:
                # Extreme outlier contamination in TotalCharges
                b_df = self.perturber.inject_numerical_outliers(
                    b_df, ["TotalCharges"], mode="scale_multiplier", intensity=5.0, fraction=0.15
                )
            scenario.append((f"Batch_T{i+1}_QualityFailure", b_df))
        return scenario


def generate_and_save_all_scenarios():
    """Generate all operational scenarios from test.csv and persist them to disk."""
    root = get_project_root()
    test_path = root / "data" / "processed" / "test.csv"
    test_df = pd.read_csv(test_path)

    simulator = ProductionDriftSimulator(test_df)
    scenarios_dir = root / "data" / "simulations"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    all_scenarios = {
        "nominal": simulator.build_scenario_nominal(),
        "critical_drift": simulator.build_scenario_critical_drift(),
        "benign_drift": simulator.build_scenario_benign_drift(),
        "quality_failure": simulator.build_scenario_quality_failure(),
    }

    print("\n" + "=" * 60)
    print("GENERATING PRODUCTION SIMULATION SCENARIO BATCHES")
    print("=" * 60)

    for sc_name, batch_list in all_scenarios.items():
        sc_folder = scenarios_dir / sc_name
        sc_folder.mkdir(parents=True, exist_ok=True)
        print(f"\nScenario: {sc_name.upper()}")
        for batch_label, b_df in batch_list:
            file_path = sc_folder / f"{batch_label}.csv"
            b_df.to_csv(file_path, index=False)
            print(f"  -> Saved {batch_label} ({b_df.shape}) to {file_path.name}")

    print(f"\n[Simulator] All scenario batches generated in: {scenarios_dir}")


if __name__ == "__main__":
    generate_and_save_all_scenarios()
