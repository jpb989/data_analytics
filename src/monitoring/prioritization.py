"""Intelligent Alert Prioritization and Decision Engine.

Synthesizes data quality signals, statistical drift metrics, and SHAP feature importance
into a Composite Impact Score (CIS) and assigns actionable triage levels (P1 to P4),
filtering out benign false alarms and directing attention to mission-critical model degradation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class AlertPrioritizationEngine:
    """Triage engine mapping statistical and quality anomalies to actionable priority levels."""

    def __init__(
        self,
        feature_importance: Dict[str, float],
        w_drift: float = 0.40,
        w_importance: float = 0.45,
        w_quality: float = 0.15,
    ):
        self.feature_importance = feature_importance
        self.w_drift = w_drift
        self.w_importance = w_importance
        self.w_quality = w_quality

    def calculate_composite_risk(
        self,
        feature_name: str,
        psi: float,
        has_defect: bool,
    ) -> float:
        """Compute normalized composite risk score [0, 1] for an individual feature.
        
        Risk is active only when drift or quality defects occur, scaled by feature importance.
        """
        # 1. Normalized drift score [0, 1] (saturate at PSI = 0.5)
        norm_drift = min(1.0, max(0.0, psi) / 0.50)

        # 2. Normalized importance score [0, 1]
        importance = self.feature_importance.get(feature_name, 0.05)
        norm_importance = min(1.0, max(0.0, importance) / 0.20)

        # 3. Quality defect penalty [0, 1]
        defect_score = 1.0 if has_defect else 0.0

        # Drift risk is interaction of drift magnitude and feature importance
        drift_risk = norm_drift * (0.35 + 0.65 * norm_importance)
        quality_risk = defect_score * (0.40 + 0.60 * norm_importance)

        total_risk = min(1.0, 0.70 * drift_risk + 0.30 * quality_risk)
        return float(round(total_risk, 4))

    def triage_batch(
        self,
        feature_drift: Dict[str, Dict[str, Any]],
        quality_defects: List[Dict[str, Any]],
        delta_f1: float,
        delta_roc_auc: float,
    ) -> Dict[str, Any]:
        """Triage batch and classify into P1 Critical, P2 Quality, P3 Benign, or P4 Normal."""
        defective_features = {d["feature"] for d in quality_defects}

        feature_priorities = []
        max_risk = 0.0
        most_critical_feature = "None"

        for feat, d_meta in feature_drift.items():
            psi_val = d_meta.get("psi", 0.0)
            has_defect = feat in defective_features
            importance = self.feature_importance.get(feat, 0.0)

            risk_score = self.calculate_composite_risk(feat, psi_val, has_defect)
            if risk_score > max_risk:
                max_risk = risk_score
                most_critical_feature = feat

            feature_priorities.append({
                "feature": feat,
                "importance": round(importance, 4),
                "importance_pct": f"{importance * 100:.1f}%",
                "psi": psi_val,
                "has_defect": has_defect,
                "composite_risk": risk_score,
                "drift_severity": d_meta.get("severity", "STABLE"),
            })

        # Sort features by risk descending
        feature_priorities.sort(key=lambda x: x["composite_risk"], reverse=True)

        # Triage Decision Logic:
        # P1: High downstream impact (delta F1 >= 0.06) OR high risk on critical features with measurable drop
        # P2: Quality defect dominating with moderate impact
        # P3: High statistical drift (PSI >= 0.25) BUT negligible downstream impact (delta F1 < 0.02) -> SUPPRESSED
        # P4: Nominal state (PSI < 0.25 and delta F1 < 0.02)
        max_batch_psi = max([d.get("psi", 0.0) for d in feature_drift.values()]) if feature_drift else 0.0

        if delta_f1 >= 0.06 or (max_risk >= 0.50 and delta_f1 >= 0.025):
            system_priority = "P1_CRITICAL"
            action = "EMERGENCY_ALERT: Critical degradation detected. Fallback to heuristic or trigger retraining."
            suppressed = False
        elif len(quality_defects) > 0 and delta_f1 >= 0.02:
            system_priority = "P2_QUALITY_ALERT"
            action = "PIPELINE_INVESTIGATION: Data quality defect detected. Inspect upstream ETL or schema."
            suppressed = False
        elif max_batch_psi >= 0.25 and delta_f1 < 0.02:
            system_priority = "P3_BENIGN_DRIFT"
            action = "LOG_AND_SUPPRESS: High statistical drift detected without model impact. Suppressing false alarm."
            suppressed = True
        elif delta_f1 < 0.02:
            system_priority = "P4_NOMINAL"
            action = "NOMINAL: Operating within acceptable statistical tolerance. No action required."
            suppressed = False
        else:
            system_priority = "P2_QUALITY_ALERT"
            action = "MONITOR: Moderate variance detected. Schedule routine review."
            suppressed = False

        return {
            "system_priority": system_priority,
            "prescribed_action": action,
            "suppressed_false_alarm": suppressed,
            "delta_f1": round(delta_f1, 4),
            "delta_roc_auc": round(delta_roc_auc, 4),
            "max_composite_risk": max_risk,
            "critical_feature": most_critical_feature,
            "total_defects": len(quality_defects),
            "feature_triage_table": feature_priorities,
        }
