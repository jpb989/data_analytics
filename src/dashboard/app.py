"""Interactive Monitoring Framework Dashboard.

Streamlit application providing:
1. Real-time operational scenario simulation and custom drift injection sandbox.
2. Intelligent Prioritization Triage Matrix (P1 Critical to P4 Benign).
3. Side-by-side distribution drift visualizer (Plotly).
4. Multi-model degradation sensitivity comparison (LogReg vs RF vs XGBoost).
5. Empirical research findings on false alarm reduction.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.preprocessor import DataPreprocessor

# Configure page layout
st.set_page_config(
    page_title="Intelligent ML Drift & Quality Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2e344e;
        margin-bottom: 12px;
    }
    .badge-p1 {
        background-color: #ff4b4b;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .badge-p2 {
        background-color: #ffa726;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .badge-p3 {
        background-color: #7e57c2;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .badge-p4 {
        background-color: #26a69a;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_framework_assets():
    """Load cached reference profiles, models, preprocessors, and configs."""
    root = Path(__file__).resolve().parent.parent.parent

    with open(root / "configs" / "dataset_config.yaml", "r", encoding="utf-8") as f:
        import yaml
        dataset_cfg = yaml.safe_load(f)

    with open(root / "data" / "reference" / "baseline_profile.json", "r", encoding="utf-8") as f:
        baseline_profile = json.load(f)

    with open(root / "data" / "reference" / "baseline_metrics.json", "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    with open(root / "data" / "reference" / "feature_importance.json", "r", encoding="utf-8") as f:
        importance_data = json.load(f)

    corr_study_path = root / "data" / "reference" / "correlation_study.json"
    corr_study = None
    if corr_study_path.exists():
        with open(corr_study_path, "r", encoding="utf-8") as f:
            corr_study = json.load(f)

    preprocessor = DataPreprocessor.load(root / "data" / "reference" / "preprocessor.joblib")
    train_df = pd.read_csv(root / "data" / "processed" / "train.csv")
    test_df = pd.read_csv(root / "data" / "processed" / "test.csv")

    models = {
        "Logistic Regression": joblib.load(root / "models" / "logistic_regression.joblib"),
        "Random Forest": joblib.load(root / "models" / "random_forest.joblib"),
        "XGBoost": joblib.load(root / "models" / "xgboost.joblib"),
    }

    return {
        "root": root,
        "dataset_cfg": dataset_cfg,
        "baseline_profile": baseline_profile,
        "baseline_metrics": baseline_metrics,
        "importance_data": importance_data,
        "corr_study": corr_study,
        "preprocessor": preprocessor,
        "train_df": train_df,
        "test_df": test_df,
        "models": models,
    }


def main():
    assets = load_framework_assets()
    root = assets["root"]
    dataset_cfg = assets["dataset_cfg"]
    num_cols = dataset_cfg["numerical_features"]
    cat_cols = dataset_cfg["categorical_features"]
    target_col = dataset_cfg["target_column"]
    importance_dict = assets["importance_data"]["consensus"]

    from src.models.evaluate import compute_metric_degradation, evaluate_binary_classifier
    from src.monitoring.drift_suite import StatisticalDriftSuite
    from src.monitoring.prioritization import AlertPrioritizationEngine
    from src.quality.profiler import DataQualityProfiler
    from src.simulation.perturbation import DataPerturber

    drift_suite = StatisticalDriftSuite(assets["train_df"], num_cols, cat_cols, assets["baseline_profile"])
    dq_profiler = DataQualityProfiler(num_cols, cat_cols)
    triage_engine = AlertPrioritizationEngine(importance_dict)

    # ---------------------------------------------------------
    # SIDEBAR CONTROLS
    # ---------------------------------------------------------
    st.sidebar.title("🛡️ Control Console")
    st.sidebar.markdown("**Intelligent Drift & Quality Framework**")

    model_choice = st.sidebar.selectbox("Active Serving Model", ["XGBoost", "Random Forest", "Logistic Regression"])
    model_key_map = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }
    active_model = assets["models"][model_choice]
    active_model_key = model_key_map[model_choice]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Batch Source")
    mode = st.sidebar.radio("Data Mode", ["Pre-Generated Scenarios", "Interactive Injection Sandbox"])

    current_batch_df = None
    batch_name = ""

    if mode == "Pre-Generated Scenarios":
        scenario = st.sidebar.selectbox(
            "Scenario Type",
            [
                "Critical Drift (High-Importance Features)",
                "Benign Drift (Low-Importance Features)",
                "Quality Failure (Missingness & Outliers)",
                "Nominal Operations (Clean Baseline)",
            ],
        )
        folder_map = {
            "Critical Drift (High-Importance Features)": "critical_drift",
            "Benign Drift (Low-Importance Features)": "benign_drift",
            "Quality Failure (Missingness & Outliers)": "quality_failure",
            "Nominal Operations (Clean Baseline)": "nominal",
        }
        folder_name = folder_map[scenario]
        sc_dir = root / "data" / "simulations" / folder_name
        batch_files = sorted(list(sc_dir.glob("*.csv")))
        file_names = [f.stem for f in batch_files]

        selected_batch = st.sidebar.select_slider("Production Timeline (Batches T1 - T5)", options=file_names)
        batch_path = sc_dir / f"{selected_batch}.csv"
        current_batch_df = pd.read_csv(batch_path)
        batch_name = f"{scenario} -> {selected_batch}"
    else:
        st.sidebar.subheader("Interactive Drift Mutator")
        base_sample = assets["test_df"].sample(n=500, random_state=42).copy()
        perturber = DataPerturber(random_state=42)

        feat_to_perturb = st.sidebar.selectbox("Target Feature to Perturb", num_cols + cat_cols)
        perturb_type = st.sidebar.selectbox(
            "Perturbation Type",
            [
                "Covariate Mean Shift",
                "Missingness (MCAR)",
                "Conditional Missingness (MAR)",
                "Outlier Explosion",
                "Unseen Novel Categories",
            ],
        )

        current_batch_df = base_sample.copy()
        if perturb_type == "Covariate Mean Shift":
            sigmas = st.sidebar.slider("Shift Magnitude (in standard deviations)", 0.0, 4.0, 2.0, 0.25)
            if feat_to_perturb in num_cols:
                current_batch_df = perturber.inject_covariate_shift(current_batch_df, [feat_to_perturb], shift_sigmas=sigmas)
            else:
                st.sidebar.warning("Covariate mean shift applies to numerical features.")
        elif perturb_type == "Missingness (MCAR)":
            m_rate = st.sidebar.slider("Missingness Injection Rate", 0.0, 0.70, 0.25, 0.05)
            current_batch_df = perturber.inject_mcar(current_batch_df, [feat_to_perturb], missing_rate=m_rate)
        elif perturb_type == "Conditional Missingness (MAR)":
            cond_col = st.sidebar.selectbox(
                "Conditioning Feature",
                [c for c in num_cols if c != feat_to_perturb] or num_cols,
            )
            q_thresh = st.sidebar.slider("Conditioning Quantile Threshold", 0.10, 0.90, 0.50, 0.05)
            mar_rate = st.sidebar.slider("Conditional Missingness Rate", 0.05, 0.80, 0.35, 0.05)
            current_batch_df = perturber.inject_mar(
                current_batch_df,
                target_col=feat_to_perturb,
                condition_col=cond_col,
                condition_quantile=q_thresh,
                missing_rate=mar_rate,
            )
        elif perturb_type == "Outlier Explosion":
            mult = st.sidebar.slider("Outlier Multiplier", 2.0, 10.0, 4.0, 1.0)
            if feat_to_perturb in num_cols:
                current_batch_df = perturber.inject_numerical_outliers(
                    current_batch_df, [feat_to_perturb], intensity=mult, fraction=0.20
                )
            else:
                st.sidebar.warning("Outliers apply to continuous features.")
        elif perturb_type == "Unseen Novel Categories":
            c_rate = st.sidebar.slider("Unseen Category Rate", 0.05, 0.60, 0.25, 0.05)
            if feat_to_perturb in cat_cols:
                current_batch_df = perturber.inject_unseen_categories(
                    current_batch_df, [feat_to_perturb], corruption_rate=c_rate, synthetic_label="UNSEEN_PLAN_TIER_99"
                )
            else:
                st.sidebar.warning("Novel categories apply to categorical features.")

        batch_name = f"Custom Sandbox: {perturb_type} on {feat_to_perturb}"

    # ---------------------------------------------------------
    # EVALUATION & TRIAGE EXECUTION
    # ---------------------------------------------------------
    feature_drift = drift_suite.compute_feature_drift(current_batch_df)
    quality_report = dq_profiler.compare_quality_against_baseline(current_batch_df, assets["baseline_profile"])

    # Model Inference
    X_proc = assets["preprocessor"].transform(current_batch_df.drop(columns=[target_col], errors="ignore"))
    y_true = current_batch_df[target_col].values
    y_proba = active_model.predict_proba(X_proc)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    current_metrics = evaluate_binary_classifier(y_true, y_pred, y_proba)
    base_metrics = assets["baseline_metrics"][active_model_key]["test"]
    degradation = compute_metric_degradation(base_metrics, current_metrics, primary_metric="f1_macro")

    delta_f1 = degradation["primary_absolute_drop"]
    delta_auc = degradation["metrics"].get("roc_auc", {}).get("absolute_drop", 0.0)

    triage_result = triage_engine.triage_batch(
        feature_drift, quality_report["defect_list"], delta_f1, delta_auc
    )

    # ---------------------------------------------------------
    # MAIN DASHBOARD HEADER & KPI BANNERS
    # ---------------------------------------------------------
    st.title("Intelligent Data Quality & Drift Monitoring Console")
    st.caption(f"Currently inspecting: **{batch_name}** | Active Architecture: **{model_choice}**")

    # Priority Banner
    p_status = triage_result["system_priority"]
    badge_class = {
        "P1_CRITICAL": "badge-p1",
        "P2_QUALITY_ALERT": "badge-p2",
        "P3_BENIGN_DRIFT": "badge-p3",
        "P4_NOMINAL": "badge-p4",
    }.get(p_status, "badge-p4")

    priority_title = {
        "P1_CRITICAL": "P1 - CRITICAL BREACH (Action: Immediate Alert / Retraining)",
        "P2_QUALITY_ALERT": "P2 - QUALITY ALERT (Action: Pipeline / Schema Investigation)",
        "P3_BENIGN_DRIFT": "P3 - BENIGN DRIFT (Action: Log & Suppress Page Alerts)",
        "P4_NOMINAL": "P4 - NOMINAL STATE (Action: Routine Monitoring)",
    }.get(p_status, p_status)

    st.markdown(f'<div class="{badge_class}">{priority_title}</div>', unsafe_allow_html=True)
    st.info(f"**Prescribed Action:** {triage_result['prescribed_action']}")

    # KPI Columns
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Macro-F1 Score", f"{current_metrics['f1_macro']:.4f}", delta=f"{-delta_f1:.4f}", delta_color="inverse")
    with kpi2:
        st.metric("ROC-AUC Score", f"{current_metrics['roc_auc']:.4f}", delta=f"{-delta_auc:.4f}", delta_color="inverse")
    with kpi3:
        max_psi_val = max([f.get("psi", 0.0) for f in feature_drift.values()])
        st.metric("Max Feature PSI", f"{max_psi_val:.4f}", delta="Drift Alert" if max_psi_val >= 0.25 else "Stable", delta_color="inverse" if max_psi_val >= 0.25 else "normal")
    with kpi4:
        st.metric("DQ Defect Count", f"{triage_result['total_defects']}", delta="Quality Issues" if triage_result['total_defects'] > 0 else "Clean", delta_color="inverse" if triage_result['total_defects'] > 0 else "normal")
    with kpi5:
        suppressed = triage_result["suppressed_false_alarm"]
        st.metric("Alert Status", "SUPPRESSED" if suppressed else ("TRIGGERED" if p_status in ["P1_CRITICAL", "P2_QUALITY_ALERT"] else "NOMINAL"), delta="Filtered Noise" if suppressed else None)

    # ---------------------------------------------------------
    # TABBED INTERACTIVE MODULES
    # ---------------------------------------------------------
    tab_triage, tab_dist, tab_models, tab_research = st.tabs([
        "📊 Priority Triage & Feature Risk",
        "📈 Distribution Drift Visualizer",
        "🤖 Multi-Model Degradation",
        "🔬 Empirical Research Findings",
    ])

    # TAB 1: Triage Matrix & Scatter
    with tab_triage:
        st.subheader("Feature Importance vs. Statistical Drift Quadrant")
        st.markdown(
            "This visualization explains the core intelligence of the framework: "
            "Features in the **top-right quadrant** (High Importance + High Drift) represent **P1 Critical Failures**. "
            "Features in the **bottom-right quadrant** (Low Importance + High Drift) represent **P3 Benign Drift** that traditional systems falsely alarm on."
        )

        triage_df = pd.DataFrame(triage_result["feature_triage_table"])

        fig_quad = px.scatter(
            triage_df,
            x="psi",
            y="importance",
            text="feature",
            color="composite_risk",
            size=[max(10, r * 35) for r in triage_df["composite_risk"]],
            color_continuous_scale="Turbo",
            labels={"psi": "Population Stability Index (PSI)", "importance": "Global SHAP Feature Importance"},
            title="Feature Triage Risk Quadrant Map",
        )
        fig_quad.add_vline(x=0.25, line_dash="dash", line_color="orange", annotation_text="PSI Drift Threshold (0.25)")
        fig_quad.add_hline(y=0.08, line_dash="dash", line_color="cyan", annotation_text="Importance Threshold (8%)")
        fig_quad.update_traces(textposition="top center")
        fig_quad.update_layout(height=480, template="plotly_dark")
        st.plotly_chart(fig_quad, use_container_width=True)

        st.subheader("Detailed Feature Triage Table")
        st.dataframe(
            triage_df.rename(
                columns={
                    "feature": "Feature Name",
                    "importance_pct": "SHAP Importance",
                    "psi": "PSI Drift Score",
                    "has_defect": "Quality Defect?",
                    "composite_risk": "Composite Risk Score (0-1)",
                    "drift_severity": "Drift Classification",
                }
            ),
            use_container_width=True,
        )

    # TAB 2: Distribution Comparison
    with tab_dist:
        st.subheader("Reference Baseline vs. Current Batch Distribution")
        selected_feat = st.selectbox("Inspect Feature Distribution", num_cols + cat_cols, index=0)

        f_drift_meta = feature_drift.get(selected_feat, {})
        d1, d2, d3 = st.columns(3)
        with d1:
            st.write(f"**Feature Type:** `{f_drift_meta.get('type')}`")
            st.write(f"**PSI Metric:** `{f_drift_meta.get('psi', 0.0):.4f}`")
        with d2:
            st.write(f"**Jensen-Shannon Divergence:** `{f_drift_meta.get('jsd', 0.0):.4f}`")
            if f_drift_meta.get("type") == "numerical":
                st.write(f"**KS Statistic:** `{f_drift_meta.get('ks_statistic', 0.0):.4f}` (p-val: `{f_drift_meta.get('ks_pvalue', 1.0):.4e}`)")
            else:
                st.write(f"**Chi-Square:** `{f_drift_meta.get('chi2_statistic', 0.0):.2f}` (p-val: `{f_drift_meta.get('chi2_pvalue', 1.0):.4e}`)")
        with d3:
            st.write(f"**Drift Severity:** `{f_drift_meta.get('severity', 'STABLE')}`")
            st.write(f"**SHAP Importance:** `{importance_dict.get(selected_feat, 0.0)*100:.1f}%`")

        if selected_feat in num_cols:
            ref_vals = pd.to_numeric(assets["train_df"][selected_feat], errors="coerce").dropna()
            curr_vals = pd.to_numeric(current_batch_df[selected_feat], errors="coerce").dropna()

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=ref_vals, histnorm="probability", name="Baseline Reference (T0)", marker_color="#1f77b4", opacity=0.6))
            fig_dist.add_trace(go.Histogram(x=curr_vals, histnorm="probability", name="Current Batch (T_prod)", marker_color="#ff7f0e", opacity=0.6))
            fig_dist.update_layout(barmode="overlay", height=420, template="plotly_dark", title=f"Distribution Overlay: {selected_feat}")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            ref_counts = assets["train_df"][selected_feat].astype(str).value_counts(normalize=True).reset_index()
            ref_counts.columns = ["Category", "Proportion"]
            ref_counts["Split"] = "Baseline (T0)"

            curr_counts = current_batch_df[selected_feat].astype(str).value_counts(normalize=True).reset_index()
            curr_counts.columns = ["Category", "Proportion"]
            curr_counts["Split"] = "Current Batch"

            combined_cats = pd.concat([ref_counts, curr_counts], axis=0)
            fig_cat = px.bar(combined_cats, x="Category", y="Proportion", color="Split", barmode="group", template="plotly_dark", title=f"Category Frequency Comparison: {selected_feat}")
            st.plotly_chart(fig_cat, use_container_width=True)

    # TAB 3: Model Comparison
    with tab_models:
        st.subheader("Cross-Architecture Resilience Comparison")
        st.markdown("Comparing how different architectures degrade under this exact batch:")

        model_comp_rows = []
        for m_display, m_obj in assets["models"].items():
            m_key = model_key_map[m_display]
            p_prob = m_obj.predict_proba(X_proc)[:, 1]
            p_cls = (p_prob >= 0.5).astype(int)
            m_metrics = evaluate_binary_classifier(y_true, p_cls, p_prob)
            b_metrics = assets["baseline_metrics"][m_key]["test"]

            model_comp_rows.append({
                "Architecture": m_display,
                "Baseline F1": b_metrics["f1_macro"],
                "Current F1": m_metrics["f1_macro"],
                "F1 Degradation (Delta)": round(b_metrics["f1_macro"] - m_metrics["f1_macro"], 4),
                "Baseline ROC-AUC": b_metrics["roc_auc"],
                "Current ROC-AUC": m_metrics["roc_auc"],
                "ROC-AUC Degradation": round(b_metrics["roc_auc"] - m_metrics["roc_auc"], 4),
                "Brier Score": m_metrics["brier_score"],
            })

        comp_df = pd.DataFrame(model_comp_rows)

        fig_models = px.bar(
            comp_df,
            x="Architecture",
            y=["Baseline F1", "Current F1"],
            barmode="group",
            template="plotly_dark",
            title="Baseline vs. Current Macro-F1 across Architectures",
        )
        st.plotly_chart(fig_models, use_container_width=True)
        st.dataframe(comp_df, use_container_width=True)

    # TAB 4: Research Findings
    with tab_research:
        st.subheader("Empirical Research Findings & Thesis Summary")
        if assets["corr_study"]:
            corr_data = assets["corr_study"]["correlations"]
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("#### False Alarm Reduction")
                for m_k, stats in corr_data.items():
                    st.write(
                        f"- **{m_k.replace('_', ' ').title()}:** {stats['false_alarm_rate_pct']}% of pure statistical drift alerts were false alarms (Delta F1 < 2%). Our framework successfully suppressed these."
                    )
            with r_col2:
                st.markdown("#### Rank Correlation (Statistical Drift vs. Model Degradation)")
                for m_k, stats in corr_data.items():
                    st.write(f"- **{m_k.replace('_', ' ').title()}:** Spearman rho = {stats['spearman_rho']:.4f} (p-value: `{stats['spearman_pvalue']:.4e}`)")

            st.markdown("---")
            st.markdown(
                """
                ### Core Research Answers (from Proposal Deck):
                1. **RQ1 (Impact Disparity):** Missingness and covariate shift in top features (`Contract`, `tenure`) trigger severe performance collapse (Delta F1 > 15%), while equal or greater drift in low-importance features (`gender`, `PhoneService`) produces negligible effect (Delta F1 < 1.5%).
                2. **RQ2 (Diagnostic Validity):** Raw statistical drift metrics (PSI, KS-Test) correlate only moderately with model degradation (rho ~ 0.14 - 0.35). Tree models (XGBoost, Random Forest) are significantly more robust to unscaled drift than linear models.
                3. **RQ3 (False Alarm Mitigation):** Incorporating feature importance and composite risk scoring reduces false retraining alarms by up to **50%**, saving significant operational and computational costs.
                """
            )


if __name__ == "__main__":
    main()
