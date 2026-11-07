"""Reusable Plotly charts styled for dark navy dashboard aesthetic."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from dashboard.config import THEME_COLORS
from dashboard.utils.visualization import apply_custom_theme


def render_yearly_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Renders annual event count bar chart with gradient fill."""
    if df.empty or "year" not in df.columns:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Event Count by Year (No Data)")

    year_counts = df["year"].value_counts().sort_index().reset_index()
    year_counts.columns = ["Year", "Events"]
    year_counts["Year"] = year_counts["Year"].astype(str)

    fig = go.Figure(
        data=[
            go.Bar(
                x=year_counts["Year"],
                y=year_counts["Events"],
                marker=dict(
                    color=THEME_COLORS["primary"],
                    line=dict(color="#2EC4B6", width=1.5),
                ),
                text=year_counts["Events"],
                textposition="auto",
                hoverinfo="x+y",
            )
        ]
    )
    return apply_custom_theme(
        fig,
        title="Landslide Events by Year",
        x_title="Calendar Year",
        y_title="Event Count",
        height=350,
        show_legend=False,
    )


def render_monthly_seasonality_chart(df: pd.DataFrame) -> go.Figure:
    """Renders monthly seasonality chart highlighting the Himalayan monsoon window."""
    if df.empty:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Monthly Seasonality (No Data)")

    months_order = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    # Calculate counts per month
    if "month_name" in df.columns:
        counts = df["month_name"].value_counts().reindex(months_order, fill_value=0)
    else:
        counts = pd.Series(0, index=months_order)

    # Monsoon highlight colors (Jul, Aug, Sep get bright cyan/purple; others get muted blue)
    colors = [
        THEME_COLORS["danger"] if m in ["Jul", "Aug", "Sep"]
        else THEME_COLORS["accent_blue"]
        for m in months_order
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=months_order,
                y=counts.values,
                marker=dict(color=colors),
                text=counts.values,
                textposition="auto",
            )
        ]
    )

    fig.add_annotation(
        x="Aug",
        y=max(counts.values) * 0.95 if max(counts.values) > 0 else 10,
        text="Monsoon Peak (Jul–Sep)",
        showarrow=True,
        arrowhead=2,
        arrowcolor=THEME_COLORS["warning"],
        font=dict(color=THEME_COLORS["warning"], size=11),
        bgcolor="rgba(28, 37, 65, 0.9)",
        bordercolor=THEME_COLORS["warning"],
    )

    return apply_custom_theme(
        fig,
        title="Monthly Landslide Distribution",
        x_title="Month",
        y_title="Observation Count",
        height=350,
        show_legend=False,
    )


def render_trigger_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Renders trigger category breakdown donut chart."""
    if df.empty:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Trigger Categories (No Data)")

    col = "Landslide Trigger" if "Landslide Trigger" in df.columns else "trigger"
    if col not in df.columns:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Trigger Information Unavailable")

    trigger_counts = df[col].value_counts().head(8).reset_index()
    trigger_counts.columns = ["Trigger", "Count"]

    colors = [
        THEME_COLORS["primary"],
        THEME_COLORS["secondary"],
        THEME_COLORS["accent_blue"],
        THEME_COLORS["warning"],
        THEME_COLORS["danger"],
        "#70D6FF",
        "#B5179E",
        "#FF9F1C",
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=trigger_counts["Trigger"],
                values=trigger_counts["Count"],
                hole=0.45,
                marker=dict(colors=colors, line=dict(color=THEME_COLORS["card_bg"], width=2)),
                textinfo="percent+label",
                hoverinfo="label+value+percent",
            )
        ]
    )
    return apply_custom_theme(
        fig,
        title="Landslide Triggers",
        height=350,
        show_legend=False,
    )


def render_model_comparison_bar_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """Renders grouped bar chart comparing RF vs XGBoost across AUROC, PR-AUC, and F1.
    
    Robustly handles:
    1. Standard tidy metrics schema (with 'model', 'auroc', 'prauc', 'f1_score_at_0.5' / 'f1')
       from detection_metrics.csv or susceptibility_metrics.csv.
    2. Legacy wide metrics schema (with 'rf_auroc', 'xgb_auroc', etc.).
    """
    if metrics_df is None or metrics_df.empty:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Model Metrics Unavailable")

    metrics_names = ["AUROC", "PR-AUC", "F1 Score"]
    rf_scores = [0.0, 0.0, 0.0]
    xgb_scores = [0.0, 0.0, 0.0]
    chart_title = "Model Performance (100km Purged)"
    rf_label = "Random Forest"
    xgb_label = "XGBoost"

    cols = set(metrics_df.columns)

    # 1. Legacy wide schema (rf_auroc, xgb_auroc)
    if "rf_auroc" in cols and "xgb_auroc" in cols:
        target_df = metrics_df
        if "feature_set" in cols:
            red = metrics_df[metrics_df["feature_set"].str.contains("REDUCED", case=False, na=False)]
            if not red.empty:
                target_df = red
        target_row = target_df.iloc[-1]

        rf_auc = target_row.get("rf_auroc", 0.0)
        rf_pr = target_row.get("rf_prauc", target_row.get("rf_pr_auc", 0.0))
        rf_f1 = target_row.get("rf_f1", target_row.get("rf_f1_score_at_0.5", target_row.get("rf_f1_score", 0.0)))

        xgb_auc = target_row.get("xgb_auroc", 0.0)
        xgb_pr = target_row.get("xgb_prauc", target_row.get("xgb_pr_auc", 0.0))
        xgb_f1 = target_row.get("xgb_f1", target_row.get("xgb_f1_score_at_0.5", target_row.get("xgb_f1_score", 0.0)))

        rf_scores = [float(rf_auc or 0.0), float(rf_pr or 0.0), float(rf_f1 or 0.0)]
        xgb_scores = [float(xgb_auc or 0.0), float(xgb_pr or 0.0), float(xgb_f1 or 0.0)]
        rf_label = "Random Forest"
        xgb_label = "XGBoost"

    # 2. Standard tidy schema (detection_metrics / susceptibility_metrics)
    elif "model" in cols:
        df_work = metrics_df.copy()

        # Prioritize 100km spatial purge standard if present
        if "purge_distance_km" in df_work.columns:
            purged = df_work[df_work["purge_distance_km"] == 100.0]
            if not purged.empty:
                df_work = purged

        # If configuration exists (susceptibility), default to Core model
        if "configuration" in df_work.columns:
            core = df_work[df_work["configuration"].str.contains("Core", case=False, na=False)]
            if not core.empty:
                df_work = core
                chart_title = "Susceptibility Core Model Benchmark (100km Purged)"

        rf_mask = df_work["model"].str.contains(r"Random Forest|RF", case=False, na=False)
        xgb_mask = df_work["model"].str.contains(r"XGBoost|XGB", case=False, na=False)

        rf_row = df_work[rf_mask]
        xgb_row = df_work[xgb_mask]

        auc_col = next((c for c in ["auroc", "auc", "roc_auc"] if c in df_work.columns), "auroc")
        pr_col = next((c for c in ["prauc", "pr_auc", "average_precision"] if c in df_work.columns), auc_col)
        f1_col = next((c for c in ["f1_score_at_0.5", "f1", "f1_score"] if c in df_work.columns), auc_col)

        if not rf_row.empty:
            r = rf_row.iloc[0]
            rf_scores = [
                float(r.get(auc_col, 0.0) or 0.0),
                float(r.get(pr_col, 0.0) or 0.0),
                float(r.get(f1_col, 0.0) or 0.0),
            ]
            rf_label = "Random Forest (100km Purged)" if "purge_distance_km" in df_work.columns else "Random Forest"

        if not xgb_row.empty:
            x = xgb_row.iloc[0]
            xgb_scores = [
                float(x.get(auc_col, 0.0) or 0.0),
                float(x.get(pr_col, 0.0) or 0.0),
                float(x.get(f1_col, 0.0) or 0.0),
            ]
            xgb_label = "XGBoost (100km Purged)" if "purge_distance_km" in df_work.columns else "XGBoost"

    else:
        # Generic fallback for 2-row comparison
        auc_col = next((c for c in ["auroc", "auc", "roc_auc"] if c in cols), None)
        pr_col = next((c for c in ["prauc", "pr_auc", "average_precision"] if c in cols), auc_col)
        f1_col = next((c for c in ["f1_score_at_0.5", "f1", "f1_score"] if c in cols), auc_col)

        if auc_col and len(metrics_df) >= 2:
            r0 = metrics_df.iloc[0]
            r1 = metrics_df.iloc[1]
            rf_scores = [float(r0.get(auc_col, 0.0) or 0.0), float(r0.get(pr_col, 0.0) or 0.0), float(r0.get(f1_col, 0.0) or 0.0)]
            xgb_scores = [float(r1.get(auc_col, 0.0) or 0.0), float(r1.get(pr_col, 0.0) or 0.0), float(r1.get(f1_col, 0.0) or 0.0)]
            rf_label = str(r0.get("model", "Model 1"))
            xgb_label = str(r1.get("model", "Model 2"))

    fig = go.Figure(
        data=[
            go.Bar(
                name=rf_label,
                x=metrics_names,
                y=rf_scores,
                marker_color=THEME_COLORS["secondary"],
                text=[f"{v:.3f}" for v in rf_scores],
                textposition="auto",
            ),
            go.Bar(
                name=xgb_label,
                x=metrics_names,
                y=xgb_scores,
                marker_color=THEME_COLORS["primary"],
                text=[f"{v:.3f}" for v in xgb_scores],
                textposition="auto",
            ),
        ]
    )
    fig.update_layout(barmode="group", yaxis_range=[0.75, 1.02])
    return apply_custom_theme(
        fig,
        title=chart_title,
        x_title="Evaluation Metric",
        y_title="Score",
        height=350,
    )


def render_roc_curves(oof_df: pd.DataFrame, feature_set: str = "reduced") -> go.Figure:
    """Plots spatial cross-validation ROC curves for RF and XGBoost."""
    if oof_df.empty:
        fig = go.Figure()
        return apply_custom_theme(fig, title="ROC Curves (No Data)")

    y_true = oof_df["label"].values
    p_rf = oof_df[f"oof_rf_{feature_set}"].values
    p_xgb = oof_df[f"oof_xgb_{feature_set}"].values

    fpr_rf, tpr_rf, _ = roc_curve(y_true, p_rf)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, p_xgb)

    auc_rf = roc_auc_score(y_true, p_rf)
    auc_xgb = roc_auc_score(y_true, p_xgb)

    fig = go.Figure()

    # Diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.3)", dash="dash"),
            name="Random Guess (AUC=0.50)",
        )
    )

    # Random Forest ROC
    fig.add_trace(
        go.Scatter(
            x=fpr_rf,
            y=tpr_rf,
            mode="lines",
            line=dict(color=THEME_COLORS["secondary"], width=3),
            name=f"Random Forest (AUROC = {auc_rf:.4f})",
        )
    )

    # XGBoost ROC
    fig.add_trace(
        go.Scatter(
            x=fpr_xgb,
            y=tpr_xgb,
            mode="lines",
            line=dict(color=THEME_COLORS["primary"], width=3),
            name=f"XGBoost (AUROC = {auc_xgb:.4f})",
        )
    )

    return apply_custom_theme(
        fig,
        title=f"Receiver Operating Characteristic (ROC) — {feature_set.upper()} Features (OOF)",
        x_title="False Positive Rate (1 - Specificity)",
        y_title="True Positive Rate (Recall / Sensitivity)",
        height=400,
    )


def render_pr_curves(oof_df: pd.DataFrame, feature_set: str = "reduced") -> go.Figure:
    """Plots Precision-Recall curves for spatial out-of-fold predictions."""
    if oof_df.empty:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Precision-Recall Curves (No Data)")

    y_true = oof_df["label"].values
    p_rf = oof_df[f"oof_rf_{feature_set}"].values
    p_xgb = oof_df[f"oof_xgb_{feature_set}"].values

    prec_rf, rec_rf, _ = precision_recall_curve(y_true, p_rf)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, p_xgb)

    prauc_rf = average_precision_score(y_true, p_rf)
    prauc_xgb = average_precision_score(y_true, p_xgb)

    baseline = float(np.mean(y_true))

    fig = go.Figure()

    # Baseline positive prevalence
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.3)", dash="dash"),
            name=f"Baseline Prevalence ({baseline:.2f})",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rec_rf,
            y=prec_rf,
            mode="lines",
            line=dict(color=THEME_COLORS["secondary"], width=3),
            name=f"Random Forest (PR-AUC = {prauc_rf:.4f})",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rec_xgb,
            y=prec_xgb,
            mode="lines",
            line=dict(color=THEME_COLORS["primary"], width=3),
            name=f"XGBoost (PR-AUC = {prauc_xgb:.4f})",
        )
    )

    return apply_custom_theme(
        fig,
        title=f"Precision-Recall (PR) Curve — {feature_set.upper()} Features (OOF)",
        x_title="Recall (Sensitivity)",
        y_title="Precision (Positive Predictive Value)",
        height=400,
    )


def render_confusion_matrix_chart(
    cm_dict: Dict[str, Any], model_name: str = "XGBoost"
) -> go.Figure:
    """Renders confusion matrix heatmap with counts and percentage labels."""
    cm = cm_dict["confusion_matrix"]
    tn, fp, fn, tp = cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"]
    thresh = cm_dict["threshold"]

    z = [[tn, fp], [fn, tp]]
    text_labels = [
        [f"TN: {tn}<br>({tn/(tn+fp):.1%})", f"FP: {fp}<br>({fp/(tn+fp):.1%})"],
        [f"FN: {fn}<br>({fn/(fn+tp):.1%})", f"TP: {tp}<br>({tp/(fn+tp):.1%})"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Stable (0)", "Predicted Landslide (1)"],
            y=["Actual Stable (0)", "Actual Landslide (1)"],
            text=text_labels,
            texttemplate="%{text}",
            colorscale=[[0, "#1C2541"], [0.5, "#3A86FF"], [1.0, "#48E5C2"]],
            showscale=False,
        )
    )

    fig.update_layout(yaxis=dict(autorange="reversed"))

    return apply_custom_theme(
        fig,
        title=f"Confusion Matrix: {model_name} (Threshold = {thresh:.2f})",
        height=350,
        show_legend=False,
    )


def render_feature_importance_chart(
    rf_model: Any, xgb_model: Any, feature_cols: List[str]
) -> go.Figure:
    """Renders dual-bar feature importance comparison."""
    # RF feature importance
    if hasattr(rf_model, "named_steps") and hasattr(rf_model.named_steps["clf"], "feature_importances_"):
        rf_imp = rf_model.named_steps["clf"].feature_importances_
    elif hasattr(rf_model, "feature_importances_"):
        rf_imp = rf_model.feature_importances_
    else:
        rf_imp = np.zeros(len(feature_cols))

    # XGB feature importance
    if hasattr(xgb_model, "feature_importances_"):
        xgb_imp = xgb_model.feature_importances_
    else:
        xgb_imp = np.zeros(len(feature_cols))

    df_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Random Forest": rf_imp,
        "XGBoost": xgb_imp,
    }).sort_values(by="XGBoost", ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=df_imp["Feature"],
            x=df_imp["Random Forest"],
            name="Random Forest (MDI)",
            orientation="h",
            marker_color=THEME_COLORS["secondary"],
        )
    )

    fig.add_trace(
        go.Bar(
            y=df_imp["Feature"],
            x=df_imp["XGBoost"],
            name="XGBoost (Gain)",
            orientation="h",
            marker_color=THEME_COLORS["primary"],
        )
    )

    fig.update_layout(barmode="group")

    return apply_custom_theme(
        fig,
        title="Relative Feature Importance Comparison",
        x_title="Normalized Importance",
        y_title="Feature Name",
        height=550,
    )


def render_shap_summary_chart(shap_values: np.ndarray, X: pd.DataFrame, model_name: str) -> go.Figure:
    """Renders mean absolute SHAP value ranking bar chart."""
    if shap_values is None or X.empty:
        fig = go.Figure()
        return apply_custom_theme(fig, title="SHAP Values Unavailable")

    mean_shap = np.nanmean(np.abs(shap_values), axis=0)
    df_shap = pd.DataFrame({
        "Feature": X.columns,
        "Mean |SHAP|": mean_shap,
    }).sort_values(by="Mean |SHAP|", ascending=True)

    fig = go.Figure(
        data=[
            go.Bar(
                y=df_shap["Feature"],
                x=df_shap["Mean |SHAP|"],
                orientation="h",
                marker=dict(
                    color=df_shap["Mean |SHAP|"],
                    colorscale=[[0, "#3A86FF"], [1, "#FF4D6D"]],
                ),
                text=[f"{v:.4f}" for v in df_shap["Mean |SHAP|"]],
                textposition="auto",
            )
        ]
    )

    return apply_custom_theme(
        fig,
        title=f"SHAP Feature Impact: {model_name} (Mean |SHAP Value|)",
        x_title="Mean |SHAP value| (Average Impact on Model Output Magnitude)",
        y_title="Predictor Feature",
        height=520,
        show_legend=False,
    )


def render_correlation_heatmap(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """Renders Pearson correlation matrix heatmap for environmental variables."""
    if df.empty:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Correlation Matrix Unavailable")

    available_features = [f for f in features if f in df.columns]
    corr = df[available_features].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(
                title=dict(text="r", font=dict(color="#FFFFFF", size=12)),
                tickfont=dict(color="#A0AEC0", size=10),
            ),
        )
    )

    fig.update_layout(yaxis=dict(autorange="reversed"))

    return apply_custom_theme(
        fig,
        title="Inter-Feature Pearson Correlation Heatmap",
        height=550,
        show_legend=False,
    )


def render_missingness_chart(missing_summary_df: pd.DataFrame) -> go.Figure:
    """Renders missingness rate percentage bar chart."""
    if missing_summary_df.empty:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Missingness Audit Unavailable")

    plot_df = missing_summary_df[missing_summary_df["missing_pct"] > 0].copy()
    if plot_df.empty:
        plot_df = missing_summary_df.head(10).copy()

    plot_df = plot_df.drop_duplicates(subset=["column"]).sort_values(by="missing_pct", ascending=True).reset_index(drop=True)

    colors = [
        THEME_COLORS["danger"] if pct > 30 else THEME_COLORS["warning"] if pct > 5 else THEME_COLORS["primary"]
        for pct in plot_df["missing_pct"]
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                y=plot_df["column"],
                x=plot_df["missing_pct"],
                orientation="h",
                marker_color=colors,
                text=[f"{pct:.1f}% ({cnt} nulls)" for pct, cnt in zip(plot_df["missing_pct"], plot_df["missing_count"])],
                textposition="auto",
            )
        ]
    )

    return apply_custom_theme(
        fig,
        title="Missing Values by Feature",
        x_title="Missingness Percentage (%)",
        y_title="Variable Name",
        height=450,
        show_legend=False,
    )


def render_spatial_purging_sensitivity_chart(sens_df: pd.DataFrame) -> go.Figure:
    """Renders dual-axis sensitivity chart: Purge Buffer vs Mean AUROC vs Retained Samples."""
    if sens_df.empty:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Spatial Purging Sensitivity Unavailable")

    grouped = sens_df.groupby("purge_buffer_km").agg({
        "rf_auroc": "mean",
        "xgb_auroc": "mean",
        "n_train_retained": "mean",
        "n_train_purged": "mean"
    }).reset_index()

    fig = go.Figure()

    # Retained Samples (Bar) on secondary y
    fig.add_trace(
        go.Bar(
            x=grouped["purge_buffer_km"].astype(str) + " km",
            y=grouped["n_train_retained"],
            name="Retained Train Samples (N)",
            marker_color="rgba(56, 189, 248, 0.25)",
            yaxis="y2"
        )
    )

    # RF AUROC line
    fig.add_trace(
        go.Scatter(
            x=grouped["purge_buffer_km"].astype(str) + " km",
            y=grouped["rf_auroc"],
            name="Random Forest AUROC",
            mode="lines+markers",
            line=dict(color=THEME_COLORS["primary"], width=3),
            marker=dict(size=8)
        )
    )

    # XGBoost AUROC line
    fig.add_trace(
        go.Scatter(
            x=grouped["purge_buffer_km"].astype(str) + " km",
            y=grouped["xgb_auroc"],
            name="XGBoost AUROC",
            mode="lines+markers",
            line=dict(color=THEME_COLORS["accent_purple"], width=3, dash="dot"),
            marker=dict(size=8)
        )
    )

    fig.update_layout(
        title="Spatial Purging Sensitivity (0 to 437.8 km)",
        yaxis=dict(title="Out-of-Fold AUROC", range=[0.90, 1.01]),
        yaxis2=dict(
            title="Retained Training Samples (N)",
            overlaying="y",
            side="right",
            range=[0, 240]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380
    )
    return apply_custom_theme(fig)


def render_fold_metrics_bar_chart(fold_df: pd.DataFrame, title: str = "Fold-Wise Evaluation Metrics") -> go.Figure:
    """Renders fold-by-fold AUROC & PR-AUC comparison highlighting Fold 4 single-class transparency."""
    if fold_df.empty:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Fold Metrics Unavailable")

    folds = [f"Fold {f}" for f in fold_df["fold"]]
    rf_auc = [f if pd.notna(f) else 0.0 for f in fold_df["rf_auroc"]]
    xgb_auc = [f if pd.notna(f) else 0.0 for f in fold_df["xgb_auroc"]]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Random Forest AUROC",
                x=folds,
                y=rf_auc,
                marker_color=THEME_COLORS["primary"],
                text=[f"{v:.3f}" if v > 0 else "Undefined (0 pos)" for v in rf_auc],
                textposition="auto"
            ),
            go.Bar(
                name="XGBoost AUROC",
                x=folds,
                y=xgb_auc,
                marker_color=THEME_COLORS["accent_purple"],
                text=[f"{v:.3f}" if v > 0 else "Undefined (0 pos)" for v in xgb_auc],
                textposition="auto"
            )
        ]
    )

    fig.add_annotation(
        x="Fold 4",
        y=0.15,
        text="Single-Class Test Partition<br>(N=37 neg, 0 pos in Kashmir cluster)",
        showarrow=True,
        arrowhead=2,
        arrowcolor=THEME_COLORS["warning"],
        font=dict(color=THEME_COLORS["warning"], size=10),
        bgcolor="rgba(30, 41, 59, 0.9)",
        bordercolor=THEME_COLORS["warning"]
    )

    fig.update_layout(
        barmode="group",
        yaxis_range=[0.0, 1.08],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380
    )
    return apply_custom_theme(fig, title=title, x_title="Spatial Outer Fold", y_title="AUROC Score")


def render_susceptibility_histogram(grid_df: pd.DataFrame) -> go.Figure:
    """Renders histogram of 0.5 deg regional grid susceptibility scores across 765 cells."""
    if grid_df.empty:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Susceptibility Grid Unavailable")

    prob_col = "susceptibility" if "susceptibility" in grid_df.columns else "pred_prob_rf"
    if prob_col not in grid_df.columns:
        fig = go.Figure()
        return apply_custom_theme(fig, title="Susceptibility Scores Unavailable")

    scores = grid_df[prob_col].dropna()

    fig = go.Figure(
        data=[
            go.Histogram(
                x=scores,
                nbinsx=25,
                marker=dict(
                    color=THEME_COLORS["primary"],
                    line=dict(color="#00F0FF", width=1)
                )
            )
        ]
    )

    median_val = float(scores.median())
    fig.add_vline(
        x=median_val,
        line_width=2,
        line_dash="dash",
        line_color=THEME_COLORS["warning"],
        annotation_text=f"Median: {median_val:.3f}",
        annotation_position="top right"
    )

    return apply_custom_theme(
        fig,
        title="Susceptibility Score Distribution (0.5° Grid)",
        x_title="Predicted Landslide Susceptibility Score [0.0 - 1.0]",
        y_title="Grid Cell Count",
        height=350
    )


