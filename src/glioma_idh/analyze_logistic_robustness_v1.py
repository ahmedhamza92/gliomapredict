from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from glioma_idh.run_baseline_v1 import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    _compute_metrics,
    _load_table,
    _load_yaml,
    _prepare_model_frame,
)


ROBUSTNESS_METRICS = ("roc_auc", "balanced_accuracy", "sensitivity", "specificity")


def _load_frozen_logistic_params(metrics_csv: Path) -> tuple[dict[str, Any], pd.Series]:
    metrics_df = pd.read_csv(metrics_csv)
    logistic_row = metrics_df.loc[metrics_df["model_name"] == "logistic_regression"]
    if logistic_row.empty:
        raise ValueError("Could not find logistic_regression row in baseline metrics CSV.")
    row = logistic_row.iloc[0]
    params = json.loads(str(row["best_params_json"]))
    return params, row


def _build_fixed_logistic_pipeline(baseline_config: dict[str, Any], fixed_params: dict[str, Any]) -> Pipeline:
    model_cfg = baseline_config["primary_model"]
    feature_cfg = baseline_config["feature_handling"]

    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy=feature_cfg["imputation"]))]
    if bool(feature_cfg["remove_zero_variance"]):
        steps.append(("variance_threshold", VarianceThreshold()))
    if bool(feature_cfg["scale_for_linear_models"]):
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "classifier",
            LogisticRegression(
                solver=model_cfg["solver"],
                class_weight=model_cfg["class_weight"],
                C=float(fixed_params["classifier__C"]),
                penalty=str(fixed_params["classifier__penalty"]),
                max_iter=5000,
                random_state=20260412,
            ),
        )
    )
    return Pipeline(steps)


def _selected_feature_names(pipeline: Pipeline, feature_columns: list[str]) -> list[str]:
    selected = list(feature_columns)
    if "variance_threshold" in pipeline.named_steps:
        variance_step: VarianceThreshold = pipeline.named_steps["variance_threshold"]
        selected = [name for name, keep in zip(selected, variance_step.get_support()) if bool(keep)]
    return selected


def _expand_coefficients(
    full_feature_names: list[str],
    selected_feature_names: list[str],
    selected_coefficients: np.ndarray,
) -> pd.Series:
    coefficients = pd.Series(0.0, index=full_feature_names, dtype=float)
    coefficients.loc[selected_feature_names] = np.asarray(selected_coefficients, dtype=float)
    return coefficients


def _metric_summary(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "median": float(series.median()),
        "p025": float(series.quantile(0.025)),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "p975": float(series.quantile(0.975)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    indices = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(indices)
    return indices


def _write_interval_svg(
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    subtitle: str,
    point_key: str,
    lower_key: str,
    upper_key: str,
) -> None:
    width = 760
    row_height = 70
    height = 110 + len(rows) * row_height
    pad_left = 190
    pad_right = 40
    pad_top = 60
    pad_bottom = 40
    plot_width = width - pad_left - pad_right

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white' />",
        f"<text x='{width / 2:.1f}' y='24' text-anchor='middle' font-size='20' font-family='Arial, sans-serif'>{title}</text>",
        f"<text x='{width / 2:.1f}' y='46' text-anchor='middle' font-size='12' font-family='Arial, sans-serif' fill='#555'>{subtitle}</text>",
    ]
    for tick in np.linspace(0.0, 1.0, 6):
        x = pad_left + tick * plot_width
        lines.append(f"<line x1='{x:.2f}' y1='{pad_top}' x2='{x:.2f}' y2='{height - pad_bottom}' stroke='#dddddd' stroke-width='1' />")
        lines.append(f"<text x='{x:.2f}' y='{height - 12}' text-anchor='middle' font-size='12' font-family='Arial, sans-serif'>{tick:.1f}</text>")

    lines.append(f"<line x1='{pad_left}' y1='{height - pad_bottom}' x2='{width - pad_right}' y2='{height - pad_bottom}' stroke='#222' stroke-width='1.5' />")
    palette = {"roc_auc": "#005f73", "balanced_accuracy": "#0a9396", "sensitivity": "#ca6702", "specificity": "#ae2012"}
    for idx, row in enumerate(rows):
        y = pad_top + idx * row_height + 20
        lower_x = pad_left + float(row[lower_key]) * plot_width
        upper_x = pad_left + float(row[upper_key]) * plot_width
        point_x = pad_left + float(row[point_key]) * plot_width
        color = palette.get(str(row["metric"]), "#333333")
        lines.append(f"<text x='{pad_left - 14}' y='{y + 5:.2f}' text-anchor='end' font-size='13' font-family='Arial, sans-serif'>{row['metric']}</text>")
        lines.append(f"<line x1='{lower_x:.2f}' y1='{y:.2f}' x2='{upper_x:.2f}' y2='{y:.2f}' stroke='{color}' stroke-width='4' />")
        lines.append(f"<circle cx='{point_x:.2f}' cy='{y:.2f}' r='5' fill='{color}' />")
        lines.append(
            f"<text x='{width - pad_right + 4}' y='{y + 5:.2f}' font-size='12' font-family='Arial, sans-serif' fill='#444'>"
            f"{row[point_key]:.3f} [{row[lower_key]:.3f}, {row[upper_key]:.3f}]</text>"
        )
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_feature_stability_svg(summary_df: pd.DataFrame, output_path: Path, top_n: int) -> None:
    display = summary_df.head(top_n).copy()
    width = 920
    row_height = 34
    height = 110 + len(display) * row_height
    pad_left = 300
    pad_right = 140
    pad_top = 60
    pad_bottom = 35
    plot_width = width - pad_left - pad_right

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white' />",
        f"<text x='{width / 2:.1f}' y='24' text-anchor='middle' font-size='20' font-family='Arial, sans-serif'>Logistic Feature Stability</text>",
        f"<text x='{width / 2:.1f}' y='46' text-anchor='middle' font-size='12' font-family='Arial, sans-serif' fill='#555'>Bar length = top-k frequency across repeated training-pool fits; color = mean coefficient sign</text>",
    ]
    for tick in np.linspace(0.0, 1.0, 6):
        x = pad_left + tick * plot_width
        lines.append(f"<line x1='{x:.2f}' y1='{pad_top}' x2='{x:.2f}' y2='{height - pad_bottom}' stroke='#e0e0e0' stroke-width='1' />")
        lines.append(f"<text x='{x:.2f}' y='{height - 10}' text-anchor='middle' font-size='12' font-family='Arial, sans-serif'>{tick:.1f}</text>")
    for idx, row in enumerate(display.itertuples(index=False)):
        y = pad_top + idx * row_height + 8
        width_bar = float(row.top_k_frequency) * plot_width
        color = "#0a9396" if float(row.mean_coefficient) >= 0 else "#ae2012"
        lines.append(f"<text x='{pad_left - 10}' y='{y + 14:.2f}' text-anchor='end' font-size='12' font-family='Arial, sans-serif'>{row.feature_name}</text>")
        lines.append(f"<rect x='{pad_left}' y='{y:.2f}' width='{width_bar:.2f}' height='18' fill='{color}' />")
        lines.append(
            f"<text x='{width - pad_right + 5}' y='{y + 14:.2f}' font-size='12' font-family='Arial, sans-serif' fill='#444'>"
            f"top-k {row.top_k_frequency:.2f}, mean coef {row.mean_coefficient:.3f}</text>"
        )
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_robustness_note(
    output_path: Path,
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    fixed_params: dict[str, Any],
) -> None:
    lines = [
        "# Logistic Robustness v1",
        "",
        "- Analysis scope: frozen primary logistic regression baseline only.",
        f"- Training-pool procedure: repeated stratified {n_splits}-fold cross-validation on the frozen training pool.",
        f"- Repeats: **{n_repeats}**, total validation fits: **{len(runs_df)}**",
        f"- Fixed logistic settings copied from the frozen baseline: `{json.dumps(fixed_params, sort_keys=True)}`",
        "- No held-out test subjects were used in this robustness analysis.",
        "",
        "## Metric variability across repeated training-pool validation folds",
        "",
        "| Metric | Mean | SD | Median | 2.5% | 97.5% | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.mean:.3f} | {row.std:.3f} | {row.median:.3f} | {row.p025:.3f} | {row.p975:.3f} | {row.min:.3f} | {row.max:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These distributions quantify how much the frozen logistic baseline varies across repeated patient-level re-partitions of the training pool.",
            "- They are robustness diagnostics, not replacements for the untouched held-out test result.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bootstrap_note(
    output_path: Path,
    summary_df: pd.DataFrame,
    n_resamples: int,
    point_estimates: dict[str, float],
) -> None:
    lines = [
        "# Logistic Bootstrap Confidence Intervals v1",
        "",
        "- Analysis scope: frozen held-out test predictions from the primary logistic baseline only.",
        f"- Bootstrap method: stratified percentile bootstrap with **{n_resamples}** resamples.",
        "- Resampling was performed within the observed `wildtype` and `mutant` test classes to preserve the fixed test-set class counts.",
        "- No model refitting and no hyperparameter updates were performed during this procedure.",
        "",
        "## Held-out metric intervals",
        "",
        "| Metric | Point estimate | 2.5% | 97.5% |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(f"| {row.metric} | {point_estimates[row.metric]:.3f} | {row.ci_lower:.3f} | {row.ci_upper:.3f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These intervals describe uncertainty around the frozen held-out test metrics under bootstrap resampling of the fixed predictions.",
            "- They should be read as uncertainty around the current internal test estimate, not as external validation.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_feature_note(
    output_path: Path,
    summary_df: pd.DataFrame,
    top_k: int,
    nonzero_threshold: float,
) -> None:
    top_display = summary_df.head(15)
    lines = [
        "# Feature Stability v1",
        "",
        "- Analysis scope: repeated training-pool fits of the frozen primary logistic baseline.",
        f"- Top-rank frequency threshold: top **{top_k}** absolute coefficients per fit.",
        f"- Non-zero threshold: **{nonzero_threshold}**",
        "",
        "## Top stable features",
        "",
        "| Feature | Full-train coef | Mean coef | SD coef | Top-k freq | Sign consistency | Non-zero freq |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_display.itertuples(index=False):
        lines.append(
            f"| {row.feature_name} | {row.full_train_coefficient:.3f} | {row.mean_coefficient:.3f} | {row.std_coefficient:.3f} | "
            f"{row.top_k_frequency:.2f} | {row.sign_consistency:.2f} | {row.nonzero_frequency:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Because the frozen primary model uses an `l2` penalty, coefficient non-zero frequency is expected to be dense; top-rank frequency and sign consistency are therefore more informative stability diagnostics.",
            "- Features near the top of this table are the most repeatedly influential under patient-level training resampling, not causal biomarkers.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_logistic_robustness(config_path: Path) -> dict[str, pd.DataFrame]:
    config = _load_yaml(config_path)
    source = config["source_artifacts"]
    artifacts = config["artifacts"]

    baseline_config = _load_yaml(Path(source["baseline_config"]))
    fixed_params, frozen_logistic_row = _load_frozen_logistic_params(Path(source["baseline_metrics_csv"]))
    modelling_df, feature_columns = _prepare_model_frame(
        features_path=Path(source["features_path"]),
        splits_path=Path(source["splits_path"]),
        target_label=baseline_config["target_label"],
    )
    train_df = modelling_df.loc[modelling_df["split_set"] == "train"].copy().reset_index(drop=True)
    train_X = train_df[feature_columns]
    train_y = train_df["idh_binary"].astype(int)

    robustness_cfg = config["training_pool_robustness"]
    splitter = RepeatedStratifiedKFold(
        n_splits=int(robustness_cfg["n_splits"]),
        n_repeats=int(robustness_cfg["n_repeats"]),
        random_state=int(robustness_cfg["random_state"]),
    )
    top_k = int(config["feature_stability"]["top_k"])
    nonzero_threshold = float(config["feature_stability"]["nonzero_threshold"])

    robustness_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for split_id, (fit_idx, val_idx) in enumerate(splitter.split(train_X, train_y)):
        repeat_id = split_id // int(robustness_cfg["n_splits"])
        fold_id = split_id % int(robustness_cfg["n_splits"])
        pipeline = _build_fixed_logistic_pipeline(baseline_config, fixed_params)
        X_fit = train_X.iloc[fit_idx]
        y_fit = train_y.iloc[fit_idx]
        X_val = train_X.iloc[val_idx]
        y_val = train_y.iloc[val_idx]
        pipeline.fit(X_fit, y_fit)

        probability = pipeline.predict_proba(X_val)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metrics = _compute_metrics(y_val.to_numpy(dtype=int), probability, prediction)
        robustness_rows.append(
            {
                "split_id": int(split_id),
                "repeat_id": int(repeat_id),
                "fold_id": int(fold_id),
                "n_fit": int(len(fit_idx)),
                "n_validation": int(len(val_idx)),
                **metrics,
            }
        )

        selected_names = _selected_feature_names(pipeline, feature_columns)
        classifier = pipeline.named_steps["classifier"]
        coefficients = _expand_coefficients(feature_columns, selected_names, np.asarray(classifier.coef_).reshape(-1))
        ranks = coefficients.abs().rank(method="first", ascending=False)
        top_features = set(ranks.loc[ranks <= top_k].index)
        for feature_name in feature_columns:
            value = float(coefficients.loc[feature_name])
            coefficient_rows.append(
                {
                    "split_id": int(split_id),
                    "repeat_id": int(repeat_id),
                    "fold_id": int(fold_id),
                    "feature_name": feature_name,
                    "coefficient": value,
                    "abs_coefficient": abs(value),
                    "nonzero": bool(abs(value) > nonzero_threshold),
                    "positive": bool(value > nonzero_threshold),
                    "negative": bool(value < -nonzero_threshold),
                    "top_k": bool(feature_name in top_features),
                }
            )

    robustness_runs = pd.DataFrame(robustness_rows)
    robustness_summary = pd.DataFrame(
        [{"metric": metric, **_metric_summary(robustness_runs[metric])} for metric in ROBUSTNESS_METRICS]
    )

    predictions = pd.read_csv(Path(source["baseline_predictions_csv"]))
    y_test = predictions["idh_binary"].to_numpy(dtype=int)
    probability = predictions["logistic_regression_probability_mutant"].to_numpy(dtype=float)
    prediction = predictions["logistic_regression_prediction_binary"].to_numpy(dtype=int)
    bootstrap_cfg = config["bootstrap_test_ci"]
    rng = np.random.default_rng(int(bootstrap_cfg["random_state"]))
    bootstrap_rows: list[dict[str, Any]] = []
    for bootstrap_id in range(int(bootstrap_cfg["n_resamples"])):
        sample_idx = _stratified_bootstrap_indices(y_test, rng)
        sampled_metrics = _compute_metrics(y_test[sample_idx], probability[sample_idx], prediction[sample_idx])
        bootstrap_rows.append({"bootstrap_id": int(bootstrap_id), **{metric: sampled_metrics[metric] for metric in ROBUSTNESS_METRICS}})
    bootstrap_resamples = pd.DataFrame(bootstrap_rows)
    lower_q = (1.0 - float(bootstrap_cfg["ci_alpha"])) / 2.0
    upper_q = 1.0 - lower_q
    point_estimates = {metric: float(frozen_logistic_row[metric]) for metric in ROBUSTNESS_METRICS}
    bootstrap_summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "point_estimate": point_estimates[metric],
                "ci_lower": float(bootstrap_resamples[metric].quantile(lower_q)),
                "ci_upper": float(bootstrap_resamples[metric].quantile(upper_q)),
                "bootstrap_mean": float(bootstrap_resamples[metric].mean()),
                "bootstrap_std": float(bootstrap_resamples[metric].std(ddof=1)),
            }
            for metric in ROBUSTNESS_METRICS
        ]
    )

    full_train_pipeline = _build_fixed_logistic_pipeline(baseline_config, fixed_params)
    full_train_pipeline.fit(train_X, train_y)
    full_train_selected = _selected_feature_names(full_train_pipeline, feature_columns)
    full_train_coefficients = _expand_coefficients(
        feature_columns,
        full_train_selected,
        np.asarray(full_train_pipeline.named_steps["classifier"].coef_).reshape(-1),
    )
    full_train_ranks = full_train_coefficients.abs().rank(method="first", ascending=False)
    coefficient_runs = pd.DataFrame(coefficient_rows)
    coefficient_stability = (
        coefficient_runs.groupby("feature_name")
        .agg(
            mean_coefficient=("coefficient", "mean"),
            std_coefficient=("coefficient", "std"),
            median_coefficient=("coefficient", "median"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            median_abs_coefficient=("abs_coefficient", "median"),
            nonzero_frequency=("nonzero", "mean"),
            positive_frequency=("positive", "mean"),
            negative_frequency=("negative", "mean"),
            top_k_frequency=("top_k", "mean"),
        )
        .reset_index()
    )
    coefficient_stability["sign_consistency"] = coefficient_stability[["positive_frequency", "negative_frequency"]].max(axis=1)
    coefficient_stability["full_train_coefficient"] = coefficient_stability["feature_name"].map(full_train_coefficients.to_dict())
    coefficient_stability["full_train_abs_rank"] = coefficient_stability["feature_name"].map(full_train_ranks.to_dict()).astype(int)
    coefficient_stability["full_train_top_k"] = coefficient_stability["full_train_abs_rank"].le(top_k)
    coefficient_stability = coefficient_stability.sort_values(
        ["top_k_frequency", "sign_consistency", "mean_abs_coefficient"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    pd.DataFrame(robustness_runs).to_csv(Path(artifacts["robustness_runs_csv"]), index=False)
    pd.DataFrame(robustness_summary).to_csv(Path(artifacts["robustness_summary_csv"]), index=False)
    pd.DataFrame(bootstrap_resamples).to_csv(Path(artifacts["bootstrap_resamples_csv"]), index=False)
    pd.DataFrame(bootstrap_summary).to_csv(Path(artifacts["bootstrap_ci_csv"]), index=False)
    pd.DataFrame(coefficient_runs).to_csv(Path(artifacts["coefficient_runs_csv"]), index=False)
    pd.DataFrame(coefficient_stability).to_csv(Path(artifacts["coefficient_stability_csv"]), index=False)

    _write_robustness_note(
        output_path=Path(artifacts["robustness_note"]),
        runs_df=robustness_runs,
        summary_df=robustness_summary,
        n_splits=int(robustness_cfg["n_splits"]),
        n_repeats=int(robustness_cfg["n_repeats"]),
        fixed_params=fixed_params,
    )
    _write_bootstrap_note(
        output_path=Path(artifacts["bootstrap_note"]),
        summary_df=bootstrap_summary,
        n_resamples=int(bootstrap_cfg["n_resamples"]),
        point_estimates=point_estimates,
    )
    _write_feature_note(
        output_path=Path(artifacts["feature_note"]),
        summary_df=coefficient_stability,
        top_k=top_k,
        nonzero_threshold=nonzero_threshold,
    )

    _write_interval_svg(
        rows=robustness_summary.to_dict(orient="records"),
        output_path=Path(artifacts["robustness_figure"]),
        title="Training-Pool Robustness",
        subtitle=f"Repeated stratified {robustness_cfg['n_splits']}-fold CV on frozen training pool ({robustness_cfg['n_repeats']} repeats)",
        point_key="mean",
        lower_key="p025",
        upper_key="p975",
    )
    _write_interval_svg(
        rows=bootstrap_summary.to_dict(orient="records"),
        output_path=Path(artifacts["bootstrap_figure"]),
        title="Held-out Test Bootstrap Intervals",
        subtitle=f"Frozen logistic predictions with stratified percentile bootstrap ({bootstrap_cfg['n_resamples']} resamples)",
        point_key="point_estimate",
        lower_key="ci_lower",
        upper_key="ci_upper",
    )
    _write_feature_stability_svg(
        summary_df=coefficient_stability,
        output_path=Path(artifacts["feature_figure"]),
        top_n=15,
    )
    return {
        "robustness_runs": robustness_runs,
        "robustness_summary": robustness_summary,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_summary": bootstrap_summary,
        "coefficient_runs": coefficient_runs,
        "coefficient_stability": coefficient_stability,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run robustness and feature-stability analysis for the frozen logistic v1 baseline.")
    parser.add_argument("--config", default="configs/logistic_robustness_v1.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    outputs = analyze_logistic_robustness(Path(args.config))
    print(f"Robustness runs: {len(outputs['robustness_runs'])}")
    print(f"Bootstrap resamples: {len(outputs['bootstrap_resamples'])}")
    print(f"Coefficient rows: {len(outputs['coefficient_runs'])}")
    print(f"Top stable feature: {outputs['coefficient_stability'].iloc[0]['feature_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
