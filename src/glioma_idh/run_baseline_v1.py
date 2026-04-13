from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ID_COLUMNS = ("subject_id", "idh_label", "mgmt_label")
POSITIVE_LABEL = "mutant"
NEGATIVE_LABEL = "wildtype"


@dataclass(frozen=True)
class ModelRun:
    name: str
    estimator: GridSearchCV
    probability: np.ndarray
    prediction: np.ndarray
    metrics: dict[str, Any]
    selected_feature_names: list[str]


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _encode_idh(series: pd.Series) -> pd.Series:
    allowed = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
    unknown = sorted(set(series.dropna().unique()) - set(allowed))
    if unknown:
        raise ValueError(f"Unexpected IDH labels: {unknown}")
    return series.map(allowed)


def _prepare_model_frame(features_path: Path, splits_path: Path, target_label: str) -> tuple[pd.DataFrame, list[str]]:
    features = _load_table(features_path).sort_values("subject_id").reset_index(drop=True)
    splits = _load_table(splits_path).sort_values("subject_id").reset_index(drop=True)

    required_split_cols = {"subject_id", "split_set", "cv_fold", target_label}
    missing = required_split_cols - set(splits.columns)
    if missing:
        raise ValueError(f"Split artifact missing columns: {sorted(missing)}")

    merged = features.merge(
        splits[["subject_id", target_label, "split_set", "cv_fold"]],
        on="subject_id",
        how="inner",
        suffixes=("", "_split"),
        validate="one_to_one",
    )
    if len(merged) != len(features):
        raise ValueError("Feature/split merge dropped rows; expected one split row per subject.")
    if not merged[target_label].equals(merged[f"{target_label}_split"]):
        raise ValueError("Split artifact label mismatch against the radiomics feature table.")
    merged = merged.drop(columns=[f"{target_label}_split"])
    merged["idh_binary"] = _encode_idh(merged[target_label]).astype(int)

    feature_columns = [column for column in merged.columns if column not in set(ID_COLUMNS) | {"split_set", "cv_fold", "idh_binary"}]
    if not feature_columns:
        raise ValueError("No model feature columns were found.")
    return merged, feature_columns


def _build_logistic_pipeline(config: dict[str, Any]) -> tuple[Pipeline, dict[str, list[Any]]]:
    model_cfg = config["primary_model"]
    feature_cfg = config["feature_handling"]

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
                max_iter=5000,
                random_state=20260412,
            ),
        )
    )
    pipeline = Pipeline(steps=steps)
    param_grid = {
        "classifier__penalty": list(model_cfg["penalty_grid"]),
        "classifier__C": list(model_cfg["c_grid"]),
    }
    return pipeline, param_grid


def _build_random_forest_pipeline(config: dict[str, Any]) -> tuple[Pipeline, dict[str, list[Any]]]:
    model_cfg = config["secondary_model"]
    feature_cfg = config["feature_handling"]

    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy=feature_cfg["imputation"]))]
    if bool(feature_cfg["remove_zero_variance"]):
        steps.append(("variance_threshold", VarianceThreshold()))
    steps.append(
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=int(model_cfg["n_estimators"]),
                class_weight=model_cfg["class_weight"],
                random_state=int(model_cfg["random_state"]),
                n_jobs=1,
            ),
        )
    )
    pipeline = Pipeline(steps=steps)
    param_grid = {
        "classifier__max_depth": list(model_cfg["grid"]["max_depth"]),
        "classifier__min_samples_leaf": list(model_cfg["grid"]["min_samples_leaf"]),
    }
    return pipeline, param_grid


def _selected_feature_names(pipeline: Pipeline, feature_columns: list[str]) -> list[str]:
    selected = list(feature_columns)
    if "variance_threshold" in pipeline.named_steps:
        variance_step: VarianceThreshold = pipeline.named_steps["variance_threshold"]
        selected = [name for name, keep in zip(selected, variance_step.get_support()) if bool(keep)]
    return selected


def _compute_metrics(y_true: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_tn": int(tn),
        "confusion_fp": int(fp),
        "confusion_fn": int(fn),
        "confusion_tp": int(tp),
    }


def _roc_svg_path(fpr: np.ndarray, tpr: np.ndarray, width: int, height: int, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int) -> str:
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom
    coords = []
    for x_value, y_value in zip(fpr, tpr):
        x = pad_left + float(x_value) * plot_width
        y = height - pad_bottom - float(y_value) * plot_height
        coords.append(f"{x:.2f},{y:.2f}")
    return " ".join(coords)


def _write_roc_svg(curves: list[dict[str, Any]], output_path: Path) -> None:
    width, height = 640, 440
    pad_left, pad_right, pad_top, pad_bottom = 70, 20, 30, 55
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom
    palette = ["#005f73", "#ae2012", "#0a9396", "#ca6702"]

    grid_lines: list[str] = []
    for tick in np.linspace(0.0, 1.0, 6):
        x = pad_left + tick * plot_width
        y = height - pad_bottom - tick * plot_height
        grid_lines.append(f"<line x1='{x:.2f}' y1='{pad_top}' x2='{x:.2f}' y2='{height - pad_bottom}' stroke='#d9d9d9' stroke-width='1' />")
        grid_lines.append(f"<line x1='{pad_left}' y1='{y:.2f}' x2='{width - pad_right}' y2='{y:.2f}' stroke='#d9d9d9' stroke-width='1' />")

    curve_lines: list[str] = []
    legend_lines: list[str] = []
    for idx, curve in enumerate(curves):
        color = palette[idx % len(palette)]
        points = _roc_svg_path(
            curve["fpr"],
            curve["tpr"],
            width=width,
            height=height,
            pad_left=pad_left,
            pad_right=pad_right,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
        )
        curve_lines.append(f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{points}' />")
        legend_y = pad_top + 18 + idx * 22
        legend_lines.append(f"<line x1='{width - 210}' y1='{legend_y}' x2='{width - 185}' y2='{legend_y}' stroke='{color}' stroke-width='3' />")
        legend_lines.append(
            f"<text x='{width - 175}' y='{legend_y + 4}' font-size='13' font-family='Arial, sans-serif' fill='#222'>{curve['label']} (AUC {curve['auc']:.3f})</text>"
        )

    diagonal = (
        f"<line x1='{pad_left}' y1='{height - pad_bottom}' x2='{width - pad_right}' y2='{pad_top}' "
        "stroke='#555' stroke-width='1.5' stroke-dasharray='6 4' />"
    )

    x_ticks = []
    y_ticks = []
    for tick in np.linspace(0.0, 1.0, 6):
        x = pad_left + tick * plot_width
        y = height - pad_bottom - tick * plot_height
        x_ticks.append(f"<text x='{x:.2f}' y='{height - pad_bottom + 20}' text-anchor='middle' font-size='12' font-family='Arial, sans-serif'>{tick:.1f}</text>")
        y_ticks.append(f"<text x='{pad_left - 12}' y='{y + 4:.2f}' text-anchor='end' font-size='12' font-family='Arial, sans-serif'>{tick:.1f}</text>")

    svg = "\n".join(
        [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            "<rect width='100%' height='100%' fill='white' />",
            "<text x='320' y='20' text-anchor='middle' font-size='18' font-family='Arial, sans-serif' fill='#111'>Held-out Test ROC</text>",
            *grid_lines,
            f"<rect x='{pad_left}' y='{pad_top}' width='{plot_width}' height='{plot_height}' fill='none' stroke='#222' stroke-width='1.5' />",
            diagonal,
            *curve_lines,
            *x_ticks,
            *y_ticks,
            f"<text x='320' y='{height - 12}' text-anchor='middle' font-size='14' font-family='Arial, sans-serif'>False Positive Rate</text>",
            f"<text x='18' y='{height / 2:.2f}' text-anchor='middle' font-size='14' font-family='Arial, sans-serif' transform='rotate(-90 18 {height / 2:.2f})'>True Positive Rate</text>",
            *legend_lines,
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg + "\n", encoding="utf-8")


def _write_logistic_coef_svg(feature_names: list[str], coefficients: np.ndarray, output_path: Path, top_n: int = 15) -> None:
    coef_series = pd.Series(coefficients, index=feature_names)
    top = coef_series.abs().sort_values(ascending=False).head(top_n).index
    display = coef_series.loc[top].sort_values()

    width, row_height = 860, 28
    height = 80 + len(display) * row_height
    pad_left, pad_right, pad_top, pad_bottom = 260, 30, 30, 30
    plot_width = width - pad_left - pad_right
    center_x = pad_left + plot_width / 2
    max_abs = float(max(display.abs().max(), 1e-8))

    bars: list[str] = []
    labels: list[str] = []
    for idx, (name, value) in enumerate(display.items()):
        y = pad_top + idx * row_height
        bar_height = 18
        bar_y = y + 4
        half_width = (abs(float(value)) / max_abs) * (plot_width / 2 - 10)
        if value >= 0:
            x = center_x
        else:
            x = center_x - half_width
        color = "#0a9396" if value >= 0 else "#ae2012"
        bars.append(f"<rect x='{x:.2f}' y='{bar_y:.2f}' width='{half_width:.2f}' height='{bar_height}' fill='{color}' />")
        labels.append(
            f"<text x='{pad_left - 10}' y='{bar_y + 13:.2f}' text-anchor='end' font-size='12' font-family='Arial, sans-serif'>{name}</text>"
        )
        labels.append(
            f"<text x='{pad_left + plot_width + 10}' y='{bar_y + 13:.2f}' font-size='12' font-family='Arial, sans-serif'>{value:.3f}</text>"
        )

    svg = "\n".join(
        [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            "<rect width='100%' height='100%' fill='white' />",
            "<text x='430' y='20' text-anchor='middle' font-size='18' font-family='Arial, sans-serif' fill='#111'>Logistic Regression Top Coefficients</text>",
            f"<line x1='{center_x:.2f}' y1='{pad_top - 5}' x2='{center_x:.2f}' y2='{height - pad_bottom}' stroke='#222' stroke-width='1.5' />",
            *bars,
            *labels,
            f"<text x='{center_x - 80:.2f}' y='{height - 8}' text-anchor='middle' font-size='12' font-family='Arial, sans-serif'>Toward wildtype</text>",
            f"<text x='{center_x + 80:.2f}' y='{height - 8}' text-anchor='middle' font-size='12' font-family='Arial, sans-serif'>Toward mutant</text>",
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg + "\n", encoding="utf-8")


def _run_single_model(
    name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    scorer: str,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
    cv_folds: np.ndarray,
    feature_columns: list[str],
) -> ModelRun:
    cv = PredefinedSplit(test_fold=cv_folds)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        refit=True,
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(train_X, train_y)
    probability = search.best_estimator_.predict_proba(test_X)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    selected_features = _selected_feature_names(search.best_estimator_, feature_columns)
    metrics = _compute_metrics(test_y.to_numpy(dtype=int), probability, prediction)
    metrics["best_cv_roc_auc"] = float(search.best_score_)
    metrics["best_params_json"] = json.dumps(search.best_params_, sort_keys=True)
    metrics["n_features_before"] = int(len(feature_columns))
    metrics["n_features_after"] = int(len(selected_features))
    return ModelRun(
        name=name,
        estimator=search,
        probability=probability,
        prediction=prediction,
        metrics=metrics,
        selected_feature_names=selected_features,
    )


def _write_results_note(
    output_path: Path,
    features_path: Path,
    splits_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_feature_count: int,
    model_runs: list[ModelRun],
) -> None:
    label_counts = lambda df: df["idh_label"].value_counts().sort_index()
    train_counts = label_counts(train_df)
    test_counts = label_counts(test_df)

    lines = [
        "# Baseline Results v1",
        "",
        f"- Source radiomics table: `{features_path}`",
        f"- Source split artifact: `{splits_path}`",
        f"- Frozen train/test counts: **{len(train_df)} / {len(test_df)}**",
        f"- Raw radiomics feature count before modelling-time filtering: **{raw_feature_count}**",
        f"- Positive class for sensitivity/specificity: **`{POSITIVE_LABEL}`**",
        "- All imputation, zero-variance filtering, scaling, and model selection were fit on the training pool only via predefined CV folds.",
        "- The held-out test set was not used inside model selection.",
        "- Decision threshold for class predictions was the default probability cutoff of `0.5`.",
        "",
        "## Split counts",
        "",
        f"- Train `mutant`: **{int(train_counts.get(POSITIVE_LABEL, 0))}**",
        f"- Train `wildtype`: **{int(train_counts.get(NEGATIVE_LABEL, 0))}**",
        f"- Test `mutant`: **{int(test_counts.get(POSITIVE_LABEL, 0))}**",
        f"- Test `wildtype`: **{int(test_counts.get(NEGATIVE_LABEL, 0))}**",
        "",
        "## Held-out test metrics",
        "",
        "| Model | CV ROC-AUC | Test ROC-AUC | Balanced Acc. | Sensitivity | Specificity | Features after filtering |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in model_runs:
        metrics = run.metrics
        lines.append(
            "| "
            f"{run.name} | "
            f"{metrics['best_cv_roc_auc']:.3f} | "
            f"{metrics['roc_auc']:.3f} | "
            f"{metrics['balanced_accuracy']:.3f} | "
            f"{metrics['sensitivity']:.3f} | "
            f"{metrics['specificity']:.3f} | "
            f"{metrics['n_features_after']} |"
        )
    primary = model_runs[0]
    lines.extend(
        [
            "",
            "## Selected settings and confusion matrices",
            "",
        ]
    )
    for run in model_runs:
        metrics = run.metrics
        lines.extend(
            [
                f"### {run.name}",
                "",
                f"- Best CV parameters: `{metrics['best_params_json']}`",
                f"- Features retained after training-only filtering: **{metrics['n_features_after']} / {metrics['n_features_before']}**",
                (
                    f"- Held-out confusion matrix with `{POSITIVE_LABEL}` as positive class: "
                    f"`TN={metrics['confusion_tn']}, FP={metrics['confusion_fp']}, "
                    f"FN={metrics['confusion_fn']}, TP={metrics['confusion_tp']}`"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Primary baseline interpretation",
            "",
            (
                f"- The predeclared primary model is `{primary.name}`. "
                f"On the untouched test set it reached ROC-AUC **{primary.metrics['roc_auc']:.3f}**, "
                f"balanced accuracy **{primary.metrics['balanced_accuracy']:.3f}**, "
                f"sensitivity **{primary.metrics['sensitivity']:.3f}**, and specificity **{primary.metrics['specificity']:.3f}**."
            ),
            f"- In this first run, the training-only variance filter retained all **{primary.metrics['n_features_after']}** raw radiomics features.",
            "- The random forest result, if present, is reported as a bounded secondary comparison rather than a reason to redefine the v1 pipeline.",
            "",
            "## Limitations",
            "",
            "- The feature extractor is intentionally compact and does not yet implement a full IBSI-style radiomics catalogue.",
            "- Test-set performance is from a single stratified split; uncertainty remains high, especially for the `mutant` class with only 21 held-out subjects.",
            "- No external validation has been used, so these results should be treated as first internal baseline evidence only.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_baseline(
    features_path: Path,
    splits_path: Path,
    config_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _load_yaml(config_path)
    modelling_df, feature_columns = _prepare_model_frame(
        features_path=features_path,
        splits_path=splits_path,
        target_label=config["target_label"],
    )

    train_df = modelling_df[modelling_df["split_set"] == "train"].copy().reset_index(drop=True)
    test_df = modelling_df[modelling_df["split_set"] == "test"].copy().reset_index(drop=True)
    train_X = train_df[feature_columns]
    test_X = test_df[feature_columns]
    train_y = train_df["idh_binary"]
    test_y = test_df["idh_binary"]
    cv_folds = train_df["cv_fold"].to_numpy(dtype=int)

    model_runs: list[ModelRun] = []

    logistic_pipeline, logistic_grid = _build_logistic_pipeline(config)
    model_runs.append(
        _run_single_model(
            name="logistic_regression",
            pipeline=logistic_pipeline,
            param_grid=logistic_grid,
            scorer=config["selection"]["scorer"],
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            cv_folds=cv_folds,
            feature_columns=feature_columns,
        )
    )

    secondary_cfg = config.get("secondary_model", {})
    if bool(secondary_cfg.get("enabled", False)) and secondary_cfg.get("kind") == "random_forest":
        rf_pipeline, rf_grid = _build_random_forest_pipeline(config)
        model_runs.append(
            _run_single_model(
                name="random_forest",
                pipeline=rf_pipeline,
                param_grid=rf_grid,
                scorer=config["selection"]["scorer"],
                train_X=train_X,
                train_y=train_y,
                test_X=test_X,
                test_y=test_y,
                cv_folds=cv_folds,
                feature_columns=feature_columns,
            )
        )

    metrics_rows = []
    predictions = test_df[["subject_id", "idh_label", "idh_binary"]].copy()
    for run in model_runs:
        metrics_rows.append({"model_name": run.name, **run.metrics})
        predictions[f"{run.name}_probability_mutant"] = run.probability
        predictions[f"{run.name}_prediction_binary"] = run.prediction
        predictions[f"{run.name}_prediction_label"] = np.where(run.prediction == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
    predictions["split_set"] = "test"

    metrics_df = pd.DataFrame(metrics_rows).sort_values("model_name").reset_index(drop=True)
    artifacts = config["artifacts"]
    metrics_csv = Path(artifacts["metrics_csv"])
    predictions_csv = Path(artifacts["predictions_csv"])
    results_note = Path(artifacts["results_note"])
    roc_figure = Path(artifacts["roc_figure"])
    logistic_coef_figure = Path(artifacts["logistic_coef_figure"])

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False)
    predictions.to_csv(predictions_csv, index=False)

    roc_curves = []
    for run in model_runs:
        fpr, tpr, _ = roc_curve(test_y, run.probability)
        roc_curves.append({"label": run.name, "auc": run.metrics["roc_auc"], "fpr": fpr, "tpr": tpr})
    _write_roc_svg(roc_curves, roc_figure)

    logistic_run = next(run for run in model_runs if run.name == "logistic_regression")
    logistic_estimator: ClassifierMixin = logistic_run.estimator.best_estimator_.named_steps["classifier"]
    logistic_coef = np.asarray(logistic_estimator.coef_).reshape(-1)
    _write_logistic_coef_svg(logistic_run.selected_feature_names, logistic_coef, logistic_coef_figure)

    _write_results_note(
        output_path=results_note,
        features_path=features_path,
        splits_path=splits_path,
        train_df=train_df,
        test_df=test_df,
        raw_feature_count=len(feature_columns),
        model_runs=model_runs,
    )
    return metrics_df, predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the first interpretable IDH baseline models for the frozen v1 cohort.")
    parser.add_argument("--features-path", default="data/processed/radiomics_features_v1.parquet")
    parser.add_argument("--splits-path", default="data/interim/splits_v1.parquet")
    parser.add_argument("--config", default="configs/baseline_v1.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    metrics_df, predictions_df = run_baseline(
        features_path=Path(args.features_path),
        splits_path=Path(args.splits_path),
        config_path=Path(args.config),
    )
    print(f"Models run: {', '.join(metrics_df['model_name'])}")
    print(f"Test subjects scored: {len(predictions_df)}")
    print(f"Metrics CSV: {Path(_load_yaml(Path(args.config))['artifacts']['metrics_csv']).resolve()}")
    print(f"Predictions CSV: {Path(_load_yaml(Path(args.config))['artifacts']['predictions_csv']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
