import numpy as np
import pandas as pd

from glioma_idh.analyze_logistic_robustness_v1 import (
    _expand_coefficients,
    _metric_summary,
    _stratified_bootstrap_indices,
)


def test_expand_coefficients_preserves_full_feature_order() -> None:
    series = _expand_coefficients(
        full_feature_names=["a", "b", "c"],
        selected_feature_names=["b", "c"],
        selected_coefficients=np.array([1.5, -2.0]),
    )
    assert series.index.tolist() == ["a", "b", "c"]
    assert np.allclose(series.to_numpy(), np.array([0.0, 1.5, -2.0]))


def test_metric_summary_percentiles() -> None:
    summary = _metric_summary(pd.Series([0.1, 0.2, 0.3, 0.4]))
    assert np.isclose(summary["mean"], 0.25)
    assert np.isclose(summary["median"], 0.25)
    assert summary["p025"] <= summary["median"] <= summary["p975"]


def test_stratified_bootstrap_indices_preserve_class_counts() -> None:
    rng = np.random.default_rng(7)
    y_true = np.array([0, 0, 0, 1, 1], dtype=int)
    sample_idx = _stratified_bootstrap_indices(y_true, rng)
    sampled = y_true[sample_idx]
    assert int((sampled == 0).sum()) == 3
    assert int((sampled == 1).sum()) == 2
