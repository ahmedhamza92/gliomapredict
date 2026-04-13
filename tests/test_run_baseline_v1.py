import numpy as np
import pandas as pd

from glioma_idh.run_baseline_v1 import _compute_metrics, _encode_idh


def test_encode_idh_labels() -> None:
    encoded = _encode_idh(pd.Series(["wildtype", "mutant", "wildtype"]))
    assert encoded.tolist() == [0, 1, 0]


def test_compute_metrics() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    probability = np.array([0.1, 0.7, 0.6, 0.8], dtype=float)
    prediction = np.array([0, 1, 1, 1], dtype=int)
    metrics = _compute_metrics(y_true, probability, prediction)
    assert metrics["confusion_tn"] == 1
    assert metrics["confusion_fp"] == 1
    assert metrics["confusion_fn"] == 0
    assert metrics["confusion_tp"] == 2
    assert np.isclose(metrics["sensitivity"], 1.0)
    assert np.isclose(metrics["specificity"], 0.5)
