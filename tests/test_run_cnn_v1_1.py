import numpy as np

from glioma_idh.run_cnn_v1_1 import _safe_divide


def test_safe_divide() -> None:
    assert np.isclose(_safe_divide(3.0, 2.0), 1.5)
    assert np.isnan(_safe_divide(1.0, 0.0))
