from glioma_idh.materialize_preproc_index import _compute_bbox, _compute_crop_bounds, _compute_norm_params

import numpy as np


def test_compute_bbox() -> None:
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:5, 3:8, 1:4] = 1
    mins, maxs, sizes = _compute_bbox(mask)
    assert mins == (2, 3, 1)
    assert maxs == (4, 7, 3)
    assert sizes == (3, 5, 3)


def test_compute_crop_bounds() -> None:
    starts, ends, sizes, clipped = _compute_crop_bounds(
        bbox_mins=(2, 3, 1),
        bbox_maxs_inclusive=(4, 7, 3),
        shape=(10, 10, 10),
        padding=(4, 4, 4),
    )
    assert starts == (0, 0, 0)
    assert ends == (9, 10, 8)
    assert sizes == (9, 10, 8)
    assert clipped is True


def test_compute_norm_params() -> None:
    image = np.array([0, 1, 2, 3, 100], dtype=np.float32)
    norm = _compute_norm_params(image, 0, 100)
    assert norm.nonzero_count == 4
    assert norm.clip_low_value == 1.0
    assert norm.clip_high_value == 100.0
    assert norm.valid is True
