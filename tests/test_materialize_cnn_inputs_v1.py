import numpy as np

from glioma_idh.materialize_cnn_inputs_v1 import build_slice_indices, select_center_slice


def test_select_center_slice_prefers_midpoint_on_tie() -> None:
    mask = np.zeros((4, 4, 7), dtype=np.uint8)
    mask[:, :, 2] = 1
    mask[:, :, 4] = 1
    assert select_center_slice(mask) == 2


def test_build_slice_indices_clamps_to_bounds() -> None:
    assert build_slice_indices(center_slice=0, depth=5, offsets=[-2, -1, 0, 1, 2]) == [0, 0, 0, 1, 2]
