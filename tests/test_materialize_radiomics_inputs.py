import numpy as np

from glioma_idh.materialize_radiomics_inputs import crop_array, normalize_volume


def test_normalize_volume_preserves_zero_background() -> None:
    volume = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    normalized = normalize_volume(volume, clip_low=1.0, clip_high=3.0, mean_value=2.0, std_value=1.0)
    assert float(normalized[0, 0]) == 0.0
    assert np.allclose(normalized[volume != 0], np.array([-1.0, 0.0, 1.0], dtype=np.float32))


def test_crop_array() -> None:
    arr = np.arange(5 * 6 * 7).reshape(5, 6, 7)
    cropped = crop_array(arr, 1, 2, 3, 4, 5, 7)
    assert cropped.shape == (3, 3, 4)
    assert cropped[0, 0, 0] == arr[1, 2, 3]
