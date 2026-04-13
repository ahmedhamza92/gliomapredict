from glioma_idh.freeze_v1 import _select_modality_paths


def test_select_modality_paths_prefers_bias_variant() -> None:
    preferred, raw, bias, variant = _select_modality_paths(
        [
            "/tmp/UCSF-PDGM-0001_T2.nii.gz",
            "/tmp/UCSF-PDGM-0001_T2_bias.nii.gz",
        ]
    )
    assert preferred == "/tmp/UCSF-PDGM-0001_T2_bias.nii.gz"
    assert raw == "/tmp/UCSF-PDGM-0001_T2.nii.gz"
    assert bias == "/tmp/UCSF-PDGM-0001_T2_bias.nii.gz"
    assert variant == "bias"
