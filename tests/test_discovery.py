from pathlib import Path

from glioma_idh.discovery import (
    canonicalize_identifier,
    extract_visit_info,
    infer_subject_id,
    is_followup_identifier,
    is_segmentation,
    normalise_modality,
    segmentation_kind,
)


def test_canonicalize_identifier() -> None:
    assert canonicalize_identifier("UCSF-PDGM-001") == "UCSFPDGM1"
    assert canonicalize_identifier("UCSF-PDGM-0001") == "UCSFPDGM1"


def test_infer_subject_id_from_path() -> None:
    dataset_root = Path("/tmp/dataset")
    path = dataset_root / "UCSF-PDGM-001" / "t2.nii.gz"
    subject_id, method = infer_subject_id(path, dataset_root)
    assert subject_id == "UCSF-PDGM-001"
    assert method == "regex_match"


def test_segmentation_detection() -> None:
    path = Path("/tmp/UCSF-PDGM-001/tumor_seg.nii.gz")
    assert is_segmentation(path)
    assert segmentation_kind(path) == "tumor"
    assert normalise_modality(path) == "segmentation"


def test_modality_detection() -> None:
    assert normalise_modality(Path("/tmp/UCSF-PDGM-001/scan_T2_FLAIR.nii.gz")) == "flair"
    assert normalise_modality(Path("/tmp/UCSF-PDGM-001/scan_t1ce.nii.gz")) == "t1c"
    assert normalise_modality(Path("/tmp/UCSF-PDGM-001/scan_t2.nii.gz")) == "t2"


def test_extract_visit_info_for_followup() -> None:
    dataset_root = Path("/tmp/dataset")
    path = dataset_root / "UCSF-PDGM-0409_FU001d_nifti" / "UCSF-PDGM-0409_T1c.nii.gz"
    patient_id, visit_id, visit_type = extract_visit_info(path, dataset_root)
    assert patient_id == "UCSF-PDGM-0409"
    assert visit_id == "UCSF-PDGM-0409_FU001d_nifti"
    assert visit_type == "followup"
    assert is_followup_identifier(visit_id)
