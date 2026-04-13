from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

NIFTI_SUFFIXES = (".nii", ".nii.gz")
TABULAR_SUFFIXES = (".csv", ".tsv", ".xlsx")

GENERIC_DIR_NAMES = {
    "images",
    "image",
    "imaging",
    "nifti",
    "nii",
    "raw",
    "rawdata",
    "derivatives",
    "masks",
    "mask",
    "labels",
    "label",
    "segs",
    "seg",
    "anat",
    "struct",
    "mr",
    "mri",
}

SUBJECT_HINT_PATTERNS = [
    re.compile(r"(?i)\b(?:sub(?:ject)?|patient|case)[-_]?[a-z0-9]+\b"),
    re.compile(r"(?i)\bucsf[-_]?pdgm[-_]?[a-z0-9]+\b"),
]

UCSF_PATIENT_PATTERN = re.compile(r"(?i)(UCSF[-_]?PDGM[-_]?\d+)")
FOLLOWUP_PATTERN = re.compile(r"(?i)_FU\d+d(?:_|$)")

SEGMENTATION_PATTERNS = [
    re.compile(r"(?i)(segmentation|mask|label|roi|tumou?r|lesion)"),
]

TUMOR_SEGMENTATION_PATTERN = re.compile(r"(?i)(tumou?r[_-]?segmentation|tumou?r|lesion)")
BRAIN_SEGMENTATION_PATTERN = re.compile(r"(?i)(brain(?:_parenchyma)?_segmentation)")

MODALITY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("t1c", re.compile(r"(?i)(t1ce|t1c|t1gd|t1_gd|t1post|post[-_]?contrast|gad|t1[-_]?post)")),
    ("flair", re.compile(r"(?i)(flair|t2[-_]?flair)")),
    ("t2", re.compile(r"(?i)(^|[^a-z0-9])t2([^a-z0-9]|$)")),
    ("t1", re.compile(r"(?i)(^|[^a-z0-9])t1([^a-z0-9]|$)")),
    ("adc", re.compile(r"(?i)(^|[^a-z0-9])adc([^a-z0-9]|$)")),
    ("dwi", re.compile(r"(?i)(^|[^a-z0-9])(dwi|tracew)([^a-z0-9]|$)")),
    ("swi", re.compile(r"(?i)(^|[^a-z0-9])swi([^a-z0-9]|$)")),
    ("asl", re.compile(r"(?i)(^|[^a-z0-9])asl([^a-z0-9]|$)")),
    ("dti_fa", re.compile(r"(?i)(^|[^a-z0-9])fa([^a-z0-9]|$)")),
    ("dti_md", re.compile(r"(?i)(^|[^a-z0-9])md([^a-z0-9]|$)")),
    ("dti_l1", re.compile(r"(?i)(^|[^a-z0-9])l1([^a-z0-9]|$)")),
    ("dti_l2", re.compile(r"(?i)(^|[^a-z0-9])l2([^a-z0-9]|$)")),
    ("dti_l3", re.compile(r"(?i)(^|[^a-z0-9])l3([^a-z0-9]|$)")),
    ("dti_noreg", re.compile(r"(?i)(dti.*noreg|noreg)")),
    ("cbv", re.compile(r"(?i)(^|[^a-z0-9])(cbv|rcbv|perfusion)([^a-z0-9]|$)")),
]


def iter_files(root: Path, suffixes: tuple[str, ...]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if any(lower_name.endswith(suffix) for suffix in suffixes):
            yield path


def canonicalize_identifier(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = UCSF_PATIENT_PATTERN.search(text)
    if match:
        numeric_match = re.search(r"(\d+)", match.group(1))
        if numeric_match:
            return f"UCSFPDGM{int(numeric_match.group(1))}"
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def extract_patient_id(value: str) -> str:
    match = UCSF_PATIENT_PATTERN.search(value)
    if match:
        return match.group(1)
    return value


def is_followup_identifier(value: str) -> bool:
    return bool(FOLLOWUP_PATTERN.search(value))


def extract_visit_info(path: Path, dataset_root: Path) -> tuple[str, str, str]:
    relative_parts = path.relative_to(dataset_root).parts
    visit_id = relative_parts[0] if relative_parts else path.parent.name
    patient_id = extract_patient_id(visit_id)
    visit_type = "followup" if is_followup_identifier(visit_id) else "baseline"
    return patient_id, visit_id, visit_type


def infer_subject_id(path: Path, dataset_root: Path) -> tuple[str, str]:
    relative_parts = path.relative_to(dataset_root).parts
    for part in relative_parts:
        for pattern in SUBJECT_HINT_PATTERNS:
            match = pattern.search(part)
            if match:
                return match.group(0), "regex_match"

    for part in reversed(relative_parts[:-1]):
        normalized = canonicalize_identifier(part)
        if normalized and part.lower() not in GENERIC_DIR_NAMES:
            return part, "directory_fallback"

    return path.stem.replace(".nii", ""), "filename_fallback"


def is_segmentation(path: Path) -> bool:
    candidate = str(path).lower()
    return any(pattern.search(candidate) for pattern in SEGMENTATION_PATTERNS)


def segmentation_kind(path: Path) -> str:
    candidate = str(path).lower()
    if BRAIN_SEGMENTATION_PATTERN.search(candidate):
        return "brain"
    if TUMOR_SEGMENTATION_PATTERN.search(candidate):
        return "tumor"
    if is_segmentation(path):
        return "other"
    return "none"


def normalise_modality(path: Path) -> str:
    candidate = path.name.lower()
    if is_segmentation(path):
        return "segmentation"
    for modality, pattern in MODALITY_RULES:
        if pattern.search(candidate):
            return modality
    for part in reversed(path.parts[:-1]):
        text = part.lower()
        for modality, pattern in MODALITY_RULES:
            if pattern.search(text):
                return modality
    return "unknown"
