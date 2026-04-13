from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import pandas as pd

from .discovery import (
    TABULAR_SUFFIXES,
    canonicalize_identifier,
    extract_visit_info,
    is_followup_identifier,
    iter_files,
    normalise_modality,
    segmentation_kind,
)


@dataclass
class MetadataTableSummary:
    path: str
    rows: int
    columns: list[str]
    identifier_column: str | None
    label_columns: dict[str, str]


@dataclass
class BuildArtifacts:
    manifest: pd.DataFrame
    metadata_tables: list[MetadataTableSummary]
    file_type_counts: dict[str, int]
    image_root: str
    visit_directory_count: int
    followup_directory_count: int
    unresolved_issues: list[str]


def _read_table(path: Path) -> pd.DataFrame:
    lower = path.name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if lower.endswith(".xlsx"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def _normalise_column_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _find_identifier_column(columns: list[str]) -> str | None:
    preferred = {
        "id",
        "subjectid",
        "subject",
        "patientid",
        "patient",
        "caseid",
        "case",
        "braintumorid",
    }
    normalized = {_normalise_column_name(column): column for column in columns}
    for candidate in preferred:
        if candidate in normalized:
            return normalized[candidate]
    for normalized_name, original in normalized.items():
        if any(token in normalized_name for token in ("subject", "patient", "case")):
            return original
    return None


def _find_label_columns(columns: list[str]) -> dict[str, str]:
    normalized = {_normalise_column_name(column): column for column in columns}
    mapping: dict[str, str] = {}
    for normalized_name, original in normalized.items():
        if "idh" in normalized_name and "idh" not in mapping:
            mapping["idh"] = original
        elif "mgmt" in normalized_name and "mgmt" not in mapping:
            mapping["mgmt"] = original
        elif "grade" in normalized_name and "grade" not in mapping:
            mapping["grade"] = original
        elif "age" in normalized_name and "age" not in mapping:
            mapping["age"] = original
        elif "sex" in normalized_name and "sex" not in mapping:
            mapping["sex"] = original
        elif "gender" in normalized_name and "gender" not in mapping:
            mapping["gender"] = original
    return mapping


def _normalise_label(value: Any, label_type: str) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    if text in {"unknown", "indeterminate", "not available", "na", "nan"}:
        return None

    if label_type == "idh":
        if "wildtype" in text or text in {"wt", "negative", "no", "0", "idhwt"}:
            return "wildtype"
        if "mutant" in text or "mutat" in text or "r132" in text or "r172" in text or text in {"positive", "yes", "1", "idhmut", "idhm"}:
            return "mutant"
        if text.startswith("idh") and "wildtype" not in text:
            return "mutant"
        return str(value)

    if label_type == "mgmt":
        if "unmethylated" in text or text in {"negative", "neg", "no", "0"}:
            return "unmethylated"
        if "methylated" in text or text in {"positive", "pos", "yes", "1"}:
            return "methylated"
        return str(value)

    return str(value)


def _discover_image_root(dataset_root: Path) -> Path:
    visit_dir_pattern = re.compile(r"(?i)^UCSF-PDGM-.*_nifti$")

    direct_visit_dirs = [path for path in dataset_root.iterdir() if path.is_dir() and visit_dir_pattern.match(path.name)]
    if direct_visit_dirs:
        return dataset_root

    candidates: list[tuple[int, Path]] = []
    for child in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        visit_count = sum(1 for path in child.iterdir() if path.is_dir() and visit_dir_pattern.match(path.name))
        if visit_count:
            candidates.append((visit_count, child))

    if len(candidates) == 1:
        return candidates[0][1]
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    return dataset_root


def _collect_metadata(dataset_root: Path) -> tuple[dict[str, dict[str, Any]], list[MetadataTableSummary], list[str]]:
    metadata_by_subject: dict[str, dict[str, Any]] = {}
    metadata_tables: list[MetadataTableSummary] = []
    unresolved: list[str] = []
    duplicate_subject_rows: Counter[str] = Counter()

    for table_path in sorted(iter_files(dataset_root, TABULAR_SUFFIXES)):
        try:
            table = _read_table(table_path)
        except Exception as exc:
            unresolved.append(f"Could not read metadata table {table_path}: {exc}")
            continue

        identifier_column = _find_identifier_column(list(table.columns))
        label_columns = _find_label_columns(list(table.columns))
        metadata_tables.append(
            MetadataTableSummary(
                path=str(table_path),
                rows=int(len(table)),
                columns=[str(column) for column in table.columns],
                identifier_column=identifier_column,
                label_columns=label_columns,
            )
        )

        if not identifier_column:
            unresolved.append(f"Skipped metadata table without recognizable subject identifier column: {table_path}")
            continue

        if not label_columns:
            unresolved.append(f"Metadata table had identifiers but no recognizable label columns: {table_path}")
            continue

        for _, row in table.iterrows():
            subject_value = row.get(identifier_column)
            canonical_subject_id = canonicalize_identifier(subject_value)
            if not canonical_subject_id:
                continue

            is_followup = is_followup_identifier(str(subject_value))
            candidate = {
                "metadata_subject_id": str(subject_value),
                "metadata_sources": {str(table_path)},
                "metadata_row_ids": {str(subject_value)},
                "metadata_row_count": 1,
                "_is_followup": is_followup,
            }

            for logical_name, column_name in label_columns.items():
                raw_value = row.get(column_name)
                if logical_name in {"idh", "mgmt"}:
                    candidate[f"{logical_name}_raw"] = raw_value if not pd.isna(raw_value) else None
                    candidate[f"{logical_name}_label"] = _normalise_label(raw_value, logical_name)
                else:
                    candidate[logical_name] = raw_value if not pd.isna(raw_value) else None

            existing = metadata_by_subject.get(canonical_subject_id)
            if existing is None:
                metadata_by_subject[canonical_subject_id] = candidate
                continue

            duplicate_subject_rows[canonical_subject_id] += 1
            existing["metadata_sources"].update(candidate["metadata_sources"])
            existing["metadata_row_ids"].update(candidate["metadata_row_ids"])
            existing["metadata_row_count"] = int(existing.get("metadata_row_count", 1)) + 1

            if existing.get("_is_followup", False) and not is_followup:
                accumulated_sources = set(existing["metadata_sources"])
                accumulated_row_ids = set(existing["metadata_row_ids"])
                accumulated_row_count = int(existing["metadata_row_count"])
                for key, value in candidate.items():
                    if key.startswith("_"):
                        continue
                    existing[key] = value
                existing["metadata_sources"] = accumulated_sources
                existing["metadata_row_ids"] = accumulated_row_ids
                existing["metadata_row_count"] = accumulated_row_count
                existing["_is_followup"] = False

    for record in metadata_by_subject.values():
        record["metadata_sources"] = sorted(record["metadata_sources"])
        record["metadata_row_ids"] = sorted(record["metadata_row_ids"])
        record.pop("_is_followup", None)
    if duplicate_subject_rows:
        unresolved.append(
            f"{len(duplicate_subject_rows)} patient identifiers had duplicate metadata rows after patient-level normalization; non-follow-up rows were preferred where available."
        )
    return metadata_by_subject, metadata_tables, unresolved


def _scan_file_types(dataset_root: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if lower_name.endswith(".nii.gz"):
            counts[".nii.gz"] += 1
        else:
            counts[path.suffix.lower() or "<no suffix>"] += 1
    return dict(sorted(counts.items()))


def build_manifest(dataset_root: Path) -> BuildArtifacts:
    image_root = _discover_image_root(dataset_root)
    image_rows: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for nifti_path in sorted(iter_files(image_root, (".nii", ".nii.gz"))):
        patient_id, visit_id, visit_type = extract_visit_info(nifti_path, image_root)
        image_rows.append(
            {
                "path": str(nifti_path),
                "subject_id": patient_id,
                "canonical_subject_id": canonicalize_identifier(patient_id),
                "visit_id": visit_id,
                "visit_type": visit_type,
                "modality": normalise_modality(nifti_path),
                "segmentation_kind": segmentation_kind(nifti_path),
            }
        )

    if not image_rows:
        raise FileNotFoundError(f"No NIfTI files were found under {dataset_root}")

    image_table = pd.DataFrame(image_rows)
    metadata_by_subject, metadata_tables, metadata_unresolved = _collect_metadata(dataset_root)
    unresolved.extend(metadata_unresolved)
    file_type_counts = _scan_file_types(dataset_root)

    visit_directory_count = int(sum(1 for path in image_root.iterdir() if path.is_dir()))
    followup_directory_count = int(image_table.loc[image_table["visit_type"] == "followup", "visit_id"].nunique())
    if followup_directory_count:
        unresolved.append(
            f"{followup_directory_count} follow-up visit directories were discovered and excluded from primary pre-operative completeness counts."
        )

    all_modalities = sorted(
        modality
        for modality in image_table["modality"].dropna().unique()
        if modality != "segmentation"
    )
    subject_records: list[dict[str, Any]] = []

    for canonical_subject_id, subject_table in image_table.groupby("canonical_subject_id", dropna=False):
        if not canonical_subject_id:
            unresolved.append("At least one NIfTI file could not be mapped to a non-empty canonical subject identifier.")
            continue

        patient_names = [value for value in subject_table["subject_id"].tolist() if value]
        subject_id = Counter(patient_names).most_common(1)[0][0] if patient_names else canonical_subject_id

        visit_rows = (
            subject_table[["visit_id", "visit_type"]]
            .drop_duplicates()
            .sort_values(["visit_type", "visit_id"])
            .to_dict(orient="records")
        )
        baseline_visits = [row["visit_id"] for row in visit_rows if row["visit_type"] == "baseline"]
        primary_visit_id = baseline_visits[0] if baseline_visits else visit_rows[0]["visit_id"]
        primary_visit_table = subject_table[subject_table["visit_id"] == primary_visit_id]
        followup_visits = [row["visit_id"] for row in visit_rows if row["visit_type"] == "followup"]

        modality_map: dict[str, list[str]] = defaultdict(list)
        tumor_segmentation_paths: list[str] = []
        brain_segmentation_paths: list[str] = []
        other_segmentation_paths: list[str] = []

        for row in primary_visit_table.to_dict(orient="records"):
            seg_kind = row["segmentation_kind"]
            if seg_kind == "tumor":
                tumor_segmentation_paths.append(row["path"])
            elif seg_kind == "brain":
                brain_segmentation_paths.append(row["path"])
            elif seg_kind == "other":
                other_segmentation_paths.append(row["path"])
            else:
                modality_map[row["modality"]].append(row["path"])

        metadata = metadata_by_subject.get(canonical_subject_id, {})
        record: dict[str, Any] = {
            "subject_id": subject_id,
            "canonical_subject_id": canonical_subject_id,
            "n_visit_dirs": int(len(visit_rows)),
            "primary_visit_id": primary_visit_id,
            "primary_visit_type": "baseline" if primary_visit_id in baseline_visits else "followup",
            "visit_ids_json": json.dumps([row["visit_id"] for row in visit_rows]),
            "followup_visit_ids_json": json.dumps(followup_visits),
            "n_primary_visit_nifti_files": int(len(primary_visit_table)),
            "n_total_subject_nifti_files": int(len(subject_table)),
            "n_modalities": int(sum(1 for paths in modality_map.values() if paths)),
            "modalities_present": ";".join(sorted(modality for modality, paths in modality_map.items() if paths)),
            "modality_file_map_json": json.dumps({key: sorted(value) for key, value in sorted(modality_map.items())}),
            "n_tumor_segmentation_files": int(len(tumor_segmentation_paths)),
            "has_tumor_segmentation": bool(tumor_segmentation_paths),
            "tumor_segmentation_paths_json": json.dumps(sorted(tumor_segmentation_paths)),
            "n_brain_segmentation_files": int(len(brain_segmentation_paths)),
            "brain_segmentation_paths_json": json.dumps(sorted(brain_segmentation_paths)),
            "n_other_segmentation_files": int(len(other_segmentation_paths)),
            "other_segmentation_paths_json": json.dumps(sorted(other_segmentation_paths)),
            "metadata_subject_id": metadata.get("metadata_subject_id"),
            "metadata_source_files": ";".join(metadata.get("metadata_sources", [])),
            "metadata_row_count": int(metadata.get("metadata_row_count", 0)),
            "metadata_row_ids_json": json.dumps(metadata.get("metadata_row_ids", [])),
            "idh_raw": metadata.get("idh_raw"),
            "idh_label": metadata.get("idh_label"),
            "mgmt_raw": metadata.get("mgmt_raw"),
            "mgmt_label": metadata.get("mgmt_label"),
            "grade_raw": metadata.get("grade"),
            "age": metadata.get("age"),
            "sex": metadata.get("sex") or metadata.get("gender"),
        }

        for modality in all_modalities:
            record[f"has_{modality}"] = bool(modality_map.get(modality))
            record[f"n_{modality}_files"] = int(len(modality_map.get(modality, [])))

        subject_records.append(record)

    manifest = pd.DataFrame(subject_records).sort_values("subject_id").reset_index(drop=True)

    image_subjects = set(manifest["canonical_subject_id"].dropna())
    metadata_subjects = set(metadata_by_subject)
    unmatched_metadata_subjects = sorted(metadata_subjects - image_subjects)
    if unmatched_metadata_subjects:
        unresolved.append(
            f"{len(unmatched_metadata_subjects)} metadata subjects did not match any discovered image subject using conservative identifier normalization."
        )

    if not metadata_tables:
        unresolved.append("No metadata tables were discovered under the dataset root, so label counts remain zero in the manifest.")

    unknown_modality_count = int((image_table["modality"] == "unknown").sum())
    if unknown_modality_count:
        unresolved.append(f"{unknown_modality_count} NIfTI files could not be mapped to a known modality label and were recorded as unknown.")

    return BuildArtifacts(
        manifest=manifest,
        metadata_tables=metadata_tables,
        file_type_counts=file_type_counts,
        image_root=str(image_root),
        visit_directory_count=visit_directory_count,
        followup_directory_count=followup_directory_count,
        unresolved_issues=unresolved,
    )


def write_manifest(manifest: pd.DataFrame, csv_path: Path, parquet_path: Path) -> tuple[bool, list[str]]:
    unresolved: list[str] = []
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(csv_path, index=False)

    parquet_written = False
    try:
        manifest.to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as exc:
        unresolved.append(f"Parquet output was not written because no working parquet engine is available: {exc}")

    return parquet_written, unresolved
