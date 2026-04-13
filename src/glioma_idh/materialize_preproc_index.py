from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOCKED_MODALITIES = ("flair", "t1c", "t2")


@dataclass
class NormParams:
    nonzero_count: int
    clip_low_value: float
    clip_high_value: float
    mean_after_clip: float
    std_after_clip: float
    valid: bool


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_nifti(path: Path):
    import nibabel as nib

    return nib.load(str(path))


def _compute_bbox(binary_mask: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    coords = np.argwhere(binary_mask > 0)
    if coords.size == 0:
        raise ValueError("Binary ROI mask is empty.")

    mins = tuple(int(value) for value in coords.min(axis=0))
    maxs = tuple(int(value) for value in coords.max(axis=0))
    sizes = tuple(int(max_v - min_v + 1) for min_v, max_v in zip(mins, maxs))
    return mins, maxs, sizes


def _compute_crop_bounds(
    bbox_mins: tuple[int, int, int],
    bbox_maxs_inclusive: tuple[int, int, int],
    shape: tuple[int, int, int],
    padding: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], bool]:
    starts: list[int] = []
    ends_exclusive: list[int] = []
    clipped = False

    for axis in range(3):
        unclipped_start = bbox_mins[axis] - padding[axis]
        unclipped_end = bbox_maxs_inclusive[axis] + 1 + padding[axis]
        start = max(0, unclipped_start)
        end = min(shape[axis], unclipped_end)
        if start != unclipped_start or end != unclipped_end:
            clipped = True
        starts.append(int(start))
        ends_exclusive.append(int(end))

    sizes = tuple(int(end - start) for start, end in zip(starts, ends_exclusive))
    return tuple(starts), tuple(ends_exclusive), sizes, clipped


def _compute_norm_params(image: np.ndarray, clip_low_pct: float, clip_high_pct: float) -> NormParams:
    nonzero = image[image != 0]
    if nonzero.size == 0:
        return NormParams(0, np.nan, np.nan, np.nan, np.nan, False)

    low_value = float(np.percentile(nonzero, clip_low_pct))
    high_value = float(np.percentile(nonzero, clip_high_pct))
    clipped = np.clip(nonzero.astype(np.float32), low_value, high_value)
    mean_value = float(clipped.mean())
    std_value = float(clipped.std())
    valid = bool(np.isfinite(std_value) and std_value > 0)
    return NormParams(
        nonzero_count=int(nonzero.size),
        clip_low_value=low_value,
        clip_high_value=high_value,
        mean_after_clip=mean_value,
        std_after_clip=std_value,
        valid=valid,
    )


def _write_binary_roi_mask(source_mask_path: Path, target_mask_path: Path) -> tuple[str, int, str, bool]:
    import nibabel as nib

    source_img = nib.load(str(source_mask_path))
    source_arr = np.asarray(source_img.dataobj)
    binary_mask = (source_arr > 0).astype(np.uint8)
    target_mask_path.parent.mkdir(parents=True, exist_ok=True)

    header = source_img.header.copy()
    header.set_data_dtype(np.uint8)
    binary_img = nib.Nifti1Image(binary_mask, source_img.affine, header=header)
    nib.save(binary_img, str(target_mask_path))

    unique_values = tuple(int(value) for value in np.unique(source_arr))
    unique_labels = ";".join(str(value) for value in unique_values)
    label_set_expected = set(unique_values) <= {0, 1, 2, 4}
    return str(target_mask_path.resolve()), int(binary_mask.sum()), unique_labels, label_set_expected


def _schema_rows() -> list[tuple[str, str, str]]:
    return [
        ("subject_id", "string", "Frozen cohort subject identifier."),
        ("idh_label", "string", "Binary v1 target label."),
        ("mgmt_label", "string", "Optional carried-through secondary label when available."),
        ("primary_visit_id", "string", "Primary pre-operative visit identifier used for all canonical paths."),
        ("flair_path", "string", "Canonical FLAIR path for downstream use."),
        ("t1c_path", "string", "Canonical T1c path for downstream use."),
        ("t2_path", "string", "Canonical T2 path for downstream use."),
        ("binary_roi_mask_path", "string", "Canonical binary whole-tumour ROI mask path."),
        ("roi_strategy", "string", "Binary ROI rule used to create the mask."),
        ("grid_shape_x/grid_shape_y/grid_shape_z", "int", "Reference image grid shape in voxels."),
        ("grid_spacing_x/grid_spacing_y/grid_spacing_z", "float", "Reference image voxel spacing in mm."),
        ("bbox_min_* / bbox_max_*_inclusive / bbox_size_*", "int", "Tight whole-tumour bounding box on the reference grid."),
        ("crop_start_* / crop_end_*_exclusive / crop_size_*", "int", "Padded crop coordinates for array slicing on the reference grid."),
        ("<modality>_norm_clip_low_value", "float", "Per-image non-zero lower clipping value."),
        ("<modality>_norm_clip_high_value", "float", "Per-image non-zero upper clipping value."),
        ("<modality>_norm_mean_after_clip", "float", "Per-image mean after clipping."),
        ("<modality>_norm_std_after_clip", "float", "Per-image std after clipping."),
        ("<modality>_nonzero_voxel_count", "int", "Number of non-zero voxels used for normalization."),
        ("qc_*", "bool", "Reproducibility and geometry checks for downstream gating."),
        ("cohort_config_path / preprocessing_config_path", "string", "Config provenance for deterministic regeneration."),
    ]


def _write_schema_note(
    output_path: Path,
    index_df: pd.DataFrame,
    cohort_config_path: Path,
    preprocessing_config_path: Path,
    roi_mask_dir: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_lines = [
        "# Preprocessing Index v1",
        "",
        "## Generation inputs",
        "",
        f"- Cohort config: `{cohort_config_path}`",
        f"- Preprocessing config: `{preprocessing_config_path}`",
        f"- Frozen cohort rows: **{len(index_df)}**",
        f"- Binary ROI directory: `{roi_mask_dir}`",
        "",
        "## Exact schema",
        "",
        "| Column group | Type | Meaning |",
        "| --- | --- | --- |",
    ]
    for column_group, dtype_name, meaning in _schema_rows():
        schema_lines.append(f"| `{column_group}` | `{dtype_name}` | {meaning} |")

    schema_lines.extend(
        [
            "",
            "## Downstream consumption",
            "",
            "- Treat each row as the authoritative per-subject preprocessing contract for v1.",
            "- Load only `flair_path`, `t1c_path`, `t2_path`, and `binary_roi_mask_path`; do not rediscover files from raw directories downstream.",
            "- Apply intensity preprocessing per modality using that row's clip low/high values, then z-score using that row's mean/std values.",
            "- Use `crop_start_*` and `crop_end_*_exclusive` directly as Python slicing bounds on the canonical grid.",
            "- Respect `qc_all_pass`; rows with `False` should be blocked or reviewed before feature extraction or CNN data generation.",
            "- `qc_padded_crop_clipped`, `qc_has_followup_visits`, and `qc_multiple_metadata_rows` are attention flags rather than automatic exclusion criteria in the current frozen cohort.",
        ]
    )

    output_path.write_text("\n".join(schema_lines) + "\n", encoding="utf-8")


def materialize_index(
    cohort_csv: Path,
    cohort_config_path: Path,
    preprocessing_config_path: Path,
    output_csv: Path,
    output_parquet: Path,
    roi_mask_dir: Path,
    schema_note_path: Path,
) -> pd.DataFrame:
    cohort_config = _load_yaml(cohort_config_path)
    preprocessing_config = _load_yaml(preprocessing_config_path)
    cohort = pd.read_csv(cohort_csv).sort_values("subject_id").reset_index(drop=True)

    expected_modalities = tuple(preprocessing_config["canonical_inputs"]["modalities"])
    if expected_modalities != LOCKED_MODALITIES:
        raise ValueError(f"Unexpected locked modalities in preprocessing config: {expected_modalities}")
    cohort_modalities = tuple(cohort_config["selection"]["locked_modalities"])
    if cohort_modalities != expected_modalities:
        raise ValueError(f"Cohort config and preprocessing config disagree on locked modalities: {cohort_modalities} vs {expected_modalities}")

    expected_shape = tuple(int(value) for value in preprocessing_config["geometry"]["expected_shape"])
    expected_spacing = tuple(float(value) for value in preprocessing_config["geometry"]["expected_spacing_mm"])
    clip_low_pct = float(preprocessing_config["normalization"]["clip_percentiles"]["low"])
    clip_high_pct = float(preprocessing_config["normalization"]["clip_percentiles"]["high"])
    padding = tuple(int(preprocessing_config["cnn_crop_generation"]["padding_voxels"][axis]) for axis in ("x", "y", "z"))
    preferred_variant = preprocessing_config["canonical_inputs"]["structural_variant"]
    cohort_preferred_variant = cohort_config["selection"]["preferred_structural_variant"]
    if cohort_preferred_variant != preferred_variant:
        raise ValueError(
            f"Cohort config and preprocessing config disagree on canonical structural variant: {cohort_preferred_variant} vs {preferred_variant}"
        )
    roi_strategy = preprocessing_config["roi"]["binary_rule"]

    rows: list[dict[str, Any]] = []
    for subject in cohort.to_dict(orient="records"):
        source_mask_path = Path(subject["tumor_segmentation_path"])
        binary_roi_mask_path = roi_mask_dir / f"{subject['subject_id']}_roi_whole_tumour.nii.gz"
        binary_roi_mask_path_str, roi_voxel_count, source_mask_labels, source_mask_labels_expected = _write_binary_roi_mask(
            source_mask_path, binary_roi_mask_path
        )

        mask_img = _load_nifti(binary_roi_mask_path)
        binary_mask = np.asarray(mask_img.dataobj, dtype=np.uint8)
        bbox_mins, bbox_maxs, bbox_sizes = _compute_bbox(binary_mask)
        crop_starts, crop_ends_exclusive, crop_sizes, crop_clipped = _compute_crop_bounds(
            bbox_mins=bbox_mins,
            bbox_maxs_inclusive=bbox_maxs,
            shape=tuple(int(value) for value in binary_mask.shape),
            padding=padding,
        )

        grid_shape = tuple(int(value) for value in mask_img.shape)
        grid_spacing = tuple(float(value) for value in mask_img.header.get_zooms()[:3])
        qc_mask_labels_expected = source_mask_labels_expected
        qc_geometry_matches_config = grid_shape == expected_shape
        qc_spacing_matches_config = all(abs(a - b) < 1e-6 for a, b in zip(grid_spacing, expected_spacing))
        qc_roi_nonempty = roi_voxel_count > 0

        row: dict[str, Any] = {
            "subject_id": subject["subject_id"],
            "canonical_subject_id": subject["canonical_subject_id"],
            "primary_visit_id": subject["primary_visit_id"],
            "idh_label": subject["idh_label"],
            "mgmt_label": subject["mgmt_label"],
            "roi_strategy": roi_strategy,
            "binary_roi_mask_path": binary_roi_mask_path_str,
            "source_tumor_mask_path": str(source_mask_path),
            "source_mask_labels": source_mask_labels,
            "roi_voxel_count": roi_voxel_count,
            "grid_shape_x": grid_shape[0],
            "grid_shape_y": grid_shape[1],
            "grid_shape_z": grid_shape[2],
            "grid_spacing_x": grid_spacing[0],
            "grid_spacing_y": grid_spacing[1],
            "grid_spacing_z": grid_spacing[2],
            "bbox_min_x": bbox_mins[0],
            "bbox_min_y": bbox_mins[1],
            "bbox_min_z": bbox_mins[2],
            "bbox_max_x_inclusive": bbox_maxs[0],
            "bbox_max_y_inclusive": bbox_maxs[1],
            "bbox_max_z_inclusive": bbox_maxs[2],
            "bbox_size_x": bbox_sizes[0],
            "bbox_size_y": bbox_sizes[1],
            "bbox_size_z": bbox_sizes[2],
            "crop_start_x": crop_starts[0],
            "crop_start_y": crop_starts[1],
            "crop_start_z": crop_starts[2],
            "crop_end_x_exclusive": crop_ends_exclusive[0],
            "crop_end_y_exclusive": crop_ends_exclusive[1],
            "crop_end_z_exclusive": crop_ends_exclusive[2],
            "crop_size_x": crop_sizes[0],
            "crop_size_y": crop_sizes[1],
            "crop_size_z": crop_sizes[2],
            "crop_padding_x": padding[0],
            "crop_padding_y": padding[1],
            "crop_padding_z": padding[2],
            "norm_clip_low_percentile": clip_low_pct,
            "norm_clip_high_percentile": clip_high_pct,
            "qc_roi_nonempty": qc_roi_nonempty,
            "qc_mask_labels_expected": qc_mask_labels_expected,
            "qc_geometry_matches_config": qc_geometry_matches_config,
            "qc_spacing_matches_config": qc_spacing_matches_config,
            "qc_padded_crop_clipped": crop_clipped,
            "qc_has_followup_visits": bool(subject["n_visit_dirs"] > 1),
            "qc_multiple_metadata_rows": bool(subject["metadata_row_count"] > 1),
            "cohort_config_path": str(cohort_config_path.resolve()),
            "preprocessing_config_path": str(preprocessing_config_path.resolve()),
            "frozen_cohort_csv_path": str(cohort_csv.resolve()),
        }

        modalities_aligned = True
        variants_match = True
        norms_valid = True
        for modality in LOCKED_MODALITIES:
            image_path = Path(subject[f"{modality}_path"])
            row[f"{modality}_path"] = str(image_path)
            row[f"{modality}_selected_variant"] = subject[f"{modality}_selected_variant"]

            image_img = _load_nifti(image_path)
            image_arr = np.asarray(image_img.dataobj, dtype=np.float32)
            image_shape = tuple(int(value) for value in image_img.shape)
            image_spacing = tuple(float(value) for value in image_img.header.get_zooms()[:3])
            modalities_aligned &= image_shape == grid_shape
            modalities_aligned &= all(abs(a - b) < 1e-6 for a, b in zip(image_spacing, grid_spacing))
            variants_match &= subject[f"{modality}_selected_variant"] == preferred_variant

            norm = _compute_norm_params(image_arr, clip_low_pct, clip_high_pct)
            norms_valid &= norm.valid

            row[f"{modality}_norm_nonzero_voxel_count"] = norm.nonzero_count
            row[f"{modality}_norm_clip_low_value"] = norm.clip_low_value
            row[f"{modality}_norm_clip_high_value"] = norm.clip_high_value
            row[f"{modality}_norm_mean_after_clip"] = norm.mean_after_clip
            row[f"{modality}_norm_std_after_clip"] = norm.std_after_clip
            row[f"{modality}_qc_norm_valid"] = norm.valid

        row["qc_modalities_aligned_to_mask"] = modalities_aligned
        row["qc_selected_variants_match_config"] = variants_match
        row["qc_norm_valid"] = norms_valid
        row["qc_all_pass"] = bool(
            qc_roi_nonempty
            and qc_mask_labels_expected
            and qc_geometry_matches_config
            and qc_spacing_matches_config
            and modalities_aligned
            and variants_match
            and norms_valid
        )
        rows.append(row)

    index_df = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(output_csv, index=False)
    index_df.to_parquet(output_parquet, index=False)
    _write_schema_note(
        output_path=schema_note_path,
        index_df=index_df,
        cohort_config_path=cohort_config_path,
        preprocessing_config_path=preprocessing_config_path,
        roi_mask_dir=roi_mask_dir,
    )
    return index_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the authoritative preprocessing index for the frozen v1 cohort.")
    parser.add_argument("--cohort-config", default="configs/cohort_v1.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing_v1.yaml")
    parser.add_argument("--cohort-csv", default="data/interim/cohort_v1.csv")
    parser.add_argument("--output-csv", default="data/processed/v1_preproc_index.csv")
    parser.add_argument("--output-parquet", default="data/processed/v1_preproc_index.parquet")
    parser.add_argument("--roi-mask-dir", default="data/processed/roi_masks_v1")
    parser.add_argument("--schema-note", default="reports/preprocessing_index_v1.md")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    index_df = materialize_index(
        cohort_csv=Path(args.cohort_csv),
        cohort_config_path=Path(args.cohort_config),
        preprocessing_config_path=Path(args.preprocessing_config),
        output_csv=Path(args.output_csv),
        output_parquet=Path(args.output_parquet),
        roi_mask_dir=Path(args.roi_mask_dir),
        schema_note_path=Path(args.schema_note),
    )
    print(f"Rows written: {len(index_df)}")
    print(f"CSV index: {Path(args.output_csv).resolve()}")
    print(f"Parquet index: {Path(args.output_parquet).resolve()}")
    print(f"Schema note: {Path(args.schema_note).resolve()}")
    print(f"ROI mask dir: {Path(args.roi_mask_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
