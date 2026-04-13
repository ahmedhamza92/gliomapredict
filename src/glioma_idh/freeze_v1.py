from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOCKED_MODALITIES = ("flair", "t1c", "t2")
ROI_STRATEGY = "whole_tumour_union_mask_gt_0"
PREFERRED_VARIANT = "bias"


@dataclass
class CohortFreezeSummary:
    starting_subjects: int
    idh_retained: int
    segmentation_retained: int
    modality_retained: int
    final_subjects: int


def _load_manifest(manifest_csv: Path) -> pd.DataFrame:
    return pd.read_csv(manifest_csv)


def _parse_json_list(value: str | float | None) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return list(json.loads(str(value)))


def _parse_json_dict(value: str | float | None) -> dict[str, list[str]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    return dict(json.loads(str(value)))


def _select_modality_paths(paths: list[str]) -> tuple[str | None, str | None, str | None, str | None]:
    raw_paths = sorted(path for path in paths if "_bias.nii.gz" not in Path(path).name)
    bias_paths = sorted(path for path in paths if "_bias.nii.gz" in Path(path).name)
    preferred_path = bias_paths[0] if bias_paths else (raw_paths[0] if raw_paths else None)
    preferred_variant = "bias" if bias_paths else ("raw" if raw_paths else None)
    return preferred_path, (raw_paths[0] if raw_paths else None), (bias_paths[0] if bias_paths else None), preferred_variant


def freeze_v1_cohort(manifest: pd.DataFrame) -> tuple[pd.DataFrame, CohortFreezeSummary]:
    starting_subjects = int(len(manifest))

    cohort = manifest.copy()
    cohort = cohort[cohort["idh_label"].notna()].copy()
    idh_retained = int(len(cohort))

    cohort = cohort[cohort["has_tumor_segmentation"].fillna(False)].copy()
    segmentation_retained = int(len(cohort))

    modality_mask = np.ones(len(cohort), dtype=bool)
    for modality in LOCKED_MODALITIES:
        modality_mask &= cohort[f"has_{modality}"].fillna(False).to_numpy()
    cohort = cohort[modality_mask].copy()
    modality_retained = int(len(cohort))

    modality_maps = cohort["modality_file_map_json"].apply(_parse_json_dict)
    tumor_masks = cohort["tumor_segmentation_paths_json"].apply(_parse_json_list)

    cohort["tumor_segmentation_path"] = tumor_masks.apply(lambda paths: sorted(paths)[0] if paths else None)
    cohort["roi_strategy"] = ROI_STRATEGY

    for modality in LOCKED_MODALITIES:
        selected = modality_maps.apply(lambda mapping: _select_modality_paths(mapping.get(modality, [])))
        cohort[f"{modality}_path"] = selected.apply(lambda item: item[0])
        cohort[f"{modality}_raw_path"] = selected.apply(lambda item: item[1])
        cohort[f"{modality}_bias_path"] = selected.apply(lambda item: item[2])
        cohort[f"{modality}_selected_variant"] = selected.apply(lambda item: item[3])

    keep_columns = [
        "subject_id",
        "canonical_subject_id",
        "primary_visit_id",
        "primary_visit_type",
        "n_visit_dirs",
        "followup_visit_ids_json",
        "idh_label",
        "idh_raw",
        "mgmt_label",
        "mgmt_raw",
        "grade_raw",
        "age",
        "sex",
        "tumor_segmentation_path",
        "roi_strategy",
        "metadata_subject_id",
        "metadata_source_files",
        "metadata_row_count",
        "metadata_row_ids_json",
    ]
    for modality in LOCKED_MODALITIES:
        keep_columns.extend(
            [
                f"{modality}_path",
                f"{modality}_raw_path",
                f"{modality}_bias_path",
                f"{modality}_selected_variant",
            ]
        )

    cohort = cohort[keep_columns].sort_values("subject_id").reset_index(drop=True)
    summary = CohortFreezeSummary(
        starting_subjects=starting_subjects,
        idh_retained=idh_retained,
        segmentation_retained=segmentation_retained,
        modality_retained=modality_retained,
        final_subjects=int(len(cohort)),
    )
    return cohort, summary


def _safe_percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def inspect_frozen_cohort(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import nibabel as nib

    modality_rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []

    for row in cohort.to_dict(orient="records"):
        mask_img = nib.load(str(row["tumor_segmentation_path"]))
        mask_arr = np.asarray(mask_img.dataobj)
        roi = mask_arr > 0
        coords = np.argwhere(roi)
        if coords.size == 0:
            bbox_min = bbox_max = bbox_size = (None, None, None)
        else:
            bbox_min = tuple(int(value) for value in coords.min(axis=0))
            bbox_max = tuple(int(value) for value in coords.max(axis=0))
            bbox_size = tuple(int(max_v - min_v + 1) for min_v, max_v in zip(bbox_min, bbox_max))

        unique_labels = tuple(int(value) for value in np.unique(mask_arr))
        mask_rows.append(
            {
                "subject_id": row["subject_id"],
                "mask_path": row["tumor_segmentation_path"],
                "mask_shape_x": int(mask_arr.shape[0]),
                "mask_shape_y": int(mask_arr.shape[1]),
                "mask_shape_z": int(mask_arr.shape[2]),
                "mask_spacing_x": float(mask_img.header.get_zooms()[0]),
                "mask_spacing_y": float(mask_img.header.get_zooms()[1]),
                "mask_spacing_z": float(mask_img.header.get_zooms()[2]),
                "mask_unique_labels": ";".join(str(value) for value in unique_labels),
                "roi_voxel_count": int(roi.sum()),
                "bbox_x": bbox_size[0],
                "bbox_y": bbox_size[1],
                "bbox_z": bbox_size[2],
                "bbox_min_x": bbox_min[0],
                "bbox_min_y": bbox_min[1],
                "bbox_min_z": bbox_min[2],
                "bbox_max_x": bbox_max[0],
                "bbox_max_y": bbox_max[1],
                "bbox_max_z": bbox_max[2],
            }
        )

        for modality in LOCKED_MODALITIES:
            path = row[f"{modality}_path"]
            img = nib.load(str(path))
            data = np.asarray(img.dataobj, dtype=np.float32)
            nonzero = data[data != 0]
            roi_values = data[roi]
            modality_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "modality": modality,
                    "selected_variant": row[f"{modality}_selected_variant"],
                    "path": path,
                    "shape_x": int(data.shape[0]),
                    "shape_y": int(data.shape[1]),
                    "shape_z": int(data.shape[2]),
                    "spacing_x": float(img.header.get_zooms()[0]),
                    "spacing_y": float(img.header.get_zooms()[1]),
                    "spacing_z": float(img.header.get_zooms()[2]),
                    "nonzero_voxel_count": int(nonzero.size),
                    "nonzero_min": float(nonzero.min()) if nonzero.size else None,
                    "nonzero_p01": _safe_percentile(nonzero, 1),
                    "nonzero_p50": _safe_percentile(nonzero, 50),
                    "nonzero_p99": _safe_percentile(nonzero, 99),
                    "nonzero_max": float(nonzero.max()) if nonzero.size else None,
                    "roi_voxel_count": int(roi_values.size),
                    "roi_min": float(roi_values.min()) if roi_values.size else None,
                    "roi_p01": _safe_percentile(roi_values, 1),
                    "roi_p50": _safe_percentile(roi_values, 50),
                    "roi_p99": _safe_percentile(roi_values, 99),
                    "roi_max": float(roi_values.max()) if roi_values.size else None,
                }
            )

    modality_qc = pd.DataFrame(modality_rows).sort_values(["modality", "subject_id"]).reset_index(drop=True)
    mask_qc = pd.DataFrame(mask_rows).sort_values("subject_id").reset_index(drop=True)
    return modality_qc, mask_qc


def _format_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


def _summarise_modality_qc(modality_qc: pd.DataFrame) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for modality, table in modality_qc.groupby("modality"):
        summary[modality] = {
            "shapes": sorted({(int(x), int(y), int(z)) for x, y, z in table[["shape_x", "shape_y", "shape_z"]].itertuples(index=False, name=None)}),
            "spacings": sorted(
                {
                    (
                        round(float(x), 5),
                        round(float(y), 5),
                        round(float(z), 5),
                    )
                    for x, y, z in table[["spacing_x", "spacing_y", "spacing_z"]].itertuples(index=False, name=None)
                }
            ),
            "nonzero_p01_min": float(table["nonzero_p01"].min()),
            "nonzero_p01_median": float(table["nonzero_p01"].median()),
            "nonzero_p50_median": float(table["nonzero_p50"].median()),
            "nonzero_p99_median": float(table["nonzero_p99"].median()),
            "nonzero_p99_max": float(table["nonzero_p99"].max()),
            "roi_p01_median": float(table["roi_p01"].median()),
            "roi_p50_median": float(table["roi_p50"].median()),
            "roi_p99_median": float(table["roi_p99"].median()),
        }
    return summary


def _summarise_mask_qc(mask_qc: pd.DataFrame) -> dict[str, object]:
    return {
        "unique_label_sets": sorted(mask_qc["mask_unique_labels"].dropna().unique().tolist()),
        "bbox_x_p95": float(mask_qc["bbox_x"].quantile(0.95)),
        "bbox_y_p95": float(mask_qc["bbox_y"].quantile(0.95)),
        "bbox_z_p95": float(mask_qc["bbox_z"].quantile(0.95)),
        "bbox_x_max": int(mask_qc["bbox_x"].max()),
        "bbox_y_max": int(mask_qc["bbox_y"].max()),
        "bbox_z_max": int(mask_qc["bbox_z"].max()),
        "roi_voxel_median": float(mask_qc["roi_voxel_count"].median()),
    }


def write_cohort_freeze_markdown(
    output_path: Path,
    summary: CohortFreezeSummary,
    cohort: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = f"""# Cohort Freeze v1

## Locked cohort definition

- primary pre-operative visits only
- non-null `IDH` label required
- tumour segmentation required
- modality subset locked to `FLAIR + T1c + T2`
- canonical structural file choice: prefer provided `_bias.nii.gz` image when available
- tumour ROI definition: `{ROI_STRATEGY}`

## Sequential counts

- Audited patient-level manifest: **{summary.starting_subjects}**
- After non-null IDH requirement: **{summary.idh_retained}**
- After tumour segmentation requirement: **{summary.segmentation_retained}**
- After locked modality requirement (`FLAIR`, `T1c`, `T2`): **{summary.modality_retained}**
- Final frozen cohort: **{summary.final_subjects}**

## Final label counts

- IDH wildtype: **{int((cohort["idh_label"] == "wildtype").sum())}**
- IDH mutant: **{int((cohort["idh_label"] == "mutant").sum())}**
- MGMT labelled within frozen cohort: **{int(cohort["mgmt_label"].notna().sum())}**

## Notes

- The manifest already resolves baseline versus follow-up duplication by preferring the non-follow-up visit as `primary_visit_id`.
- All frozen subjects retain tumour masks and both raw plus bias-corrected structural images for the locked modalities.
"""
    output_path.write_text(markdown, encoding="utf-8")


def write_preprocessing_design_markdown(
    output_path: Path,
    cohort: pd.DataFrame,
    modality_qc: pd.DataFrame,
    mask_qc: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    modality_summary = _summarise_modality_qc(modality_qc)
    mask_summary = _summarise_mask_qc(mask_qc)

    modality_lines = []
    for modality in LOCKED_MODALITIES:
        stats = modality_summary[modality]
        modality_lines.append(f"- `{modality}` shapes: {stats['shapes']}")
        modality_lines.append(f"- `{modality}` spacings: {stats['spacings']}")
        modality_lines.append(
            f"- `{modality}` nonzero intensity summary across cohort: "
            f"p01 median {_format_float(stats['nonzero_p01_median'])}, "
            f"p50 median {_format_float(stats['nonzero_p50_median'])}, "
            f"p99 median {_format_float(stats['nonzero_p99_median'])}, "
            f"max cohort p99 {_format_float(stats['nonzero_p99_max'])}"
        )
        modality_lines.append(
            f"- `{modality}` ROI intensity summary across cohort: "
            f"p01 median {_format_float(stats['roi_p01_median'])}, "
            f"p50 median {_format_float(stats['roi_p50_median'])}, "
            f"p99 median {_format_float(stats['roi_p99_median'])}"
        )

    markdown = f"""# Preprocessing Design v1

## Geometry decision

- Additional resampling is **not required** for v1.
- Rationale: every frozen `FLAIR`, `T1c`, and `T2` image was measured on the same grid and spacing, and the tumour masks are already aligned to that grid.
- Observed mask label sets across the frozen cohort: {", ".join(mask_summary["unique_label_sets"])}
- Whole-tumour ROI for v1: binary union of all non-zero tumour labels, implemented as `tumor_segmentation > 0`.

## Locked canonical inputs

- Use the provided bias-corrected structural images as canonical inputs for `FLAIR`, `T1c`, and `T2`.
- Keep the raw paths in the frozen cohort table for traceability, but do not make raw structural images the default modelling input in v1.

## Measured cohort properties

{chr(10).join(modality_lines)}
- Tumour mask bounding-box p95: x={_format_float(mask_summary["bbox_x_p95"])}, y={_format_float(mask_summary["bbox_y_p95"])}, z={_format_float(mask_summary["bbox_z_p95"])}
- Tumour mask bounding-box max: x={mask_summary["bbox_x_max"]}, y={mask_summary["bbox_y_max"]}, z={mask_summary["bbox_z_max"]}
- Median whole-tumour voxel count: {_format_float(mask_summary["roi_voxel_median"])}

## Minimum preprocessing: radiomics baseline

- Inputs: canonical bias-corrected `FLAIR`, `T1c`, `T2` plus the binary whole-tumour mask.
- Resampling: none.
- Intensity normalization: **per-image**, not ROI-based.
- Normalization rule: on each modality separately, use non-zero voxels as the skull-stripped brain region, clip intensities to the non-zero `[1, 99]` percentile range, then z-score using the non-zero voxel mean and standard deviation.
- ROI handling: extract shape features from the binary whole-tumour mask and intensity/texture features from the normalized images within that mask.
- Rationale for per-image normalization: MRI intensity scales still vary across subjects even after bias correction, while ROI-only normalization would discard potentially useful tumour-versus-background contrast structure.

## Minimum preprocessing: tumour-centred crop generation for later CNN use

- Inputs: the same canonical bias-corrected, per-image normalized `FLAIR`, `T1c`, and `T2` volumes plus the binary whole-tumour mask.
- Crop anchor: compute the tight whole-tumour bounding box from `mask > 0`.
- Padding rule: expand the bounding box by a fixed margin of `16` voxels in x/y and `8` voxels in z, then clip to image bounds.
- Store the padded crop coordinates now; defer final 2.5D slice sampling policy to the later CNN-specific step.
- Because the grid is already uniform, crop generation should be index-based only, with no additional interpolation.

## File-level outputs to create for downstream modelling

- `data/interim/cohort_v1.csv` and `data/interim/cohort_v1.parquet`: frozen cohort with canonical modality paths and label columns
- `data/interim/cohort_v1_modality_qc.csv` and `data/interim/cohort_v1_modality_qc.parquet`: per-subject modality geometry and intensity QC
- `data/interim/cohort_v1_mask_qc.csv` and `data/interim/cohort_v1_mask_qc.parquet`: per-subject mask labels, voxel counts, and tumour bounding boxes
- `data/processed/v1_preproc_index.parquet`: next-step index of canonical paths plus normalization parameters and padded crop coordinates

## Not doing yet

- no model fitting
- no augmentation policy
- no CNN-specific slice sampler
- no additional harmonisation beyond the measured uniform grid and per-image normalization
"""
    output_path.write_text(markdown, encoding="utf-8")


def write_dataframe_with_parquet(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze the audited v1 cohort and inspect preprocessing-relevant image properties.")
    parser.add_argument("--manifest-csv", default="data/interim/patient_manifest.csv")
    parser.add_argument("--cohort-csv", default="data/interim/cohort_v1.csv")
    parser.add_argument("--cohort-parquet", default="data/interim/cohort_v1.parquet")
    parser.add_argument("--modality-qc-csv", default="data/interim/cohort_v1_modality_qc.csv")
    parser.add_argument("--modality-qc-parquet", default="data/interim/cohort_v1_modality_qc.parquet")
    parser.add_argument("--mask-qc-csv", default="data/interim/cohort_v1_mask_qc.csv")
    parser.add_argument("--mask-qc-parquet", default="data/interim/cohort_v1_mask_qc.parquet")
    parser.add_argument("--cohort-md", default="reports/cohort_freeze_v1.md")
    parser.add_argument("--preproc-md", default="reports/preprocessing_design_v1.md")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    manifest = _load_manifest(Path(args.manifest_csv))
    cohort, summary = freeze_v1_cohort(manifest)
    modality_qc, mask_qc = inspect_frozen_cohort(cohort)

    write_dataframe_with_parquet(cohort, Path(args.cohort_csv), Path(args.cohort_parquet))
    write_dataframe_with_parquet(modality_qc, Path(args.modality_qc_csv), Path(args.modality_qc_parquet))
    write_dataframe_with_parquet(mask_qc, Path(args.mask_qc_csv), Path(args.mask_qc_parquet))
    write_cohort_freeze_markdown(Path(args.cohort_md), summary, cohort)
    write_preprocessing_design_markdown(Path(args.preproc_md), cohort, modality_qc, mask_qc)

    print(f"Frozen cohort CSV: {Path(args.cohort_csv).resolve()}")
    print(f"Frozen cohort Parquet: {Path(args.cohort_parquet).resolve()}")
    print(f"Modality QC CSV: {Path(args.modality_qc_csv).resolve()}")
    print(f"Mask QC CSV: {Path(args.mask_qc_csv).resolve()}")
    print(f"Cohort note: {Path(args.cohort_md).resolve()}")
    print(f"Preprocessing note: {Path(args.preproc_md).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
