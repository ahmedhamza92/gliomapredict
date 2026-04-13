from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOCKED_MODALITIES = ("flair", "t1c", "t2")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_nifti(path: Path):
    import nibabel as nib

    return nib.load(str(path))


def normalize_volume(volume: np.ndarray, clip_low: float, clip_high: float, mean_value: float, std_value: float) -> np.ndarray:
    normalized = np.zeros(volume.shape, dtype=np.float32)
    nonzero_mask = volume != 0
    if not np.any(nonzero_mask):
        return normalized
    clipped = np.clip(volume[nonzero_mask].astype(np.float32), clip_low, clip_high)
    normalized[nonzero_mask] = (clipped - mean_value) / std_value
    return normalized


def crop_array(
    array: np.ndarray,
    start_x: int,
    start_y: int,
    start_z: int,
    end_x: int,
    end_y: int,
    end_z: int,
) -> np.ndarray:
    return array[start_x:end_x, start_y:end_y, start_z:end_z]


def _write_npz(path: Path, key: str, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: array})
    return str(path.resolve())


def _schema_rows() -> list[tuple[str, str, str]]:
    return [
        ("subject_id", "string", "Frozen cohort subject identifier."),
        ("idh_label / mgmt_label", "string", "Label columns copied from the frozen cohort."),
        ("<modality>_norm_crop_path", "string", "Compressed NumPy archive for the normalized padded crop."),
        ("roi_mask_crop_path", "string", "Compressed NumPy archive for the cropped binary ROI mask."),
        ("crop_shape_x/y/z", "int", "Stored crop array shape."),
        ("crop_start_* / crop_end_*_exclusive", "int", "Padded crop bounds copied from the preprocessing index."),
        ("<modality>_norm_*", "float/int", "Per-subject normalization parameters copied from the preprocessing index."),
        ("qc_*", "bool", "Copied preprocessing QC flags for downstream gating."),
    ]


def _write_schema_note(output_path: Path, index_df: pd.DataFrame, output_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Radiomics Inputs v1",
        "",
        f"- Rows: **{len(index_df)}**",
        f"- Output directory: `{output_dir}`",
        "",
        "## Exact schema",
        "",
        "| Column group | Type | Meaning |",
        "| --- | --- | --- |",
    ]
    for group, dtype_name, meaning in _schema_rows():
        lines.append(f"| `{group}` | `{dtype_name}` | {meaning} |")
    lines.extend(
        [
            "",
            "## Downstream use",
            "",
            "- Feature extraction should load only the normalized crop files and the cropped ROI mask from this index.",
            "- These crops are already normalized using the per-subject parameters frozen in `v1_preproc_index`.",
            "- Do not renormalize from raw images inside the radiomics extraction or baseline modelling step.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_radiomics_inputs(
    preproc_index_csv: Path,
    preprocessing_config_path: Path,
    output_dir: Path,
    output_csv: Path,
    output_parquet: Path,
    note_path: Path,
) -> pd.DataFrame:
    config = _load_yaml(preprocessing_config_path)
    index_df = pd.read_csv(preproc_index_csv).sort_values("subject_id").reset_index(drop=True)

    expected_modalities = tuple(config["canonical_inputs"]["modalities"])
    if expected_modalities != LOCKED_MODALITIES:
        raise ValueError(f"Unexpected locked modalities in preprocessing config: {expected_modalities}")

    rows: list[dict[str, Any]] = []
    for row in index_df.to_dict(orient="records"):
        subject_dir = output_dir / row["subject_id"]

        mask_img = _load_nifti(Path(row["binary_roi_mask_path"]))
        mask_arr = np.asarray(mask_img.dataobj, dtype=np.uint8)
        cropped_mask = crop_array(
            mask_arr,
            int(row["crop_start_x"]),
            int(row["crop_start_y"]),
            int(row["crop_start_z"]),
            int(row["crop_end_x_exclusive"]),
            int(row["crop_end_y_exclusive"]),
            int(row["crop_end_z_exclusive"]),
        )
        roi_mask_crop_path = _write_npz(subject_dir / "roi_mask_crop.npz", "mask", cropped_mask.astype(np.uint8))

        out_row: dict[str, Any] = {
            "subject_id": row["subject_id"],
            "idh_label": row["idh_label"],
            "mgmt_label": row["mgmt_label"],
            "roi_mask_crop_path": roi_mask_crop_path,
            "crop_shape_x": int(cropped_mask.shape[0]),
            "crop_shape_y": int(cropped_mask.shape[1]),
            "crop_shape_z": int(cropped_mask.shape[2]),
            "crop_start_x": int(row["crop_start_x"]),
            "crop_start_y": int(row["crop_start_y"]),
            "crop_start_z": int(row["crop_start_z"]),
            "crop_end_x_exclusive": int(row["crop_end_x_exclusive"]),
            "crop_end_y_exclusive": int(row["crop_end_y_exclusive"]),
            "crop_end_z_exclusive": int(row["crop_end_z_exclusive"]),
            "roi_voxel_count": int(row["roi_voxel_count"]),
            "qc_all_pass": bool(row["qc_all_pass"]),
            "qc_padded_crop_clipped": bool(row["qc_padded_crop_clipped"]),
            "qc_has_followup_visits": bool(row["qc_has_followup_visits"]),
            "qc_multiple_metadata_rows": bool(row["qc_multiple_metadata_rows"]),
            "source_preproc_index_csv": str(preproc_index_csv.resolve()),
            "source_preprocessing_config": str(preprocessing_config_path.resolve()),
        }

        for modality in LOCKED_MODALITIES:
            image_img = _load_nifti(Path(row[f"{modality}_path"]))
            image_arr = np.asarray(image_img.dataobj, dtype=np.float32)
            normalized = normalize_volume(
                volume=image_arr,
                clip_low=float(row[f"{modality}_norm_clip_low_value"]),
                clip_high=float(row[f"{modality}_norm_clip_high_value"]),
                mean_value=float(row[f"{modality}_norm_mean_after_clip"]),
                std_value=float(row[f"{modality}_norm_std_after_clip"]),
            )
            cropped_norm = crop_array(
                normalized,
                int(row["crop_start_x"]),
                int(row["crop_start_y"]),
                int(row["crop_start_z"]),
                int(row["crop_end_x_exclusive"]),
                int(row["crop_end_y_exclusive"]),
                int(row["crop_end_z_exclusive"]),
            )
            norm_crop_path = _write_npz(subject_dir / f"{modality}_norm_crop.npz", "image", cropped_norm.astype(np.float32))

            out_row[f"{modality}_norm_crop_path"] = norm_crop_path
            out_row[f"{modality}_selected_variant"] = row[f"{modality}_selected_variant"]
            out_row[f"{modality}_norm_nonzero_voxel_count"] = int(row[f"{modality}_norm_nonzero_voxel_count"])
            out_row[f"{modality}_norm_clip_low_value"] = float(row[f"{modality}_norm_clip_low_value"])
            out_row[f"{modality}_norm_clip_high_value"] = float(row[f"{modality}_norm_clip_high_value"])
            out_row[f"{modality}_norm_mean_after_clip"] = float(row[f"{modality}_norm_mean_after_clip"])
            out_row[f"{modality}_norm_std_after_clip"] = float(row[f"{modality}_norm_std_after_clip"])
            out_row[f"{modality}_qc_norm_valid"] = bool(row[f"{modality}_qc_norm_valid"])
            out_row[f"{modality}_qc_nonfinite_crop"] = bool(not np.isfinite(cropped_norm).all())

        rows.append(out_row)

    radiomics_index = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    radiomics_index.to_csv(output_csv, index=False)
    radiomics_index.to_parquet(output_parquet, index=False)
    _write_schema_note(note_path, radiomics_index, output_dir)
    return radiomics_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize normalized radiomics-ready crops from the preprocessing index.")
    parser.add_argument("--preproc-index-csv", default="data/processed/v1_preproc_index.csv")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing_v1.yaml")
    parser.add_argument("--output-dir", default="data/processed/radiomics_inputs_v1")
    parser.add_argument("--output-csv", default="data/processed/radiomics_inputs_v1_index.csv")
    parser.add_argument("--output-parquet", default="data/processed/radiomics_inputs_v1_index.parquet")
    parser.add_argument("--note-path", default="reports/radiomics_inputs_v1.md")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    radiomics_index = materialize_radiomics_inputs(
        preproc_index_csv=Path(args.preproc_index_csv),
        preprocessing_config_path=Path(args.preprocessing_config),
        output_dir=Path(args.output_dir),
        output_csv=Path(args.output_csv),
        output_parquet=Path(args.output_parquet),
        note_path=Path(args.note_path),
    )
    print(f"Rows written: {len(radiomics_index)}")
    print(f"Radiomics-ready index CSV: {Path(args.output_csv).resolve()}")
    print(f"Radiomics-ready index Parquet: {Path(args.output_parquet).resolve()}")
    print(f"Normalized crops dir: {Path(args.output_dir).resolve()}")
    print(f"Note: {Path(args.note_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
