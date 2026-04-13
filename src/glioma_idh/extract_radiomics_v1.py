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


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        return data[key]


def _safe_entropy(values: np.ndarray, bins: int) -> float:
    if values.size == 0:
        return np.nan
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(min_value, max_value), density=False)
    probs = hist[hist > 0].astype(np.float64)
    probs /= probs.sum()
    return float(-(probs * np.log2(probs)).sum())


def _safe_skew(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    mean_value = float(values.mean())
    std_value = float(values.std())
    if std_value == 0.0:
        return 0.0
    centered = (values - mean_value) / std_value
    return float(np.mean(centered ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    mean_value = float(values.mean())
    std_value = float(values.std())
    if std_value == 0.0:
        return 0.0
    centered = (values - mean_value) / std_value
    return float(np.mean(centered ** 4) - 3.0)


def _compute_shape_features(mask: np.ndarray) -> dict[str, float]:
    from scipy import ndimage

    roi = mask > 0
    coords = np.argwhere(roi)
    if coords.size == 0:
        raise ValueError("Empty ROI mask encountered during radiomics extraction.")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bbox_sizes = (maxs - mins + 1).astype(int)
    volume = int(roi.sum())
    bbox_volume = int(np.prod(bbox_sizes))
    extent = float(volume / bbox_volume)

    eroded = ndimage.binary_erosion(roi, iterations=1, border_value=0)
    boundary = roi & ~eroded
    surface_voxels = int(boundary.sum())
    surface_to_volume = float(surface_voxels / volume)

    centered = coords.astype(np.float64) - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T) if coords.shape[0] > 1 else np.eye(3)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    major = float(np.sqrt(eigvals[0])) if eigvals[0] > 0 else 0.0
    minor = float(np.sqrt(eigvals[1])) if eigvals[1] > 0 else 0.0
    least = float(np.sqrt(eigvals[2])) if eigvals[2] > 0 else 0.0
    elongation = float(minor / major) if major > 0 else 0.0
    flatness = float(least / major) if major > 0 else 0.0

    sphericity = 0.0
    if surface_voxels > 0:
        sphericity = float((np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / surface_voxels)

    return {
        "shape_volume_voxels": float(volume),
        "shape_bbox_x": float(bbox_sizes[0]),
        "shape_bbox_y": float(bbox_sizes[1]),
        "shape_bbox_z": float(bbox_sizes[2]),
        "shape_bbox_volume": float(bbox_volume),
        "shape_extent": extent,
        "shape_surface_voxels": float(surface_voxels),
        "shape_surface_to_volume": surface_to_volume,
        "shape_major_axis": major,
        "shape_minor_axis": minor,
        "shape_least_axis": least,
        "shape_elongation": elongation,
        "shape_flatness": flatness,
        "shape_sphericity_approx": sphericity,
    }


def _compute_modality_features(image: np.ndarray, mask: np.ndarray, bins: int) -> dict[str, float]:
    from scipy import ndimage

    roi = mask > 0
    values = image[roi].astype(np.float32)
    if values.size == 0:
        raise ValueError("Empty ROI values encountered during radiomics extraction.")

    gradients = [ndimage.sobel(image, axis=axis, mode="nearest") for axis in range(3)]
    grad_mag = np.sqrt(sum(component ** 2 for component in gradients))[roi]
    laplace = ndimage.laplace(image, mode="nearest")[roi]

    p10, p25, p75, p90 = np.percentile(values, [10, 25, 75, 90])
    mean_value = float(values.mean())
    std_value = float(values.std())
    energy = float(np.mean(values ** 2))

    return {
        "voxel_count": float(values.size),
        "mean": mean_value,
        "std": std_value,
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "iqr": float(p75 - p25),
        "robust_range": float(p90 - p10),
        "mean_abs_dev": float(np.mean(np.abs(values - mean_value))),
        "skewness": _safe_skew(values),
        "kurtosis": _safe_kurtosis(values),
        "entropy": _safe_entropy(values, bins=bins),
        "energy": energy,
        "rms": float(np.sqrt(energy)),
        "grad_mean": float(grad_mag.mean()),
        "grad_std": float(grad_mag.std()),
        "laplace_mean": float(laplace.mean()),
        "laplace_std": float(laplace.std()),
    }


def _feature_qc_table(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for feature_name in feature_columns:
        series = features[feature_name]
        finite_series = series[np.isfinite(series)] if series.dtype.kind in {"f", "i"} else series
        variance = float(series.var()) if series.dtype.kind in {"f", "i"} and series.notna().any() else np.nan
        rows.append(
            {
                "feature_name": feature_name,
                "missing_count": int(series.isna().sum()),
                "nonfinite_count": int((~np.isfinite(series)).sum()) if series.dtype.kind in {"f", "i"} else 0,
                "n_unique_nonnull": int(series.dropna().nunique()),
                "variance": variance,
                "is_constant_nonnull": bool(series.dropna().nunique() <= 1),
            }
        )
    return pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)


def _write_note(
    output_path: Path,
    features: pd.DataFrame,
    feature_qc: pd.DataFrame,
    settings: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    id_columns = ["subject_id", "idh_label", "mgmt_label"]
    feature_columns = [column for column in features.columns if column not in id_columns]
    lines = [
        "# Radiomics Features v1",
        "",
        f"- Rows: **{len(features)}**",
        f"- Identifier/label columns: `{', '.join(id_columns)}`",
        f"- Feature columns: **{len(feature_columns)}**",
        f"- Histogram bins for entropy: **{settings['histogram_bins']}**",
        "",
        "## Feature families",
        "",
        "- Shape features from the binary whole-tumour ROI only",
        "- First-order intensity features for normalized `FLAIR`, `T1c`, and `T2` ROI voxels",
        "- Simple filter-response heterogeneity features via Sobel gradient magnitude and Laplacian response inside the ROI",
        "",
        "## QC summary",
        "",
        f"- Features with any missing values: **{int((feature_qc['missing_count'] > 0).sum())}**",
        f"- Features with any non-finite values: **{int((feature_qc['nonfinite_count'] > 0).sum())}**",
        f"- Features constant across the full cohort: **{int(feature_qc['is_constant_nonnull'].sum())}**",
        "",
        "## Modelling note",
        "",
        "- This table is the raw radiomics feature table for the frozen cohort.",
        "- Any feature filtering used for baseline modelling must be fit on the training pool only, not on the full dataset.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_radiomics(
    input_index_path: Path,
    config_path: Path,
    output_csv: Path,
    output_parquet: Path,
    feature_qc_csv: Path,
    note_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _load_yaml(config_path)
    inputs = pd.read_parquet(input_index_path) if input_index_path.suffix == ".parquet" else pd.read_csv(input_index_path)
    inputs = inputs.sort_values("subject_id").reset_index(drop=True)
    bins = int(config["settings"]["histogram_bins"])

    rows: list[dict[str, Any]] = []
    for row in inputs.to_dict(orient="records"):
        mask = _load_npz_array(Path(row["roi_mask_crop_path"]), "mask").astype(bool)
        feature_row: dict[str, Any] = {
            "subject_id": row["subject_id"],
            "idh_label": row["idh_label"],
            "mgmt_label": row["mgmt_label"],
        }
        feature_row.update(_compute_shape_features(mask))
        for modality in LOCKED_MODALITIES:
            image = _load_npz_array(Path(row[f"{modality}_norm_crop_path"]), "image").astype(np.float32)
            modality_features = _compute_modality_features(image=image, mask=mask, bins=bins)
            for feature_name, feature_value in modality_features.items():
                feature_row[f"{modality}_{feature_name}"] = feature_value
        rows.append(feature_row)

    features = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    feature_columns = [column for column in features.columns if column not in {"subject_id", "idh_label", "mgmt_label"}]
    feature_qc = _feature_qc_table(features, feature_columns)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    feature_qc_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)
    features.to_parquet(output_parquet, index=False)
    feature_qc.to_csv(feature_qc_csv, index=False)
    _write_note(note_path, features, feature_qc, config["settings"])
    return features, feature_qc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract bounded radiomics features for the frozen v1 cohort.")
    parser.add_argument("--input-index", default="data/processed/radiomics_inputs_v1_index.parquet")
    parser.add_argument("--config", default="configs/radiomics_v1.yaml")
    parser.add_argument("--output-csv", default="data/processed/radiomics_features_v1.csv")
    parser.add_argument("--output-parquet", default="data/processed/radiomics_features_v1.parquet")
    parser.add_argument("--feature-qc-csv", default="data/processed/radiomics_feature_qc_v1.csv")
    parser.add_argument("--note-path", default="reports/radiomics_features_v1.md")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    features, feature_qc = extract_radiomics(
        input_index_path=Path(args.input_index),
        config_path=Path(args.config),
        output_csv=Path(args.output_csv),
        output_parquet=Path(args.output_parquet),
        feature_qc_csv=Path(args.feature_qc_csv),
        note_path=Path(args.note_path),
    )
    print(f"Rows written: {len(features)}")
    print(f"Feature columns: {len([c for c in features.columns if c not in {'subject_id', 'idh_label', 'mgmt_label'}])}")
    print(f"Feature QC rows: {len(feature_qc)}")
    print(f"Features CSV: {Path(args.output_csv).resolve()}")
    print(f"Features Parquet: {Path(args.output_parquet).resolve()}")
    print(f"Feature QC CSV: {Path(args.feature_qc_csv).resolve()}")
    print(f"Note: {Path(args.note_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
