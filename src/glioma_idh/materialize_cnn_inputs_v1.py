from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        return data[key]


def select_center_slice(mask_stack: np.ndarray) -> int:
    slice_areas = mask_stack.sum(axis=(0, 1))
    candidates = np.flatnonzero(slice_areas == slice_areas.max())
    midpoint = (mask_stack.shape[2] - 1) / 2.0
    ranked = sorted(((abs(float(idx) - midpoint), int(idx)) for idx in candidates))
    return int(ranked[0][1])


def build_slice_indices(center_slice: int, depth: int, offsets: list[int]) -> list[int]:
    return [int(min(max(center_slice + int(offset), 0), depth - 1)) for offset in offsets]


def resize_slice(array_2d: np.ndarray, height: int, width: int, resample: int) -> np.ndarray:
    pil_img = Image.fromarray(array_2d.astype(np.float32), mode="F")
    resized = pil_img.resize((width, height), resample=resample)
    return np.asarray(resized, dtype=np.float32)


def tensorize_subject(
    row: dict[str, Any],
    modalities: list[str],
    slice_offsets: list[int],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], int]:
    mask = _load_npz_array(Path(row["roi_mask_crop_path"]), "mask").astype(np.uint8)
    center_slice = select_center_slice(mask)
    slice_indices = build_slice_indices(center_slice=center_slice, depth=mask.shape[2], offsets=slice_offsets)
    images = {
        modality: _load_npz_array(Path(row[f"{modality}_norm_crop_path"]), "image").astype(np.float32)
        for modality in modalities
    }

    channels: list[np.ndarray] = []
    resized_mask_stack = []
    display_t1c = None
    for slice_position, slice_index in enumerate(slice_indices):
        resized_mask = resize_slice(mask[:, :, slice_index], height=height, width=width, resample=Image.Resampling.NEAREST)
        resized_mask_stack.append((resized_mask > 0).astype(np.uint8))
        for modality in modalities:
            resized = resize_slice(images[modality][:, :, slice_index], height=height, width=width, resample=Image.Resampling.BILINEAR)
            channels.append(resized)
            if modality == "t1c" and slice_position == len(slice_indices) // 2:
                display_t1c = resized
    if display_t1c is None:
        raise ValueError(f"Failed to extract central T1c slice for {row['subject_id']}")
    image_tensor = np.stack(channels, axis=0).astype(np.float32)
    mask_tensor = np.stack(resized_mask_stack, axis=0).astype(np.uint8)
    center_mask = mask_tensor[len(slice_indices) // 2]
    return image_tensor, mask_tensor, display_t1c.astype(np.float32), slice_indices, center_slice


def _role_from_split(split_set: str, cv_fold: int, validation_fold: int) -> str:
    if split_set == "test":
        return "test"
    if int(cv_fold) == int(validation_fold):
        return "validation"
    return "train_fit"


def _write_note(
    output_path: Path,
    index_df: pd.DataFrame,
    config_path: Path,
    representation: dict[str, Any],
    validation_fold: int,
) -> None:
    lines = [
        "# CNN Inputs v1",
        "",
        f"- Source config: `{config_path}`",
        f"- Rows: **{len(index_df)}**",
        f"- Deterministic representation: axial 2.5D slice stack with offsets `{representation['slice_offsets']}` around the ROI-max-area slice.",
        f"- Modalities: `{', '.join(representation['modalities'])}`",
        f"- Channel order: `{representation['channel_order']}`",
        f"- Tensor shape: **({len(representation['slice_offsets']) * len(representation['modalities'])}, {representation['resize_height']}, {representation['resize_width']})**",
        f"- Validation fold inside frozen training pool: **{validation_fold}**",
        "",
        "## Schema",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        "| `subject_id` | Frozen cohort subject identifier |",
        "| `idh_label` | Frozen v1 target label |",
        "| `split_set` / `cv_fold` | Frozen split artifact fields |",
        "| `cnn_role` | Deterministic role: `train_fit`, `validation`, or `test` |",
        "| `tensor_path` | Authoritative `.npz` tensor artifact for the subject |",
        "| `tensor_channels`, `tensor_height`, `tensor_width` | Final tensor shape metadata |",
        "| `center_slice_index` | Selected axial slice with maximum ROI area |",
        "| `selected_slice_indices_json` | Exact sampled axial slice indices after boundary clamping |",
        "| `source_crop_shape_*` | Original padded crop shape before 2D resizing |",
        "",
        "## Tensor contents",
        "",
        "- `image`: float32 tensor with shape `(channels, height, width)`",
        "- `mask_stack`: uint8 resized ROI slice stack aligned to the sampled slices",
        "- `center_mask`: uint8 ROI mask for the central sampled slice",
        "- `display_t1c`: float32 central T1c slice used for qualitative overlay visualizations",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_cnn_inputs(config_path: Path) -> pd.DataFrame:
    config = _load_yaml(config_path)
    source_index = pd.read_csv(Path(config["sources"]["radiomics_inputs_index_csv"])).sort_values("subject_id").reset_index(drop=True)
    splits = pd.read_csv(Path(config["sources"]["splits_csv"])).sort_values("subject_id").reset_index(drop=True)
    merged = source_index.merge(
        splits[["subject_id", "idh_label", "split_set", "cv_fold"]],
        on=["subject_id", "idh_label"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(source_index):
        raise ValueError("CNN input merge did not preserve one row per frozen subject.")

    representation = config["representation"]
    output_dir = Path(representation["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_fold = int(config["training"]["validation_fold"])
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        tensor, mask_stack, display_t1c, slice_indices, center_slice = tensorize_subject(
            row=row,
            modalities=list(representation["modalities"]),
            slice_offsets=list(representation["slice_offsets"]),
            height=int(representation["resize_height"]),
            width=int(representation["resize_width"]),
        )
        tensor_path = output_dir / f"{row['subject_id']}.npz"
        np.savez_compressed(
            tensor_path,
            image=tensor,
            mask_stack=mask_stack,
            center_mask=mask_stack[len(slice_indices) // 2],
            display_t1c=display_t1c,
            selected_slice_indices=np.asarray(slice_indices, dtype=np.int16),
        )
        rows.append(
            {
                "subject_id": row["subject_id"],
                "idh_label": row["idh_label"],
                "split_set": row["split_set"],
                "cv_fold": int(row["cv_fold"]),
                "cnn_role": _role_from_split(row["split_set"], int(row["cv_fold"]), validation_fold),
                "tensor_path": str(tensor_path.resolve()),
                "tensor_channels": int(tensor.shape[0]),
                "tensor_height": int(tensor.shape[1]),
                "tensor_width": int(tensor.shape[2]),
                "center_slice_index": int(center_slice),
                "selected_slice_indices_json": json.dumps(slice_indices),
                "slice_offsets_json": json.dumps(list(representation["slice_offsets"])),
                "source_crop_shape_x": int(row["crop_shape_x"]),
                "source_crop_shape_y": int(row["crop_shape_y"]),
                "source_crop_shape_z": int(row["crop_shape_z"]),
                "roi_mask_crop_path": row["roi_mask_crop_path"],
            }
        )

    output_index = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    output_csv = Path(representation["output_index_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_index.to_csv(output_csv, index=False)
    _write_note(
        output_path=Path(representation["note_path"]),
        index_df=output_index,
        config_path=config_path,
        representation=representation,
        validation_fold=validation_fold,
    )
    return output_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize deterministic CNN-ready 2.5D tensors from the frozen v1 artifacts.")
    parser.add_argument("--config", default="configs/cnn_v1_1.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    index_df = materialize_cnn_inputs(Path(args.config))
    print(f"Rows written: {len(index_df)}")
    print(f"CNN tensor index: {Path(_load_yaml(Path(args.config))['representation']['output_index_csv']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
