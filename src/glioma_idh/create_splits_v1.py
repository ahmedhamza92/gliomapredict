from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def create_splits(
    features_path: Path,
    config_path: Path,
    output_csv: Path,
    output_parquet: Path,
    note_path: Path,
) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold, train_test_split

    config = _load_yaml(config_path)
    features = pd.read_parquet(features_path) if features_path.suffix == ".parquet" else pd.read_csv(features_path)
    label_col = config["target_label"]
    subjects = features[["subject_id", label_col]].copy().sort_values("subject_id").reset_index(drop=True)

    train_subjects, test_subjects = train_test_split(
        subjects,
        test_size=float(config["test_size"]),
        random_state=int(config["random_state"]),
        stratify=subjects[label_col] if bool(config["stratify"]) else None,
    )

    train_subjects = train_subjects.sort_values("subject_id").reset_index(drop=True)
    test_subjects = test_subjects.sort_values("subject_id").reset_index(drop=True)

    train_subjects["split_set"] = "train"
    test_subjects["split_set"] = "test"
    train_subjects["cv_fold"] = -1
    test_subjects["cv_fold"] = -1

    skf = StratifiedKFold(
        n_splits=int(config["train_cv_folds"]),
        shuffle=True,
        random_state=int(config["random_state"]),
    )
    for fold_id, (_, val_idx) in enumerate(skf.split(train_subjects["subject_id"], train_subjects[label_col])):
        train_subjects.loc[val_idx, "cv_fold"] = int(fold_id)

    splits = pd.concat([train_subjects, test_subjects], ignore_index=True).sort_values("subject_id").reset_index(drop=True)
    splits["is_test"] = splits["split_set"].eq("test")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(output_csv, index=False)
    splits.to_parquet(output_parquet, index=False)

    train_counts = train_subjects[label_col].value_counts().sort_index()
    test_counts = test_subjects[label_col].value_counts().sort_index()
    note_lines = [
        "# Split Design v1",
        "",
        f"- Source feature table: `{features_path}`",
        f"- Target label: `{label_col}`",
        f"- Test size: **{config['test_size']}**",
        f"- Random state: **{config['random_state']}**",
        f"- Training CV folds: **{config['train_cv_folds']}**",
        "",
        "## Counts",
        "",
        f"- Train subjects: **{len(train_subjects)}**",
        f"- Test subjects: **{len(test_subjects)}**",
        "",
        "### Train label counts",
        "",
    ]
    for label, count in train_counts.items():
        note_lines.append(f"- `{label}`: **{int(count)}**")
    note_lines.extend(["", "### Test label counts", ""])
    for label, count in test_counts.items():
        note_lines.append(f"- `{label}`: **{int(count)}**")
    note_lines.extend(
        [
            "",
            "## Leakage control",
            "",
            "- Splitting is strictly patient-level because the feature table has one row per subject.",
            "- The test set is held out before any model selection.",
            "- The `cv_fold` assignments apply only inside the training pool for hyperparameter selection and internal validation.",
        ]
    )
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    return splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the patient-level IDH split artifact for v1 baseline modelling.")
    parser.add_argument("--features-path", default="data/processed/radiomics_features_v1.parquet")
    parser.add_argument("--config", default="configs/splits_v1.yaml")
    parser.add_argument("--output-csv", default="data/interim/splits_v1.csv")
    parser.add_argument("--output-parquet", default="data/interim/splits_v1.parquet")
    parser.add_argument("--note-path", default="reports/split_design_v1.md")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    splits = create_splits(
        features_path=Path(args.features_path),
        config_path=Path(args.config),
        output_csv=Path(args.output_csv),
        output_parquet=Path(args.output_parquet),
        note_path=Path(args.note_path),
    )
    print(f"Rows written: {len(splits)}")
    print(f"Train subjects: {int((splits['split_set'] == 'train').sum())}")
    print(f"Test subjects: {int((splits['split_set'] == 'test').sum())}")
    print(f"CSV: {Path(args.output_csv).resolve()}")
    print(f"Parquet: {Path(args.output_parquet).resolve()}")
    print(f"Note: {Path(args.note_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
