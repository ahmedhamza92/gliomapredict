from __future__ import annotations

import argparse
from pathlib import Path

from .audit import summarise_manifest, write_audit_markdown, write_modality_figure
from .manifest import build_manifest, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a patient-level manifest and dataset audit for a local UCSF-PDGM download.")
    parser.add_argument("--dataset-root", required=True, help="Absolute or relative path to the downloaded dataset root.")
    parser.add_argument("--manifest-csv", default="data/interim/patient_manifest.csv", help="Output CSV manifest path.")
    parser.add_argument("--manifest-parquet", default="data/interim/patient_manifest.parquet", help="Output Parquet manifest path.")
    parser.add_argument("--audit-md", default="reports/dataset_audit.md", help="Output dataset audit markdown path.")
    parser.add_argument(
        "--figure-path",
        default="reports/figures/modality_completeness.png",
        help="Output modality completeness figure path.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        parser.error(f"Dataset root does not exist: {dataset_root}")

    artifacts = build_manifest(dataset_root)
    parquet_written, parquet_unresolved = write_manifest(
        manifest=artifacts.manifest,
        csv_path=Path(args.manifest_csv),
        parquet_path=Path(args.manifest_parquet),
    )
    unresolved = list(artifacts.unresolved_issues) + list(parquet_unresolved)

    summary = summarise_manifest(
        manifest=artifacts.manifest,
        metadata_tables=artifacts.metadata_tables,
        file_type_counts=artifacts.file_type_counts,
        visit_directory_count=artifacts.visit_directory_count,
        followup_directory_count=artifacts.followup_directory_count,
        unresolved_issues=unresolved,
        parquet_written=parquet_written,
    )
    figure_path = Path(args.figure_path)
    write_modality_figure(summary, figure_path)
    write_audit_markdown(summary, Path(args.audit_md), figure_path)

    print(f"Manifest CSV: {Path(args.manifest_csv).resolve()}")
    print(f"Manifest Parquet: {Path(args.manifest_parquet).resolve()} ({'written' if parquet_written else 'not written'})")
    print(f"Audit report: {Path(args.audit_md).resolve()}")
    print(f"Modality figure: {figure_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
