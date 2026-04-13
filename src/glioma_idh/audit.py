from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .manifest import MetadataTableSummary


@dataclass
class AuditSummary:
    subject_count: int
    visit_directory_count: int
    followup_directory_count: int
    idh_label_count: int
    mgmt_label_count: int
    tumor_segmentation_subject_count: int
    modality_counts: list[tuple[str, int, float]]
    unresolved_issues: list[str]
    metadata_tables: list[MetadataTableSummary]
    parquet_written: bool
    file_type_counts: dict[str, int]
    file_storage_type: str


def _classify_file_storage(file_type_counts: dict[str, int]) -> str:
    has_nifti = any(ext in file_type_counts for ext in (".nii", ".nii.gz"))
    has_dicom = any(ext in file_type_counts for ext in (".dcm", ".ima"))
    if has_nifti and has_dicom:
        return "mixed"
    if has_nifti:
        return "nifti"
    if has_dicom:
        return "dicom"
    return "unknown"


def summarise_manifest(
    manifest: pd.DataFrame,
    metadata_tables: list[MetadataTableSummary],
    file_type_counts: dict[str, int],
    visit_directory_count: int,
    followup_directory_count: int,
    unresolved_issues: list[str],
    parquet_written: bool,
) -> AuditSummary:
    subject_count = int(len(manifest))
    idh_label_count = int(manifest["idh_label"].notna().sum()) if "idh_label" in manifest else 0
    mgmt_label_count = int(manifest["mgmt_label"].notna().sum()) if "mgmt_label" in manifest else 0
    tumor_segmentation_subject_count = (
        int(manifest["has_tumor_segmentation"].fillna(False).sum()) if "has_tumor_segmentation" in manifest else 0
    )

    modality_counts: list[tuple[str, int, float]] = []
    for column in sorted(col for col in manifest.columns if col.startswith("has_") and col != "has_tumor_segmentation"):
        modality = column.removeprefix("has_")
        count = int(manifest[column].fillna(False).sum())
        proportion = (count / subject_count) if subject_count else 0.0
        modality_counts.append((modality, count, proportion))

    return AuditSummary(
        subject_count=subject_count,
        visit_directory_count=visit_directory_count,
        followup_directory_count=followup_directory_count,
        idh_label_count=idh_label_count,
        mgmt_label_count=mgmt_label_count,
        tumor_segmentation_subject_count=tumor_segmentation_subject_count,
        modality_counts=modality_counts,
        unresolved_issues=unresolved_issues,
        metadata_tables=metadata_tables,
        parquet_written=parquet_written,
        file_type_counts=file_type_counts,
        file_storage_type=_classify_file_storage(file_type_counts),
    )


def write_modality_figure(summary: AuditSummary, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    top_margin = 80
    left_margin = 220
    right_margin = 80
    row_height = 58
    plot_height = max(220, row_height * max(1, len(summary.modality_counts)))
    height = top_margin + plot_height + 80
    image = Image.new("RGB", (width, height), color=(248, 248, 245))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

    draw.text((40, 24), "Modality completeness across audited patients", fill=(20, 20, 20), font=title_font)
    draw.text(
        (40, 46),
        f"Patients: {summary.subject_count} | visit directories scanned: {summary.visit_directory_count}",
        fill=(80, 80, 80),
        font=body_font,
    )

    bar_left = left_margin
    bar_right = width - right_margin
    axis_y0 = top_margin
    axis_y1 = top_margin + plot_height
    draw.line((bar_left, axis_y0, bar_left, axis_y1), fill=(70, 70, 70), width=1)

    for index, (modality, count, proportion) in enumerate(summary.modality_counts):
        y = top_margin + index * row_height + 18
        bar_width = int((bar_right - bar_left) * proportion)
        draw.text((40, y), modality.upper(), fill=(20, 20, 20), font=body_font)
        draw.rectangle((bar_left, y, bar_left + bar_width, y + 24), fill=(44, 113, 164))
        label = f"{count}/{summary.subject_count} ({proportion:.1%})"
        draw.text((bar_left + bar_width + 12, y + 6), label, fill=(20, 20, 20), font=body_font)

    image.save(figure_path)


def _markdown_modality_table(summary: AuditSummary) -> str:
    lines = [
        "| Modality | Patients | Completeness |",
        "| --- | ---: | ---: |",
    ]
    for modality, count, proportion in summary.modality_counts:
        lines.append(f"| `{modality}` | {count} | {proportion:.1%} |")
    return "\n".join(lines)


def _markdown_metadata_table(summary: AuditSummary) -> str:
    if not summary.metadata_tables:
        return "No metadata or label table was discovered under the dataset root."

    lines = []
    for table in summary.metadata_tables:
        label_columns = ", ".join(f"`{logical}` -> `{column}`" for logical, column in sorted(table.label_columns.items())) or "none"
        columns = ", ".join(f"`{column}`" for column in table.columns)
        lines.append(f"- `{table.path}`")
        lines.append(f"  rows: {table.rows}")
        lines.append(f"  identifier column: `{table.identifier_column}`" if table.identifier_column else "  identifier column: none")
        lines.append(f"  detected label columns: {label_columns}")
        lines.append(f"  columns: {columns}")
    return "\n".join(lines)


def _markdown_file_types(summary: AuditSummary) -> str:
    lines = [f"- Storage classification: **{summary.file_storage_type.upper()}**"]
    for ext, count in summary.file_type_counts.items():
        lines.append(f"- `{ext}`: {count}")
    return "\n".join(lines)


def recommend_v1_subset(summary: AuditSummary) -> tuple[str, str]:
    eligible = [(modality, count, proportion) for modality, count, proportion in summary.modality_counts if proportion > 0]
    priority_order = {"flair": 0, "t1c": 1, "t2": 2, "t1": 3}
    eligible.sort(key=lambda item: (-item[2], priority_order.get(item[0], 99), item[0]))

    if not eligible:
        return (
            "No modality recommendation can be made yet because no MRI modalities were discovered.",
            "No cohort definition can be recommended yet because the dataset discovery returned zero audited subjects.",
        )

    preferred = [item for item in eligible if item[0] in priority_order]
    preferred.sort(key=lambda item: (-item[2], priority_order.get(item[0], 99), item[0]))

    minimal_subset = [item[0] for item in preferred[:3] if item[2] >= 0.8]
    if len(minimal_subset) < 2:
        minimal_subset = [item[0] for item in eligible[: min(2, len(eligible))]]

    recommended_modalities = ", ".join(f"`{modality}`" for modality in minimal_subset) if minimal_subset else "none"
    cohort_text = (
        "Recommend v1 as patients with a non-null `idh_label`, at least one tumour segmentation file "
        f"in the primary pre-operative visit, and complete availability of the minimal audited modality subset: {recommended_modalities}."
    )
    modality_text = f"Recommend the smallest high-completeness structural subset that survives the audit: {recommended_modalities}."
    return modality_text, cohort_text


def write_audit_markdown(summary: AuditSummary, audit_path: Path, figure_path: Path) -> tuple[str, str]:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    modality_text, cohort_text = recommend_v1_subset(summary)

    unresolved = summary.unresolved_issues or ["None recorded during automated audit."]
    markdown = f"""# Dataset Audit

## Verified counts

- Patient-level subjects discovered: **{summary.subject_count}**
- Visit directories discovered: **{summary.visit_directory_count}**
- Follow-up visit directories excluded from primary counts: **{summary.followup_directory_count}**
- Subjects with non-null IDH label: **{summary.idh_label_count}**
- Subjects with non-null MGMT label: **{summary.mgmt_label_count}**
- Subjects with at least one tumour segmentation file in the primary visit: **{summary.tumor_segmentation_subject_count}**
- Parquet manifest written: **{"yes" if summary.parquet_written else "no"}**

## Metadata and label tables

{_markdown_metadata_table(summary)}

## File types

{_markdown_file_types(summary)}

## Modality completeness

{_markdown_modality_table(summary)}

Figure: `{figure_path}`

## Recommended v1 cohort definition

- {cohort_text}

## Recommended minimal modality subset

- {modality_text}

## Unresolved issues

""" + "\n".join(f"- {issue}" for issue in unresolved) + "\n"

    audit_path.write_text(markdown, encoding="utf-8")
    return modality_text, cohort_text
