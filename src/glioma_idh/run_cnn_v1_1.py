from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from glioma_idh.run_baseline_v1 import _compute_metrics


POSITIVE_LABEL = "mutant"
NEGATIVE_LABEL = "wildtype"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _encode_label(label: str) -> float:
    if label == NEGATIVE_LABEL:
        return 0.0
    if label == POSITIVE_LABEL:
        return 1.0
    raise ValueError(f"Unexpected IDH label: {label}")


def _count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _resize_cam(cam: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(cam.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized[0, 0].detach().cpu().numpy()


def _plot_curves_svg(
    curves: list[dict[str, Any]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 660, 460
    pad_left, pad_right, pad_top, pad_bottom = 70, 40, 40, 55
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom
    palette = ["#005f73", "#ae2012", "#0a9396", "#ca6702"]

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white' />",
        f"<text x='{width / 2:.1f}' y='24' text-anchor='middle' font-size='18' font-family='Arial, sans-serif'>{title}</text>",
    ]
    for tick in np.linspace(0.0, 1.0, 6):
        x = pad_left + tick * plot_width
        y = height - pad_bottom - tick * plot_height
        lines.append(f"<line x1='{x:.2f}' y1='{pad_top}' x2='{x:.2f}' y2='{height - pad_bottom}' stroke='#e2e2e2' stroke-width='1' />")
        lines.append(f"<line x1='{pad_left}' y1='{y:.2f}' x2='{width - pad_right}' y2='{y:.2f}' stroke='#e2e2e2' stroke-width='1' />")
        lines.append(f"<text x='{x:.2f}' y='{height - 14}' text-anchor='middle' font-size='12' font-family='Arial, sans-serif'>{tick:.1f}</text>")
        lines.append(f"<text x='{pad_left - 12}' y='{y + 4:.2f}' text-anchor='end' font-size='12' font-family='Arial, sans-serif'>{tick:.1f}</text>")
    lines.append(f"<rect x='{pad_left}' y='{pad_top}' width='{plot_width}' height='{plot_height}' fill='none' stroke='#222' stroke-width='1.5' />")
    for idx, curve in enumerate(curves):
        color = palette[idx % len(palette)]
        if curve["kind"] == "xy":
            points = []
            for x_value, y_value in zip(curve["x"], curve["y"]):
                x = pad_left + float(x_value) * plot_width
                y = height - pad_bottom - float(y_value) * plot_height
                points.append(f"{x:.2f},{y:.2f}")
        else:
            x_values = np.asarray(curve["x"], dtype=float)
            y_values = np.asarray(curve["y"], dtype=float)
            x_min = float(curve.get("x_min", x_values.min()))
            x_max = float(curve.get("x_max", x_values.max()))
            y_min = float(curve.get("y_min", y_values.min()))
            y_max = float(curve.get("y_max", y_values.max()))
            denom_x = max(x_max - x_min, 1e-8)
            denom_y = max(y_max - y_min, 1e-8)
            points = []
            for x_value, y_value in zip(x_values, y_values):
                x = pad_left + ((x_value - x_min) / denom_x) * plot_width
                y = height - pad_bottom - ((y_value - y_min) / denom_y) * plot_height
                points.append(f"{x:.2f},{y:.2f}")
        lines.append(f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{' '.join(points)}' />")
        legend_y = pad_top + 18 + idx * 20
        lines.append(f"<line x1='{width - 205}' y1='{legend_y}' x2='{width - 180}' y2='{legend_y}' stroke='{color}' stroke-width='3' />")
        lines.append(f"<text x='{width - 170}' y='{legend_y + 4}' font-size='12' font-family='Arial, sans-serif'>{curve['label']}</text>")
    lines.append(f"<text x='{width / 2:.1f}' y='{height - 10}' text-anchor='middle' font-size='14' font-family='Arial, sans-serif'>{x_label}</text>")
    lines.append(
        f"<text x='18' y='{height / 2:.1f}' text-anchor='middle' font-size='14' font-family='Arial, sans-serif' transform='rotate(-90 18 {height / 2:.1f})'>{y_label}</text>"
    )
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_random_shift(image: Tensor, max_shift: int) -> Tensor:
    if max_shift <= 0:
        return image
    shift_y = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    shift_x = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    if shift_y == 0 and shift_x == 0:
        return image
    output = torch.zeros_like(image)
    src_y_start = max(0, -shift_y)
    src_y_end = image.shape[1] - max(0, shift_y)
    dst_y_start = max(0, shift_y)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    src_x_start = max(0, -shift_x)
    src_x_end = image.shape[2] - max(0, shift_x)
    dst_x_start = max(0, shift_x)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    output[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[:, src_y_start:src_y_end, src_x_start:src_x_end]
    return output


class CNNInputDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool, shift_pixels: int, gaussian_noise_std: float) -> None:
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.shift_pixels = int(shift_pixels)
        self.gaussian_noise_std = float(gaussian_noise_std)

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        with np.load(row["tensor_path"]) as data:
            image = torch.from_numpy(data["image"].astype(np.float32))
            center_mask = torch.from_numpy(data["center_mask"].astype(np.float32))
            display_t1c = torch.from_numpy(data["display_t1c"].astype(np.float32))
        if self.augment:
            image = _apply_random_shift(image, self.shift_pixels)
            center_mask = _apply_random_shift(center_mask.unsqueeze(0), self.shift_pixels).squeeze(0)
            display_t1c = _apply_random_shift(display_t1c.unsqueeze(0), self.shift_pixels).squeeze(0)
            if self.gaussian_noise_std > 0:
                sigma = torch.rand(1).item() * self.gaussian_noise_std
                image = image + torch.randn_like(image) * sigma
        return {
            "image": image,
            "label": torch.tensor(_encode_label(str(row["idh_label"])), dtype=torch.float32),
            "subject_id": str(row["subject_id"]),
            "display_t1c": display_t1c,
            "center_mask": center_mask,
        }


class Compact2p5DCNN(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], dropout: float) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward_features(self, image: Tensor) -> Tensor:
        x = self.block1(image)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward_from_features(self, features: Tensor) -> Tensor:
        pooled = self.pool(features)
        logits = self.head(pooled)
        return logits.squeeze(1)

    def forward(self, image: Tensor) -> Tensor:
        features = self.forward_features(image)
        return self.forward_from_features(features)


def _frame_split(index_df: pd.DataFrame, validation_fold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_fit = index_df.loc[index_df["cnn_role"] == "train_fit"].copy().reset_index(drop=True)
    validation = index_df.loc[index_df["cnn_role"] == "validation"].copy().reset_index(drop=True)
    test = index_df.loc[index_df["cnn_role"] == "test"].copy().reset_index(drop=True)
    if validation["cv_fold"].nunique() != 1 or int(validation["cv_fold"].iloc[0]) != int(validation_fold):
        raise ValueError("Validation fold assignment in CNN input index does not match the frozen config.")
    return train_fit, validation, test


def _build_loaders(
    train_fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    training_cfg: dict[str, Any],
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataset = CNNInputDataset(
        train_fit,
        augment=True,
        shift_pixels=int(training_cfg["augmentation"]["shift_pixels"]),
        gaussian_noise_std=float(training_cfg["augmentation"]["gaussian_noise_std"]),
    )
    eval_kwargs = {
        "augment": False,
        "shift_pixels": 0,
        "gaussian_noise_std": 0.0,
    }
    validation_dataset = CNNInputDataset(validation, **eval_kwargs)
    test_dataset = CNNInputDataset(test, **eval_kwargs)
    common_kwargs = {
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": int(training_cfg["num_workers"]),
        "pin_memory": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)
    return train_loader, validation_loader, test_loader


def _build_full_train_loader(index_df: pd.DataFrame, training_cfg: dict[str, Any], seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = CNNInputDataset(
        index_df,
        augment=True,
        shift_pixels=int(training_cfg["augmentation"]["shift_pixels"]),
        gaussian_noise_std=float(training_cfg["augmentation"]["gaussian_noise_std"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=False,
    )


def _build_eval_loader(index_df: pd.DataFrame, training_cfg: dict[str, Any]) -> DataLoader:
    dataset = CNNInputDataset(index_df, augment=False, shift_pixels=0, gaussian_noise_std=0.0)
    return DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=False,
    )


def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
    model.eval()
    losses = []
    subjects: list[str] = []
    labels: list[float] = []
    probabilities: list[float] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = model(image)
            loss = criterion(logits, label)
            probability = torch.sigmoid(logits)
            prediction = (probability >= 0.5).int()
            losses.append(float(loss.item()) * len(label))
            subjects.extend(batch["subject_id"])
            labels.extend(label.detach().cpu().tolist())
            probabilities.extend(probability.detach().cpu().tolist())
            predictions.extend(prediction.detach().cpu().tolist())
    labels_arr = np.asarray(labels, dtype=int)
    probabilities_arr = np.asarray(probabilities, dtype=float)
    predictions_arr = np.asarray(predictions, dtype=int)
    metrics = _compute_metrics(labels_arr, probabilities_arr, predictions_arr)
    metrics["loss"] = float(sum(losses) / max(len(labels_arr), 1))
    return {
        "metrics": metrics,
        "subjects": subjects,
        "labels": labels_arr,
        "probabilities": probabilities_arr,
        "predictions": predictions_arr,
    }


def _train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(label)
        total_items += len(label)
    return float(total_loss / max(total_items, 1))


def _positive_weight(frame: pd.DataFrame) -> torch.Tensor:
    label_counts = frame["idh_label"].value_counts()
    positive = int(label_counts.get(POSITIVE_LABEL, 0))
    negative = int(label_counts.get(NEGATIVE_LABEL, 0))
    return torch.tensor(_safe_divide(negative, positive), dtype=torch.float32)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "payload": payload,
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def _load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def _roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def _history_figure(history: pd.DataFrame, output_path: Path) -> None:
    selection = history.loc[history["stage"] == "selection"].copy()
    if selection.empty:
        return
    epoch = selection["epoch"].to_numpy(dtype=float)
    x_min = float(epoch.min())
    x_max = float(epoch.max())
    y_values = np.concatenate([selection["train_loss"].to_numpy(dtype=float), selection["validation_loss"].to_numpy(dtype=float), selection["validation_roc_auc"].to_numpy(dtype=float)])
    curves = [
        {"label": "Train loss", "x": epoch, "y": selection["train_loss"].to_numpy(dtype=float), "kind": "series", "x_min": x_min, "x_max": x_max, "y_min": float(y_values.min()), "y_max": float(y_values.max())},
        {"label": "Val loss", "x": epoch, "y": selection["validation_loss"].to_numpy(dtype=float), "kind": "series", "x_min": x_min, "x_max": x_max, "y_min": float(y_values.min()), "y_max": float(y_values.max())},
        {"label": "Val ROC-AUC", "x": epoch, "y": selection["validation_roc_auc"].to_numpy(dtype=float), "kind": "series", "x_min": x_min, "x_max": x_max, "y_min": float(y_values.min()), "y_max": float(y_values.max())},
    ]
    _plot_curves_svg(curves, output_path, title="CNN Selection-Stage Training History", x_label="Epoch", y_label="Metric / Loss")


def _roc_figure(
    cnn_labels: np.ndarray,
    cnn_probabilities: np.ndarray,
    baseline_predictions_csv: Path,
    output_path: Path,
) -> None:
    baseline = pd.read_csv(baseline_predictions_csv)
    logistic_prob = baseline["logistic_regression_probability_mutant"].to_numpy(dtype=float)
    logistic_label = baseline["idh_binary"].to_numpy(dtype=int)
    fpr_log, tpr_log = _roc_curve(logistic_label, logistic_prob)
    fpr_cnn, tpr_cnn = _roc_curve(cnn_labels, cnn_probabilities)
    curves = [
        {"label": "Frozen logistic baseline", "x": fpr_log, "y": tpr_log, "kind": "xy"},
        {"label": "Compact 2.5D CNN", "x": fpr_cnn, "y": tpr_cnn, "kind": "xy"},
    ]
    _plot_curves_svg(curves, output_path, title="Held-out Test ROC Comparison", x_label="False Positive Rate", y_label="True Positive Rate")


def _normalize_for_display(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    if np.allclose(array.max(), array.min()):
        return np.zeros_like(array, dtype=np.uint8)
    normalized = (array - array.min()) / (array.max() - array.min())
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _heatmap_rgb(cam: np.ndarray) -> np.ndarray:
    cam = np.clip(cam, 0.0, 1.0)
    r = np.clip(255 * cam, 0, 255)
    g = np.clip(255 * np.sqrt(cam), 0, 255)
    b = np.clip(255 * (1.0 - cam), 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _gradcam(model: Compact2p5DCNN, image: Tensor, device: torch.device) -> np.ndarray:
    image = image.unsqueeze(0).to(device)
    image.requires_grad_(True)
    model.eval()
    features = model.forward_features(image)
    logits = model.forward_from_features(features)
    gradients = torch.autograd.grad(logits[0], features)[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32)


def _select_interpretability_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    cases = []
    mapping = [
        ("true_positive", (predictions["idh_binary"] == 1) & (predictions["cnn_prediction_binary"] == 1), "cnn_probability_mutant", False),
        ("true_negative", (predictions["idh_binary"] == 0) & (predictions["cnn_prediction_binary"] == 0), "cnn_probability_mutant", True),
        ("false_positive", (predictions["idh_binary"] == 0) & (predictions["cnn_prediction_binary"] == 1), "cnn_probability_mutant", False),
        ("false_negative", (predictions["idh_binary"] == 1) & (predictions["cnn_prediction_binary"] == 0), "cnn_probability_mutant", True),
    ]
    for case_name, mask, sort_col, ascending in mapping:
        subset = predictions.loc[mask].sort_values(sort_col, ascending=ascending)
        if not subset.empty:
            row = subset.iloc[0].copy()
            row["case_type"] = case_name
            cases.append(row)
    return pd.DataFrame(cases)


def _write_gradcam_figure(
    cases: pd.DataFrame,
    index_df: pd.DataFrame,
    model: Compact2p5DCNN,
    device: torch.device,
    output_path: Path,
) -> None:
    if cases.empty:
        return
    index_lookup = index_df.set_index("subject_id")
    panels = []
    for row in cases.to_dict(orient="records"):
        tensor_path = Path(index_lookup.loc[row["subject_id"], "tensor_path"])
        with np.load(tensor_path) as data:
            image_np = data["image"].astype(np.float32)
            display_t1c = data["display_t1c"].astype(np.float32)
            center_mask = data["center_mask"].astype(np.float32)
        image = torch.from_numpy(image_np)
        cam = _gradcam(model, image, device=device)
        base = _normalize_for_display(display_t1c)
        heat = _heatmap_rgb(cam)
        rgb = np.stack([base, base, base], axis=-1)
        overlay = np.clip(0.6 * rgb + 0.4 * heat, 0, 255).astype(np.uint8)
        border = np.zeros_like(overlay)
        border[:, :, 0] = (center_mask > 0).astype(np.uint8) * 255
        border_overlay = np.clip(overlay.astype(np.int16) + (0.25 * border.astype(np.int16)), 0, 255).astype(np.uint8)
        panel = Image.fromarray(border_overlay)
        draw = ImageDraw.Draw(panel)
        text = f"{row['case_type']}\n{row['subject_id']}\ntrue={row['idh_label']} pred={row['cnn_prediction_label']}\np={row['cnn_probability_mutant']:.3f}"
        draw.rectangle((0, 0, 170, 58), fill=(255, 255, 255))
        draw.text((6, 4), text, fill=ImageColor.getrgb("black"))
        panels.append(panel)

    columns = 2
    panel_width, panel_height = panels[0].size
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * panel_width, rows * panel_height), color="white")
    for idx, panel in enumerate(panels):
        x = (idx % columns) * panel_width
        y = (idx // columns) * panel_height
        canvas.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _design_note(
    output_path: Path,
    config: dict[str, Any],
    parameter_count: int,
    in_channels: int,
) -> None:
    training = config["training"]
    model_cfg = config["model"]
    lines = [
        "# CNN Design v1",
        "",
        "- Stage: bounded v1.1 image-model comparison against the frozen radiomics logistic baseline.",
        "- Architecture family: compact custom 2.5D CNN on deterministic axial slice stacks.",
        f"- Input channels: **{in_channels}**",
        f"- Convolution widths: `{model_cfg['channels']}`",
        f"- Dropout: **{model_cfg['dropout']}**",
        f"- Trainable parameters: **{parameter_count}**",
        f"- Optimizer: `AdamW` with learning rate **{training['learning_rate']}** and weight decay **{training['weight_decay']}**",
        f"- Batch size: **{training['batch_size']}**",
        f"- Loss: `{config['loss']['kind']}` with training-subset class weighting",
        f"- Max epochs: **{training['max_epochs']}**",
        f"- Early stopping patience: **{training['early_stopping_patience']}**",
        "",
        "## Train/validation protocol",
        "",
        f"- The frozen held-out test set remains untouched until final evaluation.",
        f"- Validation inside the training pool uses the frozen `cv_fold == {training['validation_fold']}` subjects only.",
        "- Epoch selection is based on validation ROC-AUC only.",
        "- After selecting the best epoch, the final checkpoint is refit once on the full frozen training pool for that fixed epoch count.",
        "- Augmentation is limited to small in-plane shifts and mild Gaussian noise on training batches only.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _results_note(
    output_path: Path,
    metrics: dict[str, float],
    comparison_df: pd.DataFrame,
    best_epoch: int,
    validation_metrics: dict[str, float],
) -> None:
    cnn_row = comparison_df.loc[comparison_df["model_name"] == "compact_2p5d_cnn"].iloc[0]
    log_row = comparison_df.loc[comparison_df["model_name"] == "logistic_regression"].iloc[0]
    lines = [
        "# CNN Results v1",
        "",
        f"- Best selection-stage epoch from validation ROC-AUC: **{best_epoch}**",
        f"- Selection-stage validation ROC-AUC at best epoch: **{validation_metrics['roc_auc']:.3f}**",
        "",
        "## Held-out test metrics",
        "",
        f"- ROC-AUC: **{metrics['roc_auc']:.3f}**",
        f"- Balanced accuracy: **{metrics['balanced_accuracy']:.3f}**",
        f"- Sensitivity: **{metrics['sensitivity']:.3f}**",
        f"- Specificity: **{metrics['specificity']:.3f}**",
        f"- Confusion matrix: `TN={metrics['confusion_tn']}, FP={metrics['confusion_fp']}, FN={metrics['confusion_fn']}, TP={metrics['confusion_tp']}`",
        "",
        "## Comparison against frozen logistic baseline",
        "",
        "| Model | ROC-AUC | Balanced Acc. | Sensitivity | Specificity |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_df.itertuples(index=False):
        lines.append(f"| {row.model_name} | {row.roc_auc:.3f} | {row.balanced_accuracy:.3f} | {row.sensitivity:.3f} | {row.specificity:.3f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- Relative to the frozen logistic baseline, the compact CNN changed ROC-AUC by "
                f"**{cnn_row.roc_auc - log_row.roc_auc:+.3f}**, balanced accuracy by **{cnn_row.balanced_accuracy - log_row.balanced_accuracy:+.3f}**, "
                f"sensitivity by **{cnn_row.sensitivity - log_row.sensitivity:+.3f}**, and specificity by **{cnn_row.specificity - log_row.specificity:+.3f}**."
            ),
            "- This comparison uses the same frozen cohort, modalities, ROI rule, preprocessing contract, and held-out test split as the radiomics baseline.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _interpretability_note(output_path: Path, cases: pd.DataFrame) -> None:
    lines = [
        "# CNN Interpretability v1",
        "",
        "- Method: Grad-CAM on the final convolutional block of the compact 2.5D CNN.",
        "- Display background: central resized T1c slice from the deterministic 2.5D tensorization.",
        "- Scope: small representative subset of held-out test cases.",
        "- Interpretation status: qualitative saliency only; not causal explanation and not a substitute for biological validation.",
        "",
        "## Included test cases",
        "",
    ]
    if cases.empty:
        lines.append("- No Grad-CAM cases were generated.")
    else:
        for row in cases.itertuples(index=False):
            lines.append(
                f"- `{row.case_type}`: `{row.subject_id}` with true label `{row.idh_label}` and predicted label `{row.cnn_prediction_label}` (mutant probability `{row.cnn_probability_mutant:.3f}`)"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cnn(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    index_df = pd.read_csv(Path(config["representation"]["output_index_csv"])).sort_values("subject_id").reset_index(drop=True)
    training_cfg = config["training"]
    validation_fold = int(training_cfg["validation_fold"])
    train_fit, validation, test = _frame_split(index_df, validation_fold)

    seed = int(training_cfg["seed"])
    _seed_everything(seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = int(index_df["tensor_channels"].iloc[0])
    model = Compact2p5DCNN(
        in_channels=in_channels,
        channels=[int(value) for value in config["model"]["channels"]],
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    parameter_count = _count_parameters(model)
    _design_note(Path(config["artifacts"]["design_note"]), config, parameter_count=parameter_count, in_channels=in_channels)

    train_loader, validation_loader, test_loader = _build_loaders(train_fit, validation, test, training_cfg, seed)
    pos_weight_fit = _positive_weight(train_fit).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_fit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg["learning_rate"]), weight_decay=float(training_cfg["weight_decay"]))

    best_epoch = 0
    best_validation_auc = -np.inf
    best_validation_metrics: dict[str, float] = {}
    patience = int(training_cfg["early_stopping_patience"])
    wait = 0
    history_rows: list[dict[str, Any]] = []
    selection_checkpoint = Path(config["artifacts"]["selection_checkpoint"])
    for epoch in range(1, int(training_cfg["max_epochs"]) + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation_eval = _evaluate(model, validation_loader, criterion, device)
        val_metrics = validation_eval["metrics"]
        history_rows.append(
            {
                "stage": "selection",
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_metrics["loss"],
                "validation_roc_auc": val_metrics["roc_auc"],
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "validation_sensitivity": val_metrics["sensitivity"],
                "validation_specificity": val_metrics["specificity"],
            }
        )
        if val_metrics["roc_auc"] > best_validation_auc + 1e-6:
            best_validation_auc = float(val_metrics["roc_auc"])
            best_epoch = int(epoch)
            best_validation_metrics = dict(val_metrics)
            wait = 0
            _save_checkpoint(
                selection_checkpoint,
                model,
                optimizer,
                epoch=epoch,
                payload={"validation_metrics": val_metrics, "parameter_count": parameter_count},
            )
        else:
            wait += 1
            if wait >= patience:
                break

    if best_epoch <= 0:
        raise RuntimeError("CNN selection stage did not produce a valid best epoch.")

    _seed_everything(seed)
    final_model = Compact2p5DCNN(
        in_channels=in_channels,
        channels=[int(value) for value in config["model"]["channels"]],
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    full_train = index_df.loc[index_df["split_set"] == "train"].copy().reset_index(drop=True)
    full_train_loader = _build_full_train_loader(full_train, training_cfg, seed)
    pos_weight_full = _positive_weight(full_train).to(device)
    final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    for epoch in range(1, best_epoch + 1):
        train_loss = _train_one_epoch(final_model, full_train_loader, final_optimizer, final_criterion, device)
        history_rows.append(
            {
                "stage": "final_refit",
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": np.nan,
                "validation_roc_auc": np.nan,
                "validation_balanced_accuracy": np.nan,
                "validation_sensitivity": np.nan,
                "validation_specificity": np.nan,
            }
        )
    final_checkpoint = Path(config["artifacts"]["final_checkpoint"])
    _save_checkpoint(
        final_checkpoint,
        final_model,
        final_optimizer,
        epoch=best_epoch,
        payload={"best_epoch_from_validation": best_epoch, "parameter_count": parameter_count},
    )

    test_eval = _evaluate(final_model, test_loader, final_criterion, device)
    test_metrics = test_eval["metrics"]
    history_df = pd.DataFrame(history_rows)
    history_csv = Path(config["artifacts"]["history_csv"])
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_csv, index=False)
    _history_figure(history_df, Path(config["artifacts"]["history_figure"]))

    predictions = pd.DataFrame(
        {
            "subject_id": test_eval["subjects"],
            "idh_binary": test_eval["labels"].astype(int),
            "idh_label": np.where(test_eval["labels"].astype(int) == 1, POSITIVE_LABEL, NEGATIVE_LABEL),
            "cnn_probability_mutant": test_eval["probabilities"],
            "cnn_prediction_binary": test_eval["predictions"].astype(int),
            "cnn_prediction_label": np.where(test_eval["predictions"].astype(int) == 1, POSITIVE_LABEL, NEGATIVE_LABEL),
            "split_set": "test",
        }
    ).sort_values("subject_id").reset_index(drop=True)
    predictions_csv = Path(config["artifacts"]["predictions_csv"])
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_csv, index=False)

    baseline_metrics = pd.read_csv(Path(config["sources"]["baseline_metrics_csv"]))
    logistic = baseline_metrics.loc[baseline_metrics["model_name"] == "logistic_regression"].iloc[0]
    comparison_df = pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "roc_auc": float(logistic["roc_auc"]),
                "balanced_accuracy": float(logistic["balanced_accuracy"]),
                "sensitivity": float(logistic["sensitivity"]),
                "specificity": float(logistic["specificity"]),
            },
            {
                "model_name": "compact_2p5d_cnn",
                "roc_auc": float(test_metrics["roc_auc"]),
                "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
                "sensitivity": float(test_metrics["sensitivity"]),
                "specificity": float(test_metrics["specificity"]),
            },
        ]
    )
    comparison_csv = Path(config["artifacts"]["comparison_csv"])
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_csv, index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "model_name": "compact_2p5d_cnn",
                **test_metrics,
                "best_epoch": int(best_epoch),
                "validation_roc_auc_at_best_epoch": float(best_validation_metrics["roc_auc"]),
                "parameter_count": int(parameter_count),
            }
        ]
    )
    metrics_csv = Path(config["artifacts"]["metrics_csv"])
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False)
    _roc_figure(
        cnn_labels=test_eval["labels"].astype(int),
        cnn_probabilities=test_eval["probabilities"],
        baseline_predictions_csv=Path(config["sources"]["baseline_predictions_csv"]),
        output_path=Path(config["artifacts"]["roc_figure"]),
    )
    _results_note(Path(config["artifacts"]["results_note"]), test_metrics, comparison_df, best_epoch, best_validation_metrics)

    cases = _select_interpretability_cases(predictions)
    gradcam_cases_csv = Path(config["artifacts"]["gradcam_cases_csv"])
    gradcam_cases_csv.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(gradcam_cases_csv, index=False)
    _write_gradcam_figure(
        cases=cases,
        index_df=index_df,
        model=final_model,
        device=device,
        output_path=Path(config["artifacts"]["gradcam_figure"]),
    )
    _interpretability_note(Path(config["artifacts"]["interpretability_note"]), cases)

    return {
        "metrics": metrics_df,
        "predictions": predictions,
        "comparison": comparison_df,
        "history": history_df,
        "cases": cases,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the bounded compact 2.5D CNN comparison model for IDH.")
    parser.add_argument("--config", default="configs/cnn_v1_1.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    outputs = run_cnn(Path(args.config))
    metrics = outputs["metrics"].iloc[0]
    print(f"Best epoch: {outputs['best_epoch']}")
    print(f"Parameter count: {outputs['parameter_count']}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Predictions CSV: {Path(_load_yaml(Path(args.config))['artifacts']['predictions_csv']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
