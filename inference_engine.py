# -*- coding: utf-8 -*-
"""Load three trained checkpoints and run inference without training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from model_definitions import (
    ConditionedPositionForceMamba,
    DynamicMambaClassifier,
    StaticMambaClassifier,
)
from preprocessing import (
    load_dynamic_input,
    load_pressure_input,
    load_static_input,
)


POSITION_LABEL_EN = {
    "zsvd": "Left top",
    "zxvd": "Left bottom",
    "zjvd": "Center",
    "ysvd": "Right top",
    "yxvd": "Right bottom",
    "left_top": "Left top",
    "left_bottom": "Left bottom",
    "center": "Center",
    "right_top": "Right top",
    "right_bottom": "Right bottom",
}


@dataclass
class InferenceResult:
    task_id: int
    primary_text: str
    secondary_text: str
    confidence: float | None
    top_items: list[tuple[str, float]] = field(default_factory=list)
    preview: np.ndarray | None = None
    source_description: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "The selected file is not a compatible checkpoint dictionary. "
            "Please select a best_model*.pth file saved by the training program."
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError("The checkpoint does not contain model_state_dict.")
    return checkpoint


def _top_items(probabilities: np.ndarray, labels: list[str], k: int = 3):
    k = min(k, len(labels))
    indices = np.argsort(probabilities)[::-1][:k]
    return [(labels[int(index)], float(probabilities[int(index)])) for index in indices]


def _format_static_label(label: str) -> str:
    return label


def _format_dynamic_label(label: str) -> str:
    if label.lower().startswith("dongtai"):
        return label[len("dongtai") :]
    return label


def _parse_task1_ground_truth(
    input_path: str | Path,
    gravity: float,
) -> tuple[float | None, float | None]:
    """
    Parse labels such as zsvd400_*.tsv or zxvd950_*.tsv.

    Returns (weight_g, force_n). Files without an encoded weight return
    (None, None), in which case the UI displays prediction stability instead
    of a regression error.
    """
    compact_name = Path(input_path).name.lower().replace(" ", "")
    if "yaliji" in compact_name:
        return None, None
    for position_key in ("zsvd", "zxvd", "zjvd", "ysvd", "yxvd"):
        match = re.match(
            rf"^{position_key}(\d+(?:\.\d+)?)",
            compact_name,
        )
        if match:
            weight_g = float(match.group(1))
            return weight_g, weight_g / 1000.0 * gravity
    return None, None


class MambaInferenceEngine:
    """Load models on demand and cache each unchanged checkpoint path."""

    def __init__(self, device_preference: str = "auto") -> None:
        self.device = self._resolve_device(device_preference)
        self._task1_model = None
        self._task2_model = None
        self._task3_model = None
        self._task1_checkpoint: dict[str, Any] | None = None
        self._task2_checkpoint: dict[str, Any] | None = None
        self._task3_checkpoint: dict[str, Any] | None = None
        self._loaded_paths: dict[int, str] = {}

    @staticmethod
    def _resolve_device(preference: str) -> torch.device:
        preference = preference.lower().strip()
        if preference == "cpu":
            return torch.device("cpu")
        if preference == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was selected, but PyTorch did not detect a compatible GPU."
                )
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return f"CUDA · {torch.cuda.get_device_name(self.device)}"
        return "CPU"

    def _is_loaded(self, task_id: int, path: str | Path) -> bool:
        return self._loaded_paths.get(task_id) == str(Path(path).resolve())

    def load_task1(self, model_path: str | Path) -> None:
        if self._is_loaded(1, model_path):
            return
        checkpoint = _load_checkpoint(model_path, self.device)
        model_name = str(checkpoint.get("model_name", ""))
        if model_name and model_name != "ConditionedPositionForceMamba":
            raise ValueError(
                "Task 1 requires mamba_conditioned_position_force_best.pth. "
                f"The selected checkpoint reports model_name={model_name!r}."
            )
        model_config = checkpoint.get(
            "model_config",
            {
                "d_model": 96,
                "d_state": 16,
                "num_layers": 2,
                "expand": 2,
                "conv_kernel": 4,
                "dropout": 0.1,
            },
        )
        model = ConditionedPositionForceMamba(**model_config).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        for key in ("X_mean", "X_std", "force_mean", "force_std"):
            if key not in checkpoint:
                raise ValueError(
                    f"Task 1 checkpoint is missing normalization parameter: {key}"
                )

        self._task1_model = model
        self._task1_checkpoint = checkpoint
        self._loaded_paths[1] = str(Path(model_path).resolve())

    def load_task2(self, model_path: str | Path) -> None:
        if self._is_loaded(2, model_path):
            return
        checkpoint = _load_checkpoint(model_path, self.device)
        class_names = list(checkpoint.get("class_names", []))
        if not class_names:
            raise ValueError("Task 2 checkpoint does not contain class_names.")

        config = checkpoint.get("mamba_config", {})
        map_shape = tuple(checkpoint.get("map_shape", (30, 38)))
        model = StaticMambaClassifier(
            num_classes=len(class_names),
            d_model=int(config.get("d_model", 96)),
            d_state=int(config.get("d_state", 16)),
            num_layers=int(config.get("num_layers", 2)),
            expand=int(config.get("expand", 2)),
            conv_kernel=int(config.get("conv_kernel", 4)),
            dropout=float(config.get("dropout", 0.1)),
            map_rows=int(map_shape[0]),
            map_cols=int(map_shape[1]),
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        self._task2_model = model
        self._task2_checkpoint = checkpoint
        self._loaded_paths[2] = str(Path(model_path).resolve())

    def load_task3(self, model_path: str | Path) -> None:
        if self._is_loaded(3, model_path):
            return
        checkpoint = _load_checkpoint(model_path, self.device)
        class_names = list(checkpoint.get("class_names", []))
        if not class_names:
            raise ValueError("Task 3 checkpoint does not contain class_names.")

        config = checkpoint.get("mamba_config", {})
        frame_shape = tuple(checkpoint.get("frame_shape", (30, 38)))
        target_frames = int(checkpoint.get("target_frames", 50))
        model = DynamicMambaClassifier(
            num_classes=len(class_names),
            depth=target_frames,
            height=int(frame_shape[0]),
            width=int(frame_shape[1]),
            d_model=int(config.get("d_model", 64)),
            d_state=int(config.get("d_state", 12)),
            num_layers=int(config.get("num_layers", 2)),
            expand=int(config.get("expand", 2)),
            conv_kernel=int(config.get("conv_kernel", 4)),
            dropout=float(config.get("dropout", 0.15)),
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        self._task3_model = model
        self._task3_checkpoint = checkpoint
        self._loaded_paths[3] = str(Path(model_path).resolve())

    def predict_task1(
        self,
        model_path: str | Path,
        input_path: str | Path,
        max_rows: int | None = 256,
        manual_position_id: int | None = None,
        batch_size: int = 64,
    ) -> InferenceResult:
        self.load_task1(model_path)
        assert self._task1_model is not None
        assert self._task1_checkpoint is not None

        checkpoint = self._task1_checkpoint
        prepared = load_pressure_input(input_path, checkpoint, max_rows=max_rows)
        matrices = (
            prepared.matrices - float(checkpoint["X_mean"])
        ) / (float(checkpoint["X_std"]) + 1e-8)
        matrices = matrices[:, np.newaxis, :, :].astype(np.float32)

        force_batches = []
        position_probability_batches = []
        used_position_batches = []
        with torch.inference_mode():
            for start in range(0, len(matrices), batch_size):
                x = torch.from_numpy(matrices[start : start + batch_size]).to(
                    self.device
                )
                pred_z, position_logits, used_position = (
                    self._task1_model.forward_auto(
                        x,
                        manual_position_id=manual_position_id,
                    )
                )
                force_n = (
                    pred_z.squeeze(1).cpu().numpy()
                    * float(checkpoint["force_std"])
                    + float(checkpoint["force_mean"])
                )
                force_batches.append(force_n)
                position_probability_batches.append(
                    torch.softmax(position_logits, dim=1).cpu().numpy()
                )
                used_position_batches.append(used_position.cpu().numpy())

        force_values = np.concatenate(force_batches)
        position_probabilities = np.concatenate(position_probability_batches)
        used_positions = np.concatenate(used_position_batches)

        mean_position_probability = position_probabilities.mean(axis=0)
        if manual_position_id is None:
            position_id = int(np.argmax(np.bincount(used_positions, minlength=5)))
        else:
            position_id = int(manual_position_id)

        id_to_position = checkpoint.get(
            "id_to_position",
            {0: "zsvd", 1: "zxvd", 2: "zjvd", 3: "ysvd", 4: "yxvd"},
        )
        position_cn = checkpoint.get("position_cn", {})
        position_key = str(
            id_to_position.get(position_id, id_to_position.get(str(position_id), position_id))
        )
        position_label = str(position_cn.get(position_key, position_key))
        position_label = POSITION_LABEL_EN.get(position_label, position_label)

        position_labels = []
        for index in range(5):
            key = str(id_to_position.get(index, id_to_position.get(str(index), index)))
            label = str(position_cn.get(key, key))
            position_labels.append(POSITION_LABEL_EN.get(label, label))

        mean_force = float(np.mean(force_values))
        std_force = float(np.std(force_values))
        gravity = float(checkpoint.get("gravity", 9.80665))
        equivalent_grams = mean_force / gravity * 1000.0
        true_weight_g, true_force_n = _parse_task1_ground_truth(
            input_path,
            gravity,
        )
        force_error_n = (
            mean_force - true_force_n if true_force_n is not None else None
        )
        relative_error_percent = (
            force_error_n / true_force_n * 100.0
            if true_force_n not in (None, 0.0)
            else None
        )
        secondary_parts = [
            f"Equivalent mass {equivalent_grams:.1f} g",
            f"Position {position_label}",
            f"Variation ±{std_force:.4f} N",
        ]
        if force_error_n is not None:
            secondary_parts.append(f"Error {force_error_n:+.4f} N")

        return InferenceResult(
            task_id=1,
            primary_text=f"{mean_force:.4f} N",
            secondary_text=" · ".join(secondary_parts),
            confidence=None,
            top_items=[],
            preview=prepared.preview,
            source_description=prepared.source_description,
            details={
                "force_n": mean_force,
                "force_std_n": std_force,
                "force_values_n": force_values.astype(float).tolist(),
                "equivalent_grams": equivalent_grams,
                "position_id": position_id,
                "position": position_label,
                "position_probabilities": (
                    mean_position_probability.astype(float).tolist()
                ),
                "sample_count": prepared.sample_count,
                "true_weight_g": true_weight_g,
                "true_force_n": true_force_n,
                "force_error_n": force_error_n,
                "relative_error_percent": relative_error_percent,
            },
        )

    def predict_task2(
        self,
        model_path: str | Path,
        input_path: str | Path,
    ) -> InferenceResult:
        self.load_task2(model_path)
        assert self._task2_model is not None
        assert self._task2_checkpoint is not None

        prepared = load_static_input(input_path)
        x = torch.from_numpy(prepared.tensor_array).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self._task2_model(x), dim=1)
        probabilities_np = probabilities[0].cpu().numpy()

        raw_labels = list(self._task2_checkpoint["class_names"])
        display_labels = [_format_static_label(str(label)) for label in raw_labels]
        predicted_id = int(np.argmax(probabilities_np))
        top = _top_items(probabilities_np, display_labels, k=3)

        return InferenceResult(
            task_id=2,
            primary_text=display_labels[predicted_id],
            secondary_text=(
                f"Static recognition · Original label "
                f"{raw_labels[predicted_id]}"
            ),
            confidence=float(probabilities_np[predicted_id]),
            top_items=top,
            preview=prepared.preview,
            source_description=prepared.source_description,
            details={
                "class_id": predicted_id,
                "class_name": str(raw_labels[predicted_id]),
                "class_labels": display_labels,
                "class_probabilities": probabilities_np.astype(float).tolist(),
            },
        )

    def predict_task3(
        self,
        model_path: str | Path,
        input_path: str | Path,
    ) -> InferenceResult:
        self.load_task3(model_path)
        assert self._task3_model is not None
        assert self._task3_checkpoint is not None

        checkpoint = self._task3_checkpoint
        frame_shape = tuple(checkpoint.get("frame_shape", (30, 38)))
        target_frames = int(checkpoint.get("target_frames", 50))
        normalize_mode = str(checkpoint.get("normalize_mode", "max_abs"))
        prepared = load_dynamic_input(
            input_path,
            frame_shape=(int(frame_shape[0]), int(frame_shape[1])),
            target_frames=target_frames,
            normalize_mode=normalize_mode,
        )

        x = torch.from_numpy(prepared.tensor_array).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self._task3_model(x), dim=1)
        probabilities_np = probabilities[0].cpu().numpy()

        raw_labels = list(checkpoint["class_names"])
        display_labels = [_format_dynamic_label(str(label)) for label in raw_labels]
        predicted_id = int(np.argmax(probabilities_np))
        top = _top_items(probabilities_np, display_labels, k=3)

        return InferenceResult(
            task_id=3,
            primary_text=display_labels[predicted_id],
            secondary_text=(
                f"Dynamic recognition · Original label "
                f"{raw_labels[predicted_id]}"
            ),
            confidence=float(probabilities_np[predicted_id]),
            top_items=top,
            preview=prepared.preview,
            source_description=prepared.source_description,
            details={
                "class_id": predicted_id,
                "class_name": str(raw_labels[predicted_id]),
                "class_labels": display_labels,
                "class_probabilities": probabilities_np.astype(float).tolist(),
            },
        )
