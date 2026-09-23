# -*- coding: utf-8 -*-
"""Lightweight inference preprocessing for the three PI-MFS tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PressureInput:
    matrices: np.ndarray
    preview: np.ndarray
    sample_count: int
    source_description: str


@dataclass
class ClassificationInput:
    tensor_array: np.ndarray
    preview: np.ndarray
    source_description: str


def _clean_array(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _sample_rows_evenly(
    rows: list[np.ndarray],
    max_rows: int | None,
    trim_ratio: float = 0.05,
) -> list[np.ndarray]:
    if not rows:
        return []

    total = len(rows)
    start = int(total * trim_ratio)
    end = int(total * (1.0 - trim_ratio))
    trimmed = rows if end <= start else rows[start:end]

    if max_rows is None or max_rows <= 0 or len(trimmed) <= max_rows:
        return trimmed
    indices = np.linspace(0, len(trimmed) - 1, max_rows).astype(int)
    return [trimmed[index] for index in indices]


def _read_tsv(
    path: Path,
    header_row: int,
    data_start_row: int,
    data_start_col: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    if len(lines) < data_start_row:
        raise ValueError(
            f"The TSV contains {len(lines)} rows; at least "
            f"{data_start_row} rows are required."
        )

    try:
        header = lines[header_row - 1].strip().split("\t")
        positions = np.asarray(
            header[data_start_col - 1 :],
            dtype=np.float64,
        )
    except Exception as exc:
        raise ValueError(
            f"Unable to read fiber positions from header row {header_row}."
        ) from exc

    if positions.size == 0:
        raise ValueError("No fiber positions were found in the TSV header.")

    rows: list[np.ndarray] = []
    for line in lines[data_start_row - 1 :]:
        parts = line.strip().split("\t")
        if len(parts) < data_start_col:
            continue
        try:
            row = np.asarray(parts[data_start_col - 1 :], dtype=np.float64)
        except ValueError:
            continue
        if row.size == 0 or np.all(np.isnan(row)):
            continue
        rows.append(row)

    if not rows:
        raise ValueError("No valid strain-data rows were found in the TSV.")
    return positions, rows


def _reconstruct_pressure_matrix(
    row_data: np.ndarray,
    positions: np.ndarray,
    center_positions: list[float],
    num_points: int,
    reverse_odd_rows: bool,
) -> np.ndarray:
    matrix_rows = []
    for row_index, center in enumerate(center_positions):
        center_index = int(np.argmin(np.abs(positions - center)))
        start = center_index - num_points // 2
        end = start + num_points

        if start < 0:
            start = 0
            end = num_points
        if end > len(row_data):
            end = len(row_data)
            start = end - num_points

        selected = row_data[start:end]
        if len(selected) != num_points:
            raise ValueError(
                f"Insufficient strain points to reconstruct a "
                f"{len(center_positions)}×{num_points} matrix."
            )
        if reverse_odd_rows and row_index % 2 == 1:
            selected = selected[::-1]
        matrix_rows.append(selected)

    return _clean_array(np.asarray(matrix_rows))


def load_pressure_input(
    file_path: str | Path,
    checkpoint: dict[str, Any],
    max_rows: int | None = 256,
) -> PressureInput:
    """Load raw Task 1 TSV/TXT directly, or accept a prepared NPY matrix."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrices = _clean_array(np.load(path))
        matrices = np.squeeze(matrices)
        if matrices.ndim == 2:
            matrices = matrices[np.newaxis, :, :]
        if matrices.ndim != 3 or matrices.shape[1:] != (30, 38):
            raise ValueError(
                f"Task 1 NPY must have shape (30,38) or (N,30,38); "
                f"received {matrices.shape}."
            )
        if max_rows and max_rows > 0 and len(matrices) > max_rows:
            indices = np.linspace(0, len(matrices) - 1, max_rows).astype(int)
            matrices = matrices[indices]
        description = f"Prepared NPY matrix · {len(matrices)} sample(s)"
    elif suffix in {".tsv", ".txt"}:
        header_row = int(checkpoint.get("header_row", 33))
        data_start_row = int(checkpoint.get("data_start_row", 34))
        data_start_col = int(checkpoint.get("data_start_col", 4))
        num_points = int(checkpoint.get("num_points", 38))
        center_positions = list(checkpoint.get("center_positions", []))
        reverse_odd_rows = bool(checkpoint.get("reverse_odd_rows", True))

        if len(center_positions) != 30:
            raise ValueError(
                "Task 1 checkpoint does not contain all 30 center positions. "
                "Please select the best checkpoint created by mamba(3).py."
            )

        positions, rows = _read_tsv(
            path,
            header_row,
            data_start_row,
            data_start_col,
        )
        selected_rows = _sample_rows_evenly(rows, max_rows=max_rows)
        matrices = np.stack(
            [
                _reconstruct_pressure_matrix(
                    row,
                    positions,
                    center_positions,
                    num_points,
                    reverse_odd_rows,
                )
                for row in selected_rows
            ],
            axis=0,
        ).astype(np.float32)
        description = (
            f"Raw TSV · {len(rows)} data rows · "
            f"{len(matrices)} stable rows used for inference"
        )
    else:
        raise ValueError("Task 1 supports .tsv, .txt, or .npy files.")

    if matrices.shape[1:] != (30, 38):
        raise ValueError(
            f"Task 1 reconstructed matrix must be (30,38); "
            f"received {matrices.shape[1:]}."
        )

    return PressureInput(
        matrices=matrices,
        preview=matrices.mean(axis=0),
        sample_count=len(matrices),
        source_description=description,
    )


def load_static_input(file_path: str | Path) -> ClassificationInput:
    """Load Task 2 NPY and apply the training-time normalization."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() != ".npy":
        raise ValueError("Task 2 requires a prepared two-dimensional .npy file.")

    matrix = _clean_array(np.squeeze(np.load(path)))
    if matrix.shape != (30, 38):
        raise ValueError(
            f"Task 2 input must have shape (30,38); received {matrix.shape}."
        )

    normalized = matrix.copy()
    maximum = float(np.max(normalized))
    if maximum > 0:
        normalized /= maximum

    return ClassificationInput(
        tensor_array=normalized[np.newaxis, np.newaxis, :, :],
        preview=normalized,
        source_description="Static two-dimensional sample · 30×38",
    )


def load_dynamic_input(
    file_path: str | Path,
    frame_shape: tuple[int, int] = (30, 38),
    target_frames: int = 50,
    normalize_mode: str = "max_abs",
) -> ClassificationInput:
    """Load a Task 3 NPY directly from the existing 30×38×50 database."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() != ".npy":
        raise ValueError("Task 3 requires a three-dimensional .npy sample.")

    volume_hwt = _clean_array(np.squeeze(np.load(path)))
    expected = (int(frame_shape[0]), int(frame_shape[1]), int(target_frames))
    if volume_hwt.shape != expected:
        raise ValueError(
            f"Task 3 input must have shape {expected}; "
            f"received {volume_hwt.shape}."
        )

    normalized = volume_hwt.copy()
    if normalize_mode == "max_abs":
        scale = float(np.max(np.abs(normalized)))
        if scale > 0:
            normalized /= scale
    elif normalize_mode == "max_positive":
        scale = float(np.max(normalized))
        if scale > 0:
            normalized /= scale
    elif normalize_mode != "none":
        raise ValueError(f"Unsupported Task 3 normalization mode: {normalize_mode}")

    # Checkpoint input shape: [B, C, T/D, H, W].
    volume_thw = np.transpose(normalized, (2, 0, 1))
    tensor_array = volume_thw[np.newaxis, np.newaxis, :, :, :]
    preview = np.max(np.abs(normalized), axis=2)

    return ClassificationInput(
        tensor_array=tensor_array.astype(np.float32),
        preview=preview.astype(np.float32),
        source_description=(
            f"Dynamic sample · {expected[0]}×{expected[1]}×"
            f"{expected[2]} · Temporal maximum projection"
        ),
    )
