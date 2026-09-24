# -*- coding: utf-8 -*-
"""Streamlit web interface for the three PI-MFS inference tasks."""

from __future__ import annotations

import base64
import html
import os
from pathlib import Path
import shutil
import tempfile
import threading

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from inference_engine import InferenceResult, MambaInferenceEngine


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
EXAMPLE_DIR = ROOT / "examples"
ASSET_DIR = ROOT / "assets"

POSITION_OPTIONS = {
    "Automatic position recognition": None,
    "Left top": 0,
    "Left bottom": 1,
    "Center": 2,
    "Right top": 3,
    "Right bottom": 4,
}

TASK_MODEL_KEYWORDS = {
    1: ("task1", "pressure", "conditioned", "position", "force"),
    2: ("task2", "static", "jingta"),
    3: ("task3", "dynamic", "dongtai", "3d"),
}


class Runtime:
    def __init__(self) -> None:
        self.engine = MambaInferenceEngine(
            device_preference=os.getenv("PI_MFS_DEVICE", "auto")
        )
        self.lock = threading.RLock()


@st.cache_resource(show_spinner=False)
def get_runtime() -> Runtime:
    return Runtime()


def _files_under(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ),
        key=lambda path: path.as_posix().lower(),
    )


def _model_candidates(task_id: int) -> list[Path]:
    all_models = _files_under(MODEL_DIR, {".pth", ".pt"})
    keywords = TASK_MODEL_KEYWORDS[task_id]
    matched = [
        path
        for path in all_models
        if any(keyword in path.name.lower() for keyword in keywords)
    ]
    return matched or all_models


def _example_candidates(task_id: int) -> list[Path]:
    suffixes = {".tsv", ".txt", ".npy"} if task_id == 1 else {".npy"}
    preferred = EXAMPLE_DIR / f"task{task_id}"
    if preferred.is_dir():
        return _files_under(preferred, suffixes)

    all_examples = _files_under(EXAMPLE_DIR, suffixes)
    tokens = {
        1: ("task1", "pressure"),
        2: ("task2", "static"),
        3: ("task3", "dynamic"),
    }[task_id]
    matched = [
        path
        for path in all_examples
        if any(token in path.as_posix().lower() for token in tokens)
    ]
    return matched or all_examples


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _image_data_uri(path: Path) -> str:
    """Embed a repository image in the custom header without an extra server route."""
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _save_upload(uploaded_file, temp_directories: list[Path]) -> Path:
    temp_directory = Path(tempfile.mkdtemp(prefix="pi_mfs_"))
    temp_directories.append(temp_directory)
    safe_name = Path(uploaded_file.name).name
    destination = temp_directory / safe_name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def _resolve_input(
    source_mode: str,
    repository_path: Path | None,
    uploaded_file,
    label: str,
    temp_directories: list[Path],
) -> Path:
    if source_mode == "Repository file":
        if repository_path is None or not Path(repository_path).is_file():
            raise ValueError(f"No {label} was found in the repository.")
        return Path(repository_path)
    if uploaded_file is None:
        raise ValueError(f"Please upload {label}.")
    return _save_upload(uploaded_file, temp_directories)


def _build_figure(result: InferenceResult):
    figure, (response_axes, diagnostic_axes) = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.2),
        dpi=120,
    )
    figure.patch.set_facecolor("white")

    preview = np.asarray(result.preview, dtype=float)
    image = response_axes.imshow(
        preview,
        cmap="turbo",
        aspect="equal",
        interpolation="nearest",
    )
    response_axes.set_title(
        {
            1: "Task 1 · Mean Strain Map",
            2: "Task 2 · Static Response Map",
            3: "Task 3 · Temporal Maximum Projection",
        }[result.task_id]
    )
    response_axes.set_xlabel("Width")
    response_axes.set_ylabel("Height")
    colorbar = figure.colorbar(image, ax=response_axes, fraction=0.050, pad=0.04)
    colorbar.set_label("Response")

    if result.task_id == 1:
        _plot_regression(diagnostic_axes, result)
    else:
        _plot_probabilities(diagnostic_axes, result)

    figure.tight_layout()
    return figure


def _plot_regression(axes, result: InferenceResult) -> None:
    values = np.asarray(result.details.get("force_values_n", []), dtype=float)
    if values.size == 0:
        axes.text(0.5, 0.5, "No force predictions available", ha="center", va="center")
        axes.set_axis_off()
        return

    indices = np.arange(1, len(values) + 1)
    predicted_mean = float(result.details["force_n"])
    axes.plot(
        indices,
        values,
        color="#168B9A",
        linewidth=1.0,
        label="Per-row prediction",
    )
    axes.axhline(
        predicted_mean,
        color="#D95D39",
        linewidth=1.5,
        label=f"Predicted mean: {predicted_mean:.3f} N",
    )

    true_force = result.details.get("true_force_n")
    error = result.details.get("force_error_n")
    if true_force is not None:
        axes.axhline(
            float(true_force),
            color="#222222",
            linestyle="--",
            linewidth=1.4,
            label=f"Reference: {float(true_force):.3f} N",
        )
        axes.set_title(f"Regression Error · {float(error):+.4f} N")
    else:
        axes.set_title("Prediction Stability")

    axes.set_xlabel("TSV row used for inference")
    axes.set_ylabel("Predicted force (N)")
    axes.grid(True, alpha=0.20)
    axes.legend(fontsize=8, frameon=False, loc="best")


def _plot_probabilities(axes, result: InferenceResult) -> None:
    labels = [str(value) for value in result.details.get("class_labels", [])]
    probabilities = np.asarray(
        result.details.get("class_probabilities", []),
        dtype=float,
    )
    if not labels or probabilities.size != len(labels):
        axes.text(0.5, 0.5, "No class probabilities available", ha="center", va="center")
        axes.set_axis_off()
        return

    predicted_id = int(result.details.get("class_id", np.argmax(probabilities)))
    colors = [
        "#168B9A" if index == predicted_id else "#A9D6DC"
        for index in range(len(labels))
    ]
    x_values = np.arange(len(labels))
    axes.bar(x_values, probabilities * 100.0, color=colors, width=0.72)
    axes.set_xticks(x_values)
    axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes.set_ylim(0.0, 105.0)
    axes.set_ylabel("Probability (%)")
    axes.set_title("Class Probability Distribution")
    axes.grid(True, axis="y", alpha=0.20)


def _result_rows(result: InferenceResult) -> list[list[str]]:
    if result.task_id == 1:
        rows = [
            ["Predicted force", f"{float(result.details['force_n']):.4f} N"],
            ["Equivalent mass", f"{float(result.details['equivalent_grams']):.1f} g"],
            ["Position", str(result.details["position"])],
            ["Variation", f"±{float(result.details['force_std_n']):.4f} N"],
            ["Samples", str(result.details["sample_count"])],
        ]
        if result.details.get("true_force_n") is not None:
            rows.extend(
                [
                    ["Reference force", f"{float(result.details['true_force_n']):.4f} N"],
                    ["Force error", f"{float(result.details['force_error_n']):+.4f} N"],
                    [
                        "Relative error",
                        f"{float(result.details['relative_error_percent']):+.2f}%",
                    ],
                ]
            )
        return rows

    return [
        [f"#{index + 1} · {label}", f"{probability * 100:.2f}%"]
        for index, (label, probability) in enumerate(result.top_items)
    ]


def _render_result(result: InferenceResult) -> None:
    confidence = ""
    if result.confidence is not None:
        confidence = f"<strong>Confidence:</strong> {result.confidence * 100:.1f}%"
    st.markdown(
        f"""
        <div class="result-box">
          <div class="result-kicker">RECOGNITION RESULT</div>
          <div class="result-value">{html.escape(result.primary_text)}</div>
          <div class="result-detail">{html.escape(result.secondary_text)}</div>
          <div class="result-confidence">{confidence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    if result.task_id == 1:
        metrics = [
            ("Equivalent mass", f"{float(result.details['equivalent_grams']):.1f} g"),
            ("Position", str(result.details["position"])),
            ("Variation", f"±{float(result.details['force_std_n']):.4f} N"),
            ("Inference rows", str(result.details["sample_count"])),
        ]
    else:
        probabilities = result.details.get("class_probabilities", [])
        metrics = [
            ("Confidence", f"{float(result.confidence or 0.0) * 100:.1f}%"),
            ("Classes", str(len(probabilities))),
            ("Task", "Static" if result.task_id == 2 else "Dynamic"),
            ("Device", get_runtime().engine.device_name),
        ]
    for column, (label, value) in zip(metric_columns, metrics):
        column.metric(label, value)

    figure = _build_figure(result)
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

    table_title = "Metrics" if result.task_id == 1 else "Top-3 predictions"
    st.markdown(f"#### {table_title}")
    st.dataframe(
        pd.DataFrame(_result_rows(result), columns=["Metric / Class", "Value / Probability"]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"{result.source_description} · {get_runtime().engine.device_name}")


def _render_empty() -> None:
    st.markdown(
        """
        <div class="empty-box">
          <div class="empty-title">Ready for inference</div>
          <div>Select a checkpoint and an input file, then run the task.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_task(
    task_id: int,
    model_mode: str,
    model_repository_path: Path | None,
    model_upload,
    data_mode: str,
    data_repository_path: Path | None,
    data_upload,
    max_rows: int | None = None,
    position_name: str | None = None,
) -> InferenceResult:
    temp_directories: list[Path] = []
    try:
        model_path = _resolve_input(
            model_mode,
            model_repository_path,
            model_upload,
            f"Task {task_id} checkpoint",
            temp_directories,
        )
        data_path = _resolve_input(
            data_mode,
            data_repository_path,
            data_upload,
            f"Task {task_id} input",
            temp_directories,
        )
        runtime = get_runtime()
        with runtime.lock:
            if task_id == 1:
                return runtime.engine.predict_task1(
                    model_path,
                    data_path,
                    max_rows=max_rows,
                    manual_position_id=POSITION_OPTIONS.get(position_name),
                )
            if task_id == 2:
                return runtime.engine.predict_task2(model_path, data_path)
            return runtime.engine.predict_task3(model_path, data_path)
    finally:
        for directory in temp_directories:
            shutil.rmtree(directory, ignore_errors=True)


def _source_controls(task_id: int, data_types: list[str]):
    model_candidates = _model_candidates(task_id)
    example_candidates = _example_candidates(task_id)

    model_mode = st.radio(
        "Model source",
        ["Repository file", "Upload file"],
        horizontal=True,
        key=f"model_mode_{task_id}",
    )
    model_path = None
    model_upload = None
    if model_mode == "Repository file":
        model_path = st.selectbox(
            "Model checkpoint",
            options=model_candidates,
            format_func=_display_path,
            index=0 if model_candidates else None,
            placeholder="No checkpoint found in models/",
            key=f"model_repo_{task_id}",
        )
        if not model_candidates:
            st.warning("No .pth or .pt file was found in models/.")
    else:
        model_upload = st.file_uploader(
            "Model checkpoint",
            type=["pth", "pt"],
            key=f"model_upload_{task_id}",
        )

    data_mode = st.radio(
        "Input source",
        ["Repository file", "Upload file"],
        horizontal=True,
        key=f"data_mode_{task_id}",
    )
    data_path = None
    data_upload = None
    if data_mode == "Repository file":
        data_path = st.selectbox(
            "Input sample",
            options=example_candidates,
            format_func=_display_path,
            index=0 if example_candidates else None,
            placeholder=f"No sample found in examples/task{task_id}/",
            key=f"data_repo_{task_id}",
        )
        if not example_candidates:
            st.warning(f"No compatible file was found in examples/task{task_id}/.")
    else:
        data_upload = st.file_uploader(
            "Input sample",
            type=data_types,
            key=f"data_upload_{task_id}",
        )

    return model_mode, model_path, model_upload, data_mode, data_path, data_upload


def _classification_tab(task_id: int, title: str, note: str) -> None:
    input_column, output_column = st.columns([0.42, 0.58], gap="large")
    with input_column:
        st.markdown(f"### {title}")
        with st.form(f"task_{task_id}_form", border=False):
            controls = _source_controls(task_id, ["npy"])
            submitted = st.form_submit_button(
                "Run Static Recognition" if task_id == 2 else "Run Dynamic Recognition",
                type="primary",
                use_container_width=True,
            )
        st.caption(note)

    if submitted:
        try:
            with st.spinner("Loading checkpoint and running inference…"):
                st.session_state[f"task_{task_id}_result"] = _run_task(
                    task_id,
                    *controls,
                )
        except Exception as exc:
            st.session_state.pop(f"task_{task_id}_result", None)
            st.error(f"{type(exc).__name__}: {exc}")

    with output_column:
        result = st.session_state.get(f"task_{task_id}_result")
        if result is None:
            _render_empty()
        else:
            _render_result(result)


st.set_page_config(
    page_title="PI-MFS Framework",
    page_icon="〽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: #f4f7fb; }
      .block-container { max-width: 1500px; padding-top: 1.2rem; padding-bottom: 2.5rem; }
      .pi-header {
        display: flex; justify-content: space-between; align-items: center;
        gap: 1.5rem; padding: 1.35rem 1.6rem; margin-bottom: 1rem;
        color: white; background: #07162d; border-radius: 1rem;
      }
      .pi-kicker { color: #9ee7df; font-size: .78rem; font-weight: 800; letter-spacing: .12em; }
      .pi-title { margin-top: .2rem; font-family: Georgia, serif; font-size: 2rem; font-weight: 800; }
      .pi-subtitle { color: #d4e1ee; font-size: .95rem; }
      .runtime-line { margin-top: .55rem; color: #9ee7df; font-size: .78rem; font-weight: 700; letter-spacing: .04em; }
      .logo-panel {
        display: flex; flex-direction: column; justify-content: center; gap: .45rem;
        width: min(25rem, 43vw); padding: .65rem .85rem;
        border: 1px solid rgba(255,255,255,.25); border-radius: .8rem;
        background: rgba(255,255,255,.96);
      }
      .logo-panel img { display: block; width: 100%; height: auto; object-fit: contain; }
      .logo-panel .university-logo { max-height: 4.2rem; }
      .logo-panel .institute-logo { max-height: 3rem; padding-top: .35rem; border-top: 1px solid #e2e8ef; }
      .result-box { min-height: 8.5rem; padding: 1.1rem 1.25rem; margin-bottom: .8rem; border: 1px solid #dce5ee; border-radius: .85rem; background: white; box-shadow: 0 12px 30px rgba(20,43,72,.06); }
      .result-kicker { color: #087f8c; font-size: .72rem; font-weight: 800; letter-spacing: .12em; }
      .result-value { margin: .2rem 0 .35rem; color: #075f6c; font-family: Georgia, serif; font-size: 2.6rem; font-weight: 800; }
      .result-detail { color: #53657a; }
      .result-confidence { margin-top: .4rem; color: #172033; }
      .empty-box { display: grid; place-content: center; min-height: 24rem; padding: 2rem; text-align: center; color: #6f7f91; border: 1px dashed #b9c8d5; border-radius: .85rem; background: rgba(255,255,255,.7); }
      .empty-title { margin-bottom: .35rem; color: #26384d; font-family: Georgia, serif; font-size: 1.35rem; font-weight: 800; }
      div[data-testid="stForm"] { padding: 0; border: 0; }
      div[data-testid="stMetric"] { padding: .7rem; border: 1px solid #e0e8ef; border-radius: .7rem; background: white; }
      @media (max-width: 760px) {
        .pi-header { align-items: flex-start; flex-direction: column; }
        .logo-panel { width: 100%; }
        .result-value { font-size: 2rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <header class="pi-header">
      <div>
        <div class="pi-kicker">MULTIDIMENSIONAL FIBER SENSING</div>
        <div class="pi-title">PI-MFS Framework</div>
        <div class="pi-subtitle">Physical-Information-Driven Framework for Multidimensional Fiber Sensing</div>
        <div class="runtime-line">Runtime device · {html.escape(get_runtime().engine.device_name)}</div>
      </div>
      <div class="logo-panel" aria-label="Institutional affiliations">
        <img class="university-logo" src="{_image_data_uri(ASSET_DIR / 'northeastern_university_logo.png')}" alt="Northeastern University">
        <img class="institute-logo" src="{_image_data_uri(ASSET_DIR / 'institute_logo.png')}" alt="Research institute">
      </div>
    </header>
    """,
    unsafe_allow_html=True,
)

pressure_tab, static_tab, dynamic_tab = st.tabs(
    ["Task 1 · Pressure", "Task 2 · Static", "Task 3 · Dynamic"]
)

with pressure_tab:
    input_column, output_column = st.columns([0.42, 0.58], gap="large")
    with input_column:
        st.markdown("### Pressure prediction")
        with st.form("task_1_form", border=False):
            task1_controls = _source_controls(1, ["tsv", "txt", "npy"])
            maximum_rows = st.number_input(
                "Inference rows (0 = all)",
                min_value=0,
                max_value=100000,
                value=256,
                step=1,
            )
            position_name = st.selectbox(
                "Force position",
                options=list(POSITION_OPTIONS),
                index=0,
            )
            task1_submitted = st.form_submit_button(
                "Run Pressure Prediction",
                type="primary",
                use_container_width=True,
            )
        st.caption(
            "The raw OFDR TSV is reconstructed into 30×38 matrices using parameters stored in the Task 1 checkpoint."
        )

    if task1_submitted:
        try:
            with st.spinner("Loading checkpoint and running inference…"):
                st.session_state["task_1_result"] = _run_task(
                    1,
                    *task1_controls,
                    max_rows=None if maximum_rows == 0 else int(maximum_rows),
                    position_name=position_name,
                )
        except Exception as exc:
            st.session_state.pop("task_1_result", None)
            st.error(f"{type(exc).__name__}: {exc}")

    with output_column:
        task1_result = st.session_state.get("task_1_result")
        if task1_result is None:
            _render_empty()
        else:
            _render_result(task1_result)

with static_tab:
    _classification_tab(
        2,
        "Static recognition",
        "Select a 30×38 NPY sample from the repository or upload a compatible file.",
    )

with dynamic_tab:
    _classification_tab(
        3,
        "Dynamic recognition",
        "Select a 30×38×50 NPY sample from the repository or upload a compatible file.",
    )
