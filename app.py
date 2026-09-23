# -*- coding: utf-8 -*-
"""Browser interface for the three PI-MFS inference tasks.

Run locally with:
    python app.py
"""

from __future__ import annotations

import html
import os
from pathlib import Path
import threading

import gradio as gr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from inference_engine import InferenceResult, MambaInferenceEngine


APP_TITLE = "PI-MFS Framework"
REPOSITORY_URL = os.getenv("GITHUB_REPOSITORY_URL", "").strip()
ENGINE = MambaInferenceEngine(device_preference=os.getenv("PI_MFS_DEVICE", "auto"))
INFERENCE_LOCK = threading.RLock()

POSITION_OPTIONS = {
    "Automatic position recognition": None,
    "Left top": 0,
    "Left bottom": 1,
    "Center": 2,
    "Right top": 3,
    "Right bottom": 4,
}


CSS = """
.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
    background: #f4f7fb !important;
    color: #172033 !important;
}
.pi-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    margin: 8px 0 16px;
    padding: 22px 26px;
    border-radius: 16px;
    color: white;
    background: #07162d;
}
.pi-header h1 {
    margin: 0 0 4px;
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 32px;
}
.pi-header p { margin: 0; color: #d4e1ee; }
.pi-header a {
    padding: 10px 14px;
    border: 1px solid rgba(255,255,255,.22);
    border-radius: 9px;
    color: white !important;
    text-decoration: none;
    white-space: nowrap;
}
.pi-card {
    border: 1px solid #dce5ee !important;
    border-radius: 14px !important;
    background: white !important;
    box-shadow: 0 16px 36px rgba(20,43,72,.07) !important;
}
.pi-input { padding: 6px !important; }
.pi-output { padding: 6px !important; }
.result-summary {
    min-height: 112px;
    padding: 16px 18px;
    border: 1px solid #dce5ee;
    border-radius: 12px;
    background: #f8fafc;
}
.result-summary h2 {
    margin: 0 0 8px;
    color: #075f6c;
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 38px;
    line-height: 1.05;
}
.result-summary p { margin: 0; color: #53657a; }
.task-note { color: #607086; font-size: 14px; line-height: 1.55; }
footer { display: none !important; }
@media (max-width: 720px) {
    .pi-header { align-items: flex-start; flex-direction: column; }
    .pi-header h1 { font-size: 27px; }
}
"""


def _path_from_upload(value: str | Path | None, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise gr.Error(f"Please select {field_name}.")
    path = Path(str(value))
    if not path.is_file():
        raise gr.Error(f"The selected {field_name} is unavailable.")
    return str(path)


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
    response_titles = {
        1: "Task 1 · Mean Strain Map",
        2: "Task 2 · Static Response Map",
        3: "Task 3 · Temporal Maximum Projection",
    }
    response_axes.set_title(response_titles[result.task_id])
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
    axes.plot(indices, values, color="#168B9A", linewidth=1.0, label="Per-row prediction")
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


def _result_summary(result: InferenceResult) -> str:
    confidence_line = ""
    if result.confidence is not None:
        confidence_line = f"<br><b>Confidence:</b> {result.confidence * 100:.1f}%"
    return (
        '<div class="result-summary">'
        f"<h2>{html.escape(result.primary_text)}</h2>"
        f"<p>{html.escape(result.secondary_text)}{confidence_line}</p>"
        "</div>"
    )


def _result_table(result: InferenceResult) -> list[list[str]]:
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


def _pack_result(result: InferenceResult):
    return (
        _result_summary(result),
        _build_figure(result),
        _result_table(result),
        f"**Input summary:** {result.source_description}  \n**Device:** {ENGINE.device_name}",
    )


def run_pressure(model_file, input_file, max_rows, position_name):
    model_path = _path_from_upload(model_file, "the Task 1 model checkpoint")
    input_path = _path_from_upload(input_file, "the raw strain input")
    rows = None if max_rows in (None, 0) else int(max_rows)
    manual_position = POSITION_OPTIONS.get(position_name)
    try:
        with INFERENCE_LOCK:
            result = ENGINE.predict_task1(
                model_path,
                input_path,
                max_rows=rows,
                manual_position_id=manual_position,
            )
    except Exception as exc:
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    return _pack_result(result)


def run_static(model_file, input_file):
    model_path = _path_from_upload(model_file, "the Task 2 model checkpoint")
    input_path = _path_from_upload(input_file, "the static NPY sample")
    try:
        with INFERENCE_LOCK:
            result = ENGINE.predict_task2(model_path, input_path)
    except Exception as exc:
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    return _pack_result(result)


def run_dynamic(model_file, input_file):
    model_path = _path_from_upload(model_file, "the Task 3 model checkpoint")
    input_path = _path_from_upload(input_file, "the dynamic NPY sample")
    try:
        with INFERENCE_LOCK:
            result = ENGINE.predict_task3(model_path, input_path)
    except Exception as exc:
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    return _pack_result(result)


def _header_html() -> str:
    repository = (
        f'<a href="{html.escape(REPOSITORY_URL, quote=True)}" target="_blank" '
        'rel="noreferrer">View repository</a>'
        if REPOSITORY_URL
        else ""
    )
    return f"""
    <header class="pi-header">
      <div>
        <h1>{APP_TITLE}</h1>
        <p>Physical-Information-Driven Framework for Multidimensional Fiber Sensing</p>
      </div>
      {repository}
    </header>
    """


def _result_components():
    result = gr.HTML(
        '<div class="result-summary"><h2>—</h2><p>Select a model checkpoint and an input file.</p></div>',
        label="Recognition Result",
    )
    plot = gr.Plot(label="Analysis")
    table = gr.Dataframe(
        headers=["Metric / Class", "Value / Probability"],
        datatype=["str", "str"],
        interactive=False,
        label="Metrics",
    )
    source = gr.Markdown("Ready", elem_classes=["task-note"])
    return result, plot, table, source


with gr.Blocks(title=APP_TITLE, css=CSS) as demo:
    gr.HTML(_header_html())

    with gr.Tabs():
        with gr.Tab("Task 1 · Pressure"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=340, elem_classes=["pi-card", "pi-input"]):
                    gr.Markdown("### Model and input")
                    task1_model = gr.File(
                        label="Pressure model",
                        file_types=[".pth", ".pt"],
                        type="filepath",
                    )
                    task1_input = gr.File(
                        label="Raw strain input",
                        file_types=[".tsv", ".txt", ".npy"],
                        type="filepath",
                    )
                    task1_rows = gr.Number(
                        label="Inference rows (0 = all)",
                        value=256,
                        minimum=0,
                        precision=0,
                    )
                    task1_position = gr.Dropdown(
                        label="Force position",
                        choices=list(POSITION_OPTIONS),
                        value="Automatic position recognition",
                    )
                    task1_run = gr.Button("Run Pressure Prediction", variant="primary")
                    gr.Markdown(
                        "The raw OFDR TSV is reconstructed into 30×38 matrices using parameters stored in the checkpoint.",
                        elem_classes=["task-note"],
                    )
                with gr.Column(scale=8, min_width=520, elem_classes=["pi-card", "pi-output"]):
                    task1_outputs = _result_components()

            task1_run.click(
                fn=run_pressure,
                inputs=[task1_model, task1_input, task1_rows, task1_position],
                outputs=list(task1_outputs),
                api_name="predict_pressure",
            )

        with gr.Tab("Task 2 · Static"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=340, elem_classes=["pi-card", "pi-input"]):
                    gr.Markdown("### Model and input")
                    task2_model = gr.File(
                        label="Static model",
                        file_types=[".pth", ".pt"],
                        type="filepath",
                    )
                    task2_input = gr.File(
                        label="2D sample",
                        file_types=[".npy"],
                        type="filepath",
                    )
                    task2_run = gr.Button("Run Static Recognition", variant="primary")
                    gr.Markdown(
                        "The webpage accepts the same normalized 30×38 NPY sample format as the desktop program.",
                        elem_classes=["task-note"],
                    )
                with gr.Column(scale=8, min_width=520, elem_classes=["pi-card", "pi-output"]):
                    task2_outputs = _result_components()

            task2_run.click(
                fn=run_static,
                inputs=[task2_model, task2_input],
                outputs=list(task2_outputs),
                api_name="predict_static",
            )

        with gr.Tab("Task 3 · Dynamic"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=340, elem_classes=["pi-card", "pi-input"]):
                    gr.Markdown("### Model and input")
                    task3_model = gr.File(
                        label="Dynamic model",
                        file_types=[".pth", ".pt"],
                        type="filepath",
                    )
                    task3_input = gr.File(
                        label="3D sample",
                        file_types=[".npy"],
                        type="filepath",
                    )
                    task3_run = gr.Button("Run Dynamic Recognition", variant="primary")
                    gr.Markdown(
                        "The webpage accepts the original 30×38×50 dynamic sample without rebuilding the database.",
                        elem_classes=["task-note"],
                    )
                with gr.Column(scale=8, min_width=520, elem_classes=["pi-card", "pi-output"]):
                    task3_outputs = _result_components()

            task3_run.click(
                fn=run_dynamic,
                inputs=[task3_model, task3_input],
                outputs=list(task3_outputs),
                api_name="predict_dynamic",
            )


demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
