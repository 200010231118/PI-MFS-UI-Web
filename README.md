# PI-MFS Web Interface

This project converts the original PySide6 desktop interface into a browser
interface while keeping the same three real inference paths.

## Preserved inputs

- Task 1: `.pth`/`.pt` checkpoint, `.tsv`/`.txt`/`.npy` strain input,
  inference-row limit, and optional force-position selection
- Task 2: `.pth`/`.pt` checkpoint and a `30×38` static `.npy` sample
- Task 3: `.pth`/`.pt` checkpoint and a `30×38×50` dynamic `.npy` sample

The webpage calls the existing `MambaInferenceEngine`; it does not substitute
mock results. Heatmaps, regression diagnostics, confidence values, and Top-3
predictions are generated from the actual inference result.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860`.

## Configuration

- `PI_MFS_DEVICE=auto|cpu|cuda` chooses the inference device.
- `PORT=7860` changes the web-server port.
- `GITHUB_REPOSITORY_URL=https://github.com/...` adds a repository button to
  the page header.

## Deployment note

This is a Python/PyTorch web service, so it must be hosted on a service that
runs Python (for example a GPU server or a suitable Hugging Face Space). A
static GitHub Pages site cannot execute the three `.pth` checkpoints.

Only expose checkpoint upload to trusted reviewers. PyTorch checkpoints should
come from a trusted repository controlled by the project authors.
