# Contributing

Thanks for your interest in this project. It explores a split edge-AI architecture:
a lightweight vision detector (YOLO / RF-DETR) feeding a LoRA-fine-tuned small
language model for on-device reasoning. See `README.md` for the overview.

## Repository layout

- `llm/` - LoRA fine-tuning, GGUF export, and model testing (`train.py`, `export_gguf.py`, `test_mlx.py`)
- `vision/` - YOLO training, validation, inference, and export (`vision/yolo/`)
- `vlm/` - vision-language datasets and related material
- `orchestration/` - routing of detections to the reasoning model
- `launch_fiftyone.py`, `verify_fiftyone.py` - FiftyOne dataset helpers

## Getting started

Each component has its own dependency file. Install only what you need:

```bash
pip install -r llm/requirements.txt            # LLM fine-tuning
pip install -r vision/requirements.txt         # YOLO
pip install -r vision/yolo_requirements.txt    # YOLO (ultralytics only)
```

These pull in large ML packages (for example `torch`), so use a virtual environment.

## Checks

CI (`.github/workflows/ci.yml`) runs on Python 3.11 and deliberately does not
install the ML dependencies. You can run the same checks locally:

```bash
python -m compileall -q launch_fiftyone.py verify_fiftyone.py llm vision orchestration
pip install ruff
ruff check .
```

The byte-compile step must pass. The `ruff check` step is currently informational
because the existing code has some findings; please avoid adding new ones and do
not mix repository-wide reformatting into functional changes.

## Pull requests

1. Fork and create a branch from `main`.
2. Keep changes focused and describe what you changed and why.
3. Do not commit secrets, `.env` files, `.DS_Store`, or `__pycache__/`.
4. Avoid adding new large binary files (model weights, datasets, training
   outputs) without discussing it first in an issue.
