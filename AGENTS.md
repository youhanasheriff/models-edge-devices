# Agent Instructions for `models_edge_devices`

This document contains instructions for AI agents (and human developers) working on this repository.

## 1. Project Overview & Architecture
This repository contains two distinct components:
1.  **Technical Report (Frontend):** A standalone single-file HTML/JS report (`index.html`) on Edge AI hardware.
2.  **LLM Pipeline (Backend):** Python scripts in `llm/` for fine-tuning and exporting reasoning models (MLX framework).

**Core Philosophy:** Strategy 1 "Modular Split Architecture" - specialized vision models (RF-DETR/YOLO) feed structured JSON to small reasoning models (SmolLM2/Phi-3) on edge devices (RDK X5, QCS6490).

## 2. Environment & Build Commands

### Frontend (`index.html`)
- **Type:** Static HTML with Tailwind/Mermaid (CDN-based).
- **Build:** None. No `package.json`.
- **Run/Preview:**
  ```bash
  python3 -m http.server 8000
  # Open http://localhost:8000
  ```
- **Testing:** Manual visual verification. Check console for JS errors. Verify Mermaid diagrams render.

### Backend (`llm/`)
- **Type:** Python 3 (MLX Framework).
- **Setup:**
  ```bash
  cd llm
  pip install -r requirements.txt
  ```
- **Training (Fine-tune):**
  ```bash
  python train.py --dataset safety_scenarios --epochs 200
  ```
- **Testing (Inference):**
  ```bash
  # Test specific model/prompt
  python test_mlx.py --dataset firesafex --prompt "Identify hazards"
  ```
- **Export (Mobile/Edge):**
  ```bash
  python export_gguf.py --adapters ./outputs/checkpoint-200/
  ```

## 3. Code Style Guidelines

### Python (`llm/*.py`)
- **Style:** Follow PEP 8.
- **Imports:** Standard library first, then third-party (mlx), then local.
- **Typing:** Use type hints for function signatures where helpful.
- **Formatting:** 4-space indentation.
- **CLI:** Use `argparse` for scripts. Provide `--help` descriptions.
- **Paths:** Use relative paths assuming execution from the `llm/` directory, or handle absolute paths robustly.

### HTML/JS (`index.html`)
- **Structure:** Semantic HTML5 (`<section>`, `<article>`).
- **Styling:**
  - Use Tailwind Utility classes (CDN) for layout/spacing.
  - Use Custom CSS (in `<style>`) for complex components (Glassmorphism, Tables).
  - Use CSS Variables (`--primary-blue`) for theming.
- **JavaScript:**
  - ES6+ syntax.
  - No external build steps (no imports/requires).
  - Wrap initialization in `DOMContentLoaded`.
  - **Mermaid:** Initialize manually to support custom interactive controls (zoom/pan).

## 4. Workflows

### Editing the Report
- Modify `index.html` directly.
- Update the Table of Contents (TOC) manually in the `<nav>` section if adding new headers.
- Ensure `<section id="...">` matches the TOC link.

### Working on Models
- **Data:** Place new JSONL datasets in `llm/data/`.
- **Adapters:** Pre-trained weights go in `llm/adapters/`.
- **Validation:** Always run `test_mlx.py` after training to compare base vs. fine-tuned model performance before exporting.

## 5. Deployment & Hardware Constraints
- **Target:** Qualcomm QCS6490 (Hexagon NPU) & RDK X5 (BPU).
- **Memory:** Strict <6GB limit (Total system).
- **Power:** <5W continuous target.
- **Optimization:** ALL models must be quantized (INT8 for vision, Q4 for LLM) before deployment.

## 6. Safety & Security
- **Secrets:** Do not commit API keys or private datasets.
- **Hallucinations:** The architecture relies on *structured inputs* from vision models. Ensure reasoning prompts enforce this constraint (e.g., "Based ONLY on the provided JSON data...").
