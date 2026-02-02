# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements **Strategy 1: The Modular Split Architecture** for Edge AI Deployment targeting:
- **Hardware Targets:** RDK X5 (8GB RAM, 10 TOPS BPU) & Qualcomm QCS6490 (Low-power NPU)
- **Use Cases:** Pocket AI Guardian (always-on wearable) & Vision AI Standalone (stationary analysis)

**Architecture Philosophy:** Run specialized, small vision models (RF-DETR, YOLO) to extract structured data (JSON), fed into a separate small reasoning model (SmolLM2, Phi-3) or logic engine. This guarantees 30+ FPS for safety-critical features and stays within thermal/memory constraints.

**Key Advantage over Monolithic VLMs:** Detection models are tiny (<20MB), run at 30+ FPS, avoid hallucinations through structured data pipelines, and fit within 8GB RAM on edge hardware.

## Project Structure

```
models_edge_devices/
├── index.html              # Strategic report and architecture guide
├── CLAUDE.md              # This file
├── llm/                   # LLM fine-tuning and reasoning engine
│   ├── train.py          # MLX LoRA fine-tuning script
│   ├── export_gguf.py    # Export for mobile deployment
│   ├── test_mlx.py       # Model testing
│   ├── requirements.txt   # Python dependencies
│   ├── data/             # Training datasets
│   ├── adapters/         # Pre-trained LoRA weights
│   ├── outputs/          # Training checkpoints & exports
│   └── docs/             # Fine-tuning guides
├── vlm/                   # Vision-Language Model datasets
│   └── data/
│       └── IndoorObstacle Detection.v9i.yolov11/  # Indoor obstacle dataset
├── vision/               # Vision detection models (Layer 1)
│   ├── requirements.txt  # Vision dependencies (ultralytics, onnx)
│   └── yolo/            # YOLO11 implementation
│       ├── train.py      # Training script for indoor obstacles
│       ├── inference.py  # JSON-structured detection output
│       ├── export.py     # Export to ONNX/INT8 for edge
│       ├── validate.py   # Benchmarking & validation
│       ├── config/
│       │   └── indoor_obstacle_data.yaml  # Dataset config
│       ├── models/       # Exported models (.onnx, .pt)
│       └── outputs/      # Training runs & checkpoints
└── orchestration/        # Pipeline coordination (Layer 2-3 bridge)
    └── detection_to_reasoning.py  # Detection → LLM prompt bridge
```

## Core Architecture: Split Modular Pipeline

### Layer 1: Vision Detection (Always-On)
**Models:** RF-DETR-Nano or YOLO11-Nano
- Input: Camera frame (640x640 or 300x300)
- Output: JSON structured data `{"objects": [{"class": "person", "bbox": [...], "confidence": 0.95}]}`
- Performance: <3ms latency (~30+ FPS)
- Size: <20MB
- Deployment: RDK X5's BPU (Brain Processing Unit) or QCS6490's Hexagon NPU
- Always running for real-time safety monitoring

**Key Optimizations:**
- INT8 quantization via llama.cpp Snapdragon backend
- V4L2 zero-copy pipelines for camera input
- Structured JSON output to prevent reasoning engine hallucinations

### Layer 2: Pose Tracking (Optional)
**Model:** YOLOv8-Pose
- Tracks body keypoints for activity recognition
- Input: Detection bounding box + frame region
- Output: `{"pose": [{"x": 0.5, "y": 0.3, "confidence": 0.9}]}`
- Only triggered when human detected in Layer 1

### Layer 3: Reasoning Engine (Event-Triggered)
**Models:** SmolLM2-1.7B or Phi-3 (Quantized to 4-bit)
- Input: System prompt + detection JSON + optional text prompt
- Output: Human-readable reasoning (e.g., "Obstacle ahead: concrete step")
- Performance: Activated only when safety event detected (<600ms acceptable)
- Size: 637MB (Q4 quantized) on QCS6490, larger model on RDK X5
- Prevents hallucinations: Model only reasons about detected objects (no null object generation)

**LoRA Fine-Tuning:** Adapt reasoning model to specific safety domains (fall detection, collision awareness, OCR interpretation)

### Layer 4: Logic Engine (Safety Rules)
**Classical CV + Geometry Validation:**
- Physics validation: Is the detected "ground" actually flat?
- Trajectory analysis: Is the obstacle moving toward the user?
- Counting logic: Correct package counts via geometry, not VLM guessing
- Activates safety protocols (vibration, audio alerts)

## Development Commands

### LLM Component Setup
```bash
# Install dependencies
cd llm/
pip install -r requirements.txt

# Test a pre-trained model
python test_mlx.py --model firesafex --prompt "Describe what's in this image"

# Fine-tune on custom dataset
python train.py --dataset custom_safety --epochs 200

# Export for mobile/edge deployment
python export_gguf.py --adapters ./adapters/custom_safety/
# Outputs: ./outputs/gguf/model-q4.gguf (637MB - mobile optimized)
```

### Vision Detection (YOLO11 Indoor Obstacles)
```bash
# Install dependencies
cd vision/
pip install -r requirements.txt

# Train YOLO11-Nano on indoor obstacle dataset
python yolo/train.py --model yolo11n.pt --epochs 100 --batch 16 --device mps

# Validate trained model (accuracy + speed benchmark)
python yolo/validate.py --model yolo/outputs/runs/*/weights/best.pt --benchmark

# Export to ONNX for edge deployment
python yolo/export.py --model yolo/outputs/runs/*/weights/best.pt --format onnx --simplify

# Export with INT8 quantization (smallest, fastest)
python yolo/export.py --model yolo/outputs/runs/*/weights/best.pt --format onnx --int8

# Test inference with JSON output
python yolo/inference.py --model yolo/models/indoor_obstacle_nano_fp32.onnx --source <image.jpg>

# Live webcam detection
python yolo/inference.py --model yolo/models/indoor_obstacle_nano_fp32.onnx --source 0 --show
```

### Hardware Deployment
```bash
# Deploy to RDK X5 (10 TOPS BPU)
# Deploy to Qualcomm QCS6490 (12 TOPS NPU, 80-150mW always-on)
# Uses llama.cpp Hexagon backend for Snapdragon optimization
```

## Key Technologies

### LLM Training (MLX Framework)
- **Framework:** MLX (Apple Silicon native, efficient on M-series)
- **Fine-tuning:** LoRA (Low-Rank Adaptation) - rank=8, alpha=16
- **Model:** TinyLlama-1.1B-Chat (base), upgradeable to SmolLM2-1.7B or Phi-3
- **Quantization:** 4-bit for deployment (<1GB memory)
- **Inference Engine:** llama.cpp with Snapdragon Hexagon backend

### Vision Detection (YOLO11 Implementation)
- **Model:** YOLO11-Nano (2.6M params, ~6.5MB .pt, ~12MB .onnx)
- **Dataset:** Indoor Obstacle Detection v9 (1,602 train / 57 val images)
- **Classes:** 8 (`closed_door`, `door`, `elevator`, `escalator`, `footpath`, `obstacle`, `person`, `wall`)
- **Quantization:** FP32, FP16, INT8 via ONNX export
- **Output:** JSON structured data with safety levels and spatial descriptions
- **Performance Target:** 30+ FPS, <20MB model size, mAP50 >0.70

### Hardware Acceleration
- **RDK X5:** 10 TOPS BPU (Brain Processing Unit) - optimized for vision
- **QCS6490:** 12 TOPS Hexagon NPU + Adreno 643 GPU
- **llama.cpp Backend:** Uses Snapdragon Hexagon HTP (Hexagon Tensor Processor)
- **Inference:** Batch processing on dedicated accelerators, minimal CPU usage

## Dataset Management

**Location:** `llm/data/`

**Format:** JSONL with system prompt, user query, assistant response
```json
{
  "text": "<|system|>\nYou are a safety advisor for blind users...\n<|user|>\nWhat does the ground ahead look like?\n<|assistant|>\nBased on the visual input, the ground is smooth concrete sloping downward 2 meters ahead..."
}
```

**Datasets Included:**
- `firesafex/` - Fire safety domain (226 Q&A pairs) - *adapt for edge safety scenarios*
- `data_analyst/` - Alternative training dataset

**Key Principle:** Keep reasoning data aligned with detection outputs to prevent hallucinations.

## Testing & Validation

### LLM Model Testing
```bash
python llm/test_mlx.py --model firesafex --question "What safety hazards are present?"
```

Compares:
- Base model output (often hallucinating details)
- Fine-tuned model output (safety-focused, grounded in detection data)

### Hardware Validation Checklist
- [ ] Detection model: 30+ FPS on target hardware
- [ ] Reasoning model: <600ms latency on event trigger
- [ ] Memory: <6GB combined footprint (leaving 2GB for OS/system)
- [ ] Thermal: Passive cooling requirement met (<5W continuous on QCS6490)
- [ ] Structured output: JSON validation of detector outputs
- [ ] Hallucination test: Reasoning model behavior with missing objects

## Deployment Pipeline

### Step 1: Fine-Tune Reasoning Model
```bash
cd llm/
python train.py --dataset safety_scenarios --iterations 200
# Creates: ./outputs/checkpoint-200/
```

### Step 2: Export to GGUF (Mobile/Edge Format)
```bash
python export_gguf.py --adapters ./outputs/checkpoint-200/
# Creates: ./outputs/gguf/model-q4.gguf (637MB)
```

### Step 3: Integrate with Vision Pipeline
- Link detection model outputs (JSON) to reasoning model inputs
- Implement orchestration daemon (conditional reasoning trigger)
- Add classical logic engine (geometry validation, safety rules)

### Step 4: Deploy to Hardware
- **RDK X5:** Use Hexagon backend, 10 TOPS BPU for vision, CPU for reasoning
- **QCS6490:** Use Hexagon NPU (80-150mW always-on), keep reasoning model in standby

## Architecture Decision: Why Strategy 1

### Comparison: Strategy 1 vs Strategy 2 (Monolithic VLM)

| Metric | Strategy 1 (Split) | Strategy 2 (VLM) |
|--------|-------------------|-----------------|
| **FPS (Safety)** | 30+ FPS | 5 FPS ❌ |
| **Memory** | <6GB (fits 8GB) | 5GB+ (crashes) ❌ |
| **Power** | 3-5W continuous | 8-15W (overheat) ❌ |
| **Hallucinations** | Prevented (structured data) | Common ❌ |
| **Complexity** | Moderate (orchestration) | Low but fragile |

**Verdict:** Strategy 1 is mandatory for always-on wearable safety and fits hardware constraints.

## Important Files & Their Roles

### LLM / Reasoning Engine (Layer 3)
- **`llm/train.py`** - MLX LoRA fine-tuning script for safety reasoning
- **`llm/export_gguf.py`** - Export to GGUF for mobile deployment
- **`llm/test_mlx.py`** - Model comparison and validation
- **`llm/data/`** - Training datasets (firesafex, data_analyst)
- **`llm/adapters/`** - Pre-trained LoRA weights

### Vision Detection (Layer 1)
- **`vision/yolo/train.py`** - YOLO11 training for indoor obstacles
- **`vision/yolo/inference.py`** - JSON-structured detection output (critical for pipeline)
- **`vision/yolo/export.py`** - Export to ONNX/INT8 for edge hardware
- **`vision/yolo/validate.py`** - Accuracy and speed benchmarking
- **`vision/yolo/config/indoor_obstacle_data.yaml`** - Dataset configuration

### Orchestration (Layer 2 Bridge)
- **`orchestration/detection_to_reasoning.py`** - Converts detection JSON to LLM prompts

### Dataset
- **`vlm/data/IndoorObstacle Detection.v9i.yolov11/`** - 8-class indoor obstacle dataset (Roboflow)

## JSON Output Format (Detection → Reasoning)

The vision layer outputs JSON that the orchestration bridge converts to LLM prompts:

```json
{
    "timestamp": "2026-01-22T10:30:00.123Z",
    "frame_id": 100,
    "inference_ms": 18.5,
    "fps": {"current": 54.1, "average": 52.3},
    "detections": [
        {
            "class": "obstacle",
            "confidence": 0.85,
            "bbox": {"x1": 50, "y1": 450, "x2": 150, "y2": 600},
            "center": {"x": 100, "y": 525},
            "area_ratio": 0.037,
            "safety_level": "high"
        }
    ],
    "summary": {
        "total_objects": 1,
        "high_priority": ["obstacle"],
        "scene_complexity": "simple"
    }
}
```

## Next Steps

1. **Train YOLO11:** Run `python vision/yolo/train.py --epochs 100` on the indoor obstacle dataset
2. **Validate Model:** Check mAP50 >0.70 and 30+ FPS with `python vision/yolo/validate.py --benchmark`
3. **Export for Edge:** Create ONNX/INT8 models with `python vision/yolo/export.py`
4. **Test End-to-End:** Run `python orchestration/detection_to_reasoning.py` to verify the pipeline
5. **Deploy to RDK X5:** Test on actual hardware with Hexagon backend
6. **Validate on QCS6490:** Verify 80-150mW power envelope for always-on operation
7. **Fine-tune Reasoning:** Customize LLM for indoor navigation safety scenarios

## References

- RDK X5 Review: https://medium.com/@zlodeibaal/rdk-x5-review-ai-board-for-robotics-017454547bc8
- RF-DETR vs YOLO: https://medium.com/@aedelon/yolo-is-dead-welcome-rf-detr-the-transformer-that-just-shattered-the-60-map-barrier-e814475d9f8c
- SmolVLM: https://openreview.net/forum?id=qMUbhGUFUb
- Vision Transformers on Edge: https://arxiv.org/html/2503.02891v1
