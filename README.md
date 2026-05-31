# Edge AI: Modular Split Architecture for On-Device Reasoning

> **An edge-AI deployment strategy** — pairing lightweight real-time vision detectors with a LoRA-fine-tuned small language model so safety-critical reasoning runs *on-device*, within tight memory, latency, and power budgets.
>
> **Target hardware:** RDK X5 (8GB RAM, 10 TOPS BPU) & Qualcomm QCS6490 (12 TOPS NPU)
> **Target platform:** Pocket AI Guardian (always-on wearable) & Vision AI Standalone

## What Is This?

This project implements a **split modular architecture** for edge AI that separates concerns:

1. **Vision Detection** (Always-On, 30+ FPS) - Lightweight object detectors (RF-DETR, YOLO)
2. **Safety Logic** (Validation) - Classical computer vision rules and geometry checks
3. **Reasoning Engine** (Event-Triggered) - Small language models for contextual understanding
4. **Safety Actions** - Haptic alerts, audio warnings, device controls

This approach **avoids the memory and latency pitfalls of monolithic VLMs** while maintaining real-time safety guarantees.

## Quick Start

```bash
# 1. Install LLM dependencies
cd llm
pip install -r requirements.txt

# 2. Test the reasoning model
python test_mlx.py --model firesafex \
  --prompt "What dangers is a blind user facing right now?"

# 3. Fine-tune for your use case
python train.py --dataset firesafex --iterations 200

# 4. Export for mobile/edge deployment
python export_gguf.py --adapters ./outputs/checkpoint-200/adapters/
# Creates: outputs/gguf/model-q4.gguf (637 MB - ready to deploy)
```

## Project Structure

```
models_edge_devices/
├── CLAUDE.md                  # Claude Code reference guide
├── README.md                  # This file
├── SETUP_SUMMARY.md           # Detailed setup instructions
├── index.html                 # Strategic report (reference)
│
├── llm/                       # Language model / Reasoning engine
│   ├── train.py              # LoRA fine-tuning script
│   ├── export_gguf.py        # Export for deployment
│   ├── test_mlx.py           # Model testing
│   ├── requirements.txt       # Python dependencies
│   ├── data/                 # Training datasets
│   ├── adapters/             # Pre-trained LoRA weights
│   ├── outputs/              # Checkpoints & exports
│   └── docs/                 # Fine-tuning guides
│
├── vision/                    # (TODO) Vision detection models
│   ├── rf_detr/              # RF-DETR detector
│   └── yolo/                 # YOLO detector
│
└── orchestration/             # (TODO) Pipeline coordinator
    └── model_manager.py       # Routes detection → reasoning
```

## Key Architecture Decisions

### Why Split Models, Not Monolithic VLMs?

| Aspect | Split Architecture | Monolithic VLM |
|--------|-------------------|-----------------|
| **Real-Time Safety** | 30+ FPS detection | 5 FPS ❌ Too slow |
| **Memory Usage** | 6GB (fits 8GB hardware) | 5GB+ (causes OOM) ❌ |
| **Power Draw** | 3-5W always-on | 8-15W (overheats) ❌ |
| **Hallucinations** | Prevented via structured JSON | Common ❌ |
| **Flexibility** | Swap detectors/models easily | Monolithic, hard to change |

**Verdict:** Split architecture is mandatory for wearable safety.

## Technologies Used

### LLM Fine-Tuning
- **MLX** - Apple Silicon optimized training
- **LoRA** - Efficient fine-tuning (rank 8, alpha 16)
- **TinyLlama-1.1B** - Base model (upgradeable to SmolLM2-1.7B)
- **llama.cpp** - Multi-platform inference engine

### Vision Detection (To Be Integrated)
- **RF-DETR** - Transformer-based detection (60+ mAP)
- **YOLO11** - Real-time CNN detection (<3ms latency)
- **INT8 Quantization** - Efficient on edge hardware

### Hardware Acceleration
- **RDK X5's BPU** - 10 TOPS brain processing unit
- **QCS6490's Hexagon NPU** - 12 TOPS with 80-150mW power envelope
- **llama.cpp Hexagon Backend** - Snapdragon optimization

## Training & Deployment Pipeline

### Step 1: Fine-Tune Reasoning Model
Adapt the included fire safety dataset to your use case:

```bash
cd llm
# Edit data/firesafex/train.jsonl with your scenarios
python train.py --dataset firesafex --iterations 200
```

### Step 2: Export for Deployment
```bash
python export_gguf.py --adapters ./outputs/checkpoint-200/adapters/
# Creates model-q4.gguf (637 MB) ready for edge devices
```

### Step 3: Integrate Vision Pipeline (TODO)
- Add RF-DETR or YOLO11 detector
- Configure for 30+ FPS on RDK X5's BPU
- Output JSON structured detections

### Step 4: Build Orchestration Daemon (TODO)
- Route detection JSON to reasoning model
- Trigger reasoning only on safety events
- Implement classical safety logic layer

### Step 5: Deploy to Hardware
- Test on RDK X5 (stationary mode)
- Validate on QCS6490 (always-on wearable)
- Verify thermal & power constraints

## Dataset Format

All training data uses JSONL format with conversational structure:

```json
{
  "text": "<|system|>\nYou are a safety advisor for users with visual impairments. Provide accurate, life-saving information about navigation hazards and safety protocols.\n<|user|>\nWhat obstacles are ahead of me based on the camera feed?\n<|assistant|>\nBased on the detected objects, a car is approaching 15 meters ahead on your current trajectory, traveling at ~20 mph from your left. The sidewalk also slopes downward 3 meters ahead. Recommend: stop and move left toward buildings."
}
```

Keep reasoning grounded in detection outputs (JSON) to prevent hallucinations.

## Important Files Reference

- **`CLAUDE.md`** - Comprehensive development guide for Claude Code
- **`SETUP_SUMMARY.md`** - Detailed setup and customization instructions
- **`llm/train.py`** - Main fine-tuning script
- **`llm/export_gguf.py`** - Export pipeline for deployment
- **`llm/test_mlx.py`** - Model validation before deployment
- **`llm/docs/Vision_Model_Fine_Tuning_Guide.md`** - How to extend to vision models

## Next Steps

1. **Customize training data** - Adapt `llm/data/firesafex/` for your safety scenarios
2. **Train reasoning model** - Run `train.py` with your domain-specific data
3. **Export for testing** - Use `export_gguf.py` to create deployable model
4. **Add vision detection** - Integrate RF-DETR or YOLO11
5. **Build orchestration** - Create `orchestration/model_manager.py` for pipeline coordination
6. **Deploy to hardware** - Test on actual RDK X5 / QCS6490 devices

## References

- **RDK X5 Specifications:** https://doc.switch-science.com/media/files/1bc25b40-4713-42e4-826a-f95f3b84a8c6.pdf
- **RF-DETR vs YOLO:** https://medium.com/@aedelon/yolo-is-dead-welcome-rf-detr-the-transformer-that-just-shattered-the-60-map-barrier-e814475d9f8c
- **SmolVLM (Reasoning Model):** https://openreview.net/forum?id=qMUbhGUFUb
- **Vision Transformers on Edge:** https://arxiv.org/html/2503.02891v1
- **Edge AI Deployment Report:** See `index.html` for full strategic analysis
