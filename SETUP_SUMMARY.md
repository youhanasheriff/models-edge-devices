# Strategy 1 Setup Summary

This document summarizes the files copied from the AI Models project and how to get started with the Modular Split Architecture implementation.

## Files Copied

### LLM Fine-Tuning Infrastructure (`llm/`)

1. **`train.py`** (154 lines)
   - MLX-based LoRA fine-tuning script
   - Fine-tunes TinyLlama-1.1B on custom datasets
   - Output: LoRA adapters for reasoning engine

2. **`export_gguf.py`** (96 lines)
   - Fuses LoRA adapters with base model
   - Exports to GGUF format (optimized for llama.cpp)
   - Produces both F16 (2GB) and Q4 (637MB) quantized versions
   - The Q4 version is what gets deployed to edge devices

3. **`test_mlx.py`** (97 lines)
   - Tests and validates model outputs
   - Compares base model vs fine-tuned model behavior
   - Verifies reasoning quality before deployment

4. **`requirements.txt`**
   - All Python dependencies for MLX fine-tuning
   - Install with: `pip install -r llm/requirements.txt`

### Training Data (`llm/data/`)

- **`firesafex/`** - Fire safety domain dataset (226 Q&A pairs)
  - Good starting point; adapt for edge safety scenarios
  - Format: JSONL with system prompt + user question + assistant response
  
- **`data_analyst/`** - Alternative dataset for testing

### Pre-Trained Adapters (`llm/adapters/`)

- LoRA weights from previous training
- Can be used as-is or fine-tuned further
- Configuration: rank=8, alpha=16 (efficient for edge)

### Documentation (`llm/docs/`)

- **SETUP_SUMMARY.md** - Previous project setup guide
- **FireSafeX_Project_Report.md** - Complete fine-tuning workflow
- **Fine_Tuning_Text_Models_on_MacBook_M4_Air.md** - M4 optimization tips
- **Vision_Model_Fine_Tuning_Guide.md** - How to adapt this for vision models
- Reference materials for understanding the full pipeline

## Quick Start

### 1. Install Dependencies
```bash
cd llm
pip install -r requirements.txt
```

### 2. Test with Existing Model
```bash
python test_mlx.py \
  --model firesafex \
  --prompt "What safety hazards could affect a blind user?"
```

### 3. Fine-Tune for Your Use Case
```bash
python train.py \
  --dataset firesafex \
  --iterations 200 \
  --batch_size 4 \
  --learning_rate 1e-5
```

### 4. Export for Edge Deployment
```bash
python export_gguf.py \
  --adapters ./outputs/checkpoint-200/adapters/
```

This creates:
- `outputs/gguf/model-f16.gguf` (2.0 GB - full precision)
- `outputs/gguf/model-q4.gguf` (637 MB - quantized for mobile/edge)

## Architecture Reminder: Split Modular Pipeline

```
Camera Frame
    ↓
┌─────────────────────────┐
│ Layer 1: Vision Detector │  ← RF-DETR or YOLO11-Nano
│ (Always-On, 30+ FPS)    │     Input: 640x640 frame
│ RDK X5's BPU            │     Output: JSON detections
└──────────────┬──────────┘
               ↓
         [JSON Objects]
               ↓
┌─────────────────────────┐
│ Layer 2: Safety Logic   │  ← Classical CV + Geometry
│ (Rules & Validation)    │     Prevents hallucinations
└──────────────┬──────────┘
               ↓
         [Validated Events]
               ↓
┌─────────────────────────┐
│ Layer 3: Reasoning Eng. │  ← SmolLM2 or Phi-3 (LoRA)
│ (Event-Triggered)       │     Input: Detection + prompt
│ QCS6490's Hexagon NPU   │     Output: Safety warnings
└──────────────┬──────────┘
               ↓
         [Safety Actions]
               ↓
        Vibration Alert / Audio / Haptic
```

## Key Optimizations for Edge

### Memory Efficiency
- TinyLlama base model: 1.1B parameters (fits in 4GB)
- LoRA adapters: ~25MB additional
- Q4 quantized export: 637MB total
- **Result:** Leaves ~6GB free on RDK X5 (8GB total) for detection models and OS

### Latency
- Detection models: <3ms per frame (30+ FPS)
- Reasoning triggered only on safety events: <600ms acceptable
- Always-on power consumption: ~3-5W on QCS6490 (passive cooling viable)

### Hardware Acceleration
- Uses llama.cpp with Snapdragon Hexagon backend
- Optimized for RDK X5's BPU (10 TOPS) and QCS6490's NPU (12 TOPS)
- INT8 quantization support for vision models
- V4L2 zero-copy pipelines for camera input

## Next Steps for Full Implementation

### 1. Add Vision Detection Layer (TODO)
- Integrate RF-DETR-Nano or YOLO11-Nano
- Set target: 30+ FPS on RDK X5
- Output: JSON structured data

### 2. Build Orchestration Daemon (TODO)
- Create `orchestration/model_manager.py`
- Route detection outputs to reasoning model
- Conditional activation (only run reasoning on events)

### 3. Implement Safety Logic Engine (TODO)
- Physics validation: Is detected "ground" actually safe?
- Geometry checks: Trajectory, distance calculations
- Counting logic: Correct package counts via geometry, not guessing

### 4. Deploy to Hardware
- Test on RDK X5 with Hexagon backend
- Verify 80-150mW power envelope on QCS6490
- Validate thermal characteristics (passive cooling)

## Important: Don't Repeat the HTML

The old `index.html` (strategic report) is there for reference only. **Focus on building the actual implementation** in:
- `llm/` - Reasoning engine
- `vision/` - Detection models (to be created)
- `orchestration/` - Pipeline coordinator (to be created)

## Customization: Adapt for Your Domain

The included datasets focus on fire safety. To customize for edge safety scenarios (falling, collisions, navigation):

1. **Modify `llm/data/firesafex/train.jsonl`** with your scenarios
2. **Update system prompt** in training data to match your use case
3. **Re-run `train.py`** with your custom dataset
4. **Export with `export_gguf.py`** for deployment

Keep the data format consistent (JSONL with system/user/assistant roles).

## References for Further Learning

- **MLX Framework:** https://github.com/ml-explore/mlx
- **llama.cpp Snapdragon:** https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-hexagon
- **RDK X5:** https://doc.switch-science.com/media/files/1bc25b40-4713-42e4-826a-f95f3b84a8c6.pdf
- **RF-DETR Detection:** https://medium.com/@aedelon/yolo-is-dead-welcome-rf-detr-the-transformer-that-just-shattered-the-60-map-barrier-e814475d9f8c
