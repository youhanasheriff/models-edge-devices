# Fine-Tuning Text Models on MacBook - PyTorch/PEFT

A practical guide to fine-tuning LLMs on Apple Silicon using PyTorch and PEFT.

## Framework Comparison

| Framework | Hardware Support | Best For |
|-----------|-----------------|----------|
| **MLX** (`train.py`) | Apple Silicon only | Native performance, MLX models |
| **PyTorch** (`train_pytorch.py`) | Apple Silicon (MPS) | HuggingFace models, cross-platform |
| **Unsloth** (`train_unsloth.py`) | NVIDIA/AMD/Intel GPUs | 2x faster training on supported GPUs |

## Quick Start (Apple Silicon)

```bash
# Setup
python3 -m venv llm_env
source llm_env/bin/activate
pip install -r requirements.txt

# Train Gemma 3 270M
python train_pytorch.py
```

## Available Models

| Model | Size | Memory |
|-------|------|--------|
| `google/gemma-3-270m-it` | 270M | ~1GB |
| `google/gemma-3-1b-it` | 1B | ~2GB |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | ~2GB |

## Dataset Format

Create `data/train.jsonl`:
```jsonl
{"text": "<system>\nAssistant</system>\n<user>\nQuestion</user>\n<assistant>\nAnswer</assistant>"}
```

## Configuration

Edit `train_pytorch.py`:
- `model_name`: Base model to fine-tune
- `max_steps`: Training iterations (default: 200)
- `lora_r`: LoRA rank (default: 8)
- `batch_size`: Samples per step (default: 1)

## Output

- `outputs/lora_adapters/` - LoRA weights
- `outputs/checkpoint-*/` - Training checkpoints

## Usage After Training

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
model = PeftModel.from_pretrained(base, "outputs/lora_adapters")
tokenizer = AutoTokenizer.from_pretrained("outputs/lora_adapters")
```

## Notes

- **Unsloth requires NVIDIA/AMD/Intel GPUs** - won't work on Apple Silicon
- **MLX is Apple-native** - fastest on Mac but only supports MLX-formatted models
- **PyTorch + MPS** - works on Apple Silicon with any HuggingFace model
