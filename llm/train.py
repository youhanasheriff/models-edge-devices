#!/usr/bin/env python3
"""
Complete LoRA Fine-tuning Script for TinyLlama using MLX
Includes automatic patching for mlx_lm logging bug
"""

import sys
import os
import logging
from pathlib import Path

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def patch_mlx_lm():
    """Patch the mlx_lm.utils module to fix the missing logging import"""
    try:
        import mlx_lm.utils as utils_module
        if not hasattr(utils_module, 'logging'):
            utils_module.logging = logging
            logger.info("✓ Patched mlx_lm.utils with logging module")
        return True
    except ImportError:
        logger.error("mlx_lm not installed. Please run: pip install mlx-lm")
        return False

def main():
    """Run LoRA fine-tuning using MLX"""
    
    # Apply patch first
    if not patch_mlx_lm():
        sys.exit(1)
    
    # Configuration
    model_id = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
    
    # Dataset Configuration
    dataset_name = "firesafex"
    data_dir = f"data/{dataset_name}/"
    adapter_path = f"outputs/mlx/{dataset_name}/adapters"
    
    batch_size = 4
    iters = 200
    learning_rate = 1e-5
    
    print("\n" + "=" * 60)
    print("Starting LoRA Fine-tuning with MLX")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {iters}")
    print(f"Learning rate: {learning_rate}")
    print(f"Adapter output path: {adapter_path}")
    print("=" * 60 + "\n")
    
    # Verify data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory '{data_dir}' not found!")
        sys.exit(1)
    
    # Check for required data files
    required_files = ['train.jsonl', 'valid.jsonl']
    missing_files = []
    for file in required_files:
        if not (data_path / file).exists():
            missing_files.append(file)
            logger.warning(f"{file} not found in {data_dir}")
    
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Create adapter directory if it doesn't exist
    Path(adapter_path).mkdir(parents=True, exist_ok=True)
    
    # Now run the actual training using mlx_lm
    logger.info("Loading MLX LoRA module...")
    
    try:
        from mlx_lm import lora
        from types import SimpleNamespace
        
        # Build configuration
        config = {
            "model": model_id,
            "data": data_dir,
            "train": True,
            "batch_size": batch_size,
            "iters": iters,
            "adapter_path": adapter_path,
            "learning_rate": float(learning_rate),
            "steps_per_report": 10,
            "steps_per_eval": 50,
            "save_every": 50,
            "seed": 42,
            "val_batches": 25,
            "test": False,
            "test_batches": 100,
            "max_seq_length": 2048,
            "grad_checkpoint": False,
            "fine_tune_type": "lora",
            "optimizer": "adam",
            "mask_prompt": False,
            "num_layers": -1,
            "grad_accumulation_steps": 1,
            "resume_adapter_file": None,
            "config": None,
            "report_to": None,
            "project_name": None,
            "lr_schedule": None,  # Learning rate schedule (None for constant LR)
            "optimizer_config": {},  # Additional optimizer configuration
            # LoRA-specific parameters
            "lora_parameters": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "scale": 10.0,
            },
        }
        
        args = SimpleNamespace(**config)
        
        logger.info("Starting training...")
        logger.info("This may take 15-30 minutes depending on your hardware.\n")
        
        # Run training
        lora.run(args)
        
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print(f"✓ Adapters saved to: {adapter_path}")
        print("=" * 60)
        print("\nNext steps:")
        print(f"1. Test your model:")
        print(f"   python -m mlx_lm.generate --model {model_id} \\")
        print(f"     --adapter-path {adapter_path} \\")
        print(f"     --prompt 'Your test prompt here'")
        print(f"\n2. Fuse adapters with base model:")
        print(f"   python -m mlx_lm.fuse --model {model_id} \\")
        print(f"     --adapter-path {adapter_path} \\")
        print(f"     --save-path my-fine-tuned-model")
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
