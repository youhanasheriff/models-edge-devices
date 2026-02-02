#!/usr/bin/env python3
"""
Export Fine-Tuned MLX Model to GGUF for Mobile/Unity

GGUF is the standard format for running LLMs on mobile devices via llama.cpp.
Works with: iOS, Android, Unity (via llama.cpp bindings)

Usage:
    python export_gguf.py --dataset firesafex
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_cmd(cmd, desc, exit_on_fail=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and exit_on_fail:
        print(f"Error: {desc} failed")
        sys.exit(1)
    return result

def main():
    parser = argparse.ArgumentParser(description="Export MLX model to GGUF")
    parser.add_argument("--dataset", "-d", type=str, default="firesafex",
                        help="Dataset name (default: firesafex)")
    parser.add_argument("--model", "-m", type=str, 
                        default="mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit",
                        help="Base model ID")
    parser.add_argument("--skip-fuse", action="store_true",
                        help="Skip fusing step (use existing fused model)")
    args = parser.parse_args()

    # Paths
    adapter_path = f"outputs/mlx/{args.dataset}/adapters"
    fused_path = f"outputs/mlx/{args.dataset}/fused_model"
    gguf_dir = f"outputs/gguf/{args.dataset}"
    gguf_file = f"{gguf_dir}/model-f16.gguf"

    # Step 1: Fuse adapters (if not skipped)
    if not args.skip_fuse:
        if not os.path.exists(adapter_path):
            print(f"Error: Adapter path '{adapter_path}' not found.")
            print("Run training first: python train.py")
            sys.exit(1)
        
        run_cmd(
            f"python -m mlx_lm fuse "
            f"--model {args.model} "
            f"--adapter-path {adapter_path} "
            f"--save-path {fused_path} "
            f"--dequantize",
            "Step 1: Fusing adapters and dequantizing model"
        )

    # Step 2: Convert to GGUF using llama.cpp
    if not os.path.exists(fused_path):
        print(f"Error: Fused model not found at {fused_path}")
        sys.exit(1)

    Path(gguf_dir).mkdir(parents=True, exist_ok=True)
    
    llama_converter = "llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(llama_converter):
        print("\nllama.cpp not found. Cloning...")
        run_cmd("git clone --depth 1 https://github.com/ggerganov/llama.cpp.git", "Cloning llama.cpp")
    
    run_cmd(
        f"python {llama_converter} {fused_path} "
        f"--outfile {gguf_file} "
        f"--outtype f16",
        "Step 2: Converting to GGUF format"
    )

    print(f"\n{'='*60}")
    print("✓ Export Complete!")
    print(f"{'='*60}")
    print(f"\nOutput: {gguf_file}")
    print(f"Size: {os.path.getsize(gguf_file) / 1024**3:.1f} GB")
    print("\n📱 Deployment Options:")
    print("  iOS:     llama.swift or llama.cpp Swift package")
    print("  Android: kotlinx-llama or llama.cpp NDK")
    print("  Unity:   LLMUnity plugin")
    print("\n💡 To quantize for smaller size (requires cmake):")
    print("  cd llama.cpp && cmake -B build && cmake --build build --target llama-quantize")
    print(f"  ./llama.cpp/build/bin/llama-quantize {gguf_file} {gguf_dir}/model-q4.gguf q4_k_m")

if __name__ == "__main__":
    main()
