#!/usr/bin/env python3
"""
Test Script for MLX TinyLlama (Original vs Fine-Tuned)

Usage:
    python test_mlx.py                          # Use default prompt
    python test_mlx.py --prompt "Your question" # Custom prompt
"""

import os
import argparse
from mlx_lm import load, generate

DEFAULT_QUESTION = "What are the essential steps to take if a fire starts in a residential building?"

def build_prompt(system_prompt, user_question):
    # ADD YOUR CHAT TEMPLATE HERE
    # Example: return f"<|system|>\n...{question}...<|assistant|>\n"
    return f"""<|system|>{system_prompt}
<|user|>{user_question}
<|assistant|>
"""

def main():
    parser = argparse.ArgumentParser(description="Compare Original vs Fine-Tuned MLX Model")
    parser.add_argument("--system_prompt", "-s", type=str, default=None,
                        help="Custom system prompt")
    parser.add_argument("--prompt", "-p", type=str, default=None,
                        help="Custom question to ask the model")
    parser.add_argument("--dataset", "-d", type=str, default="firesafex",
                        help="Dataset name for adapter path (default: firesafex)")
    args = parser.parse_args()

    # Configuration
    model_id = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
    adapter_path = f"outputs/mlx/{args.dataset}/adapters"
    
    # Build prompt
    system_prompt = args.system_prompt if args.system_prompt else "You are a helpful assistant."
    user_question = args.prompt if args.prompt else DEFAULT_QUESTION
    prompt = build_prompt(system_prompt, user_question)
    
    print("=" * 60)
    print("Comparing Original vs Fine-Tuned MLX Model")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Adapters: {adapter_path}")
    print(f"System Prompt: {system_prompt}")
    print(f"Question: {user_question}")
    print("=" * 60)

    # 1. Test Original Model
    print("\nLoading Original Model...")
    try:
        model, tokenizer = load(model_id)
        
        print("\n--- Original Model Response ---")
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=200,
            verbose=True 
        )
    except Exception as e:
        print(f"Error with original model: {e}")
        return

    # 2. Test Fine-Tuned Model
    print("\n" + "-" * 60)
    print("Loading Fine-Tuned Model (with adapters)...")

    if not os.path.exists(adapter_path):
        print(f"Warning: Adapter path '{adapter_path}' does not exist.")
        print("Run 'python train.py' first to generate adapters.")
        return

    try:
        model_ft, tokenizer_ft = load(model_id, adapter_path=adapter_path)
        
        print("\n--- Fine-Tuned Model Response ---")
        response_ft = generate(
            model_ft,
            tokenizer_ft,
            prompt=prompt,
            max_tokens=200,
            verbose=True
        )
    except Exception as e:
        print(f"Error with fine-tuned model: {e}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
