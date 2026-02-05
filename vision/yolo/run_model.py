#!/usr/bin/env python3
"""
Run the trained YOLO11 Indoor Obstacle Detection Model.
Wrapper around inference.py for easy execution.
"""

import sys
from pathlib import Path

# Add current directory to path to allow imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from inference import IndoorObstacleDetector, main as inference_main

# Absolute path to the trained model
MODEL_PATH = "/Users/youhanasheriff/Desktop/Work/Mako/r_and_d_projects/models_edge_devices/vision/yolo/outputs/runs/indoor_obstacle_yolo11n_20260202_1911/weights/best.pt"

def run():
    print(f"Loading model from: {MODEL_PATH}")
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print("Error: Model file not found!")
        print("Please ensure the training has completed successfully.")
        sys.exit(1)

    # If arguments are provided, pass them to the original main function
    # but inject the model path if not provided
    if len(sys.argv) > 1:
        if '--model' not in sys.argv:
            sys.argv.extend(['--model', MODEL_PATH])
        inference_main()
    else:
        # Default behavior: Run on webcam with visualization
        print("No arguments provided. Defaulting to webcam (source=0) with visualization.")
        print("Usage: python run_model.py --source <image/video> [--output <file.json>]")
        
        detector = IndoorObstacleDetector(
            model_path=MODEL_PATH,
            device='mps',  # Default to Apple Silicon acceleration
            conf_threshold=0.25
        )
        
        # Run stream
        detector.detect_stream(
            source=0,
            show=True
        )

if __name__ == "__main__":
    run()
