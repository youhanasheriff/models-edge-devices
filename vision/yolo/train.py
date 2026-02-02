#!/usr/bin/env python3
"""
YOLO11 Training Script for Indoor Obstacle Detection
Part of Strategy 1: Modular Split Architecture for Edge AI

Target Hardware: RDK X5 (10 TOPS BPU) & QCS6490 (Hexagon NPU)
Goal: <20MB model, 30+ FPS inference, mAP50 > 0.70
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR / 'config'
OUTPUT_DIR = SCRIPT_DIR / 'outputs' / 'runs'
DATA_YAML = CONFIG_DIR / 'indoor_obstacle_data.yaml'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train YOLO11 for Indoor Obstacle Detection'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11n.pt',
        choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt'],
        help='YOLO11 model variant (nano recommended for edge)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (adjust based on GPU memory)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device: 0 (GPU), cpu, or mps (Apple Silicon)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='DataLoader workers'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint path'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )
    return parser.parse_args()


def validate_dataset(data_yaml: Path) -> bool:
    """Validate dataset configuration and files exist."""
    import yaml

    if not data_yaml.exists():
        logger.error(f"Data config not found: {data_yaml}")
        return False

    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    data_path = Path(config['path'])

    required_dirs = [
        data_path / config['train'],
        data_path / config['val'],
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            logger.error(f"Missing directory: {dir_path}")
            return False

    # Count images
    train_images = list((data_path / config['train']).glob('*.jpg'))
    val_images = list((data_path / config['val']).glob('*.jpg'))

    logger.info("Dataset validated successfully:")
    logger.info(f"  Training images: {len(train_images)}")
    logger.info(f"  Validation images: {len(val_images)}")
    logger.info(f"  Classes: {config['nc']} ({', '.join(config['names'].values())})")

    return True


def get_training_config(args) -> dict:
    """Build training configuration dictionary."""

    # Generate experiment name if not provided
    if args.name:
        exp_name = args.name
    else:
        model_name = args.model.replace('.pt', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        exp_name = f"indoor_obstacle_{model_name}_{timestamp}"

    config = {
        # Data
        'data': str(DATA_YAML),
        'imgsz': args.imgsz,

        # Training
        'epochs': args.epochs,
        'batch': args.batch,
        'patience': args.patience,
        'workers': args.workers,
        'device': args.device,

        # Optimizer (AdamW for better generalization)
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # Augmentation (balanced for edge deployment)
        'augment': True,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,

        # Learning rate schedule
        'cos_lr': True,
        'close_mosaic': 10,

        # Mixed precision
        'amp': True,

        # Output
        'project': str(OUTPUT_DIR),
        'name': exp_name,
        'exist_ok': True,
        'save': True,
        'save_period': 10,

        # Validation
        'val': True,
        'plots': True,
        'verbose': True,
    }

    return config


def print_banner():
    """Print training banner."""
    print("\n" + "=" * 70)
    print("YOLO11 Indoor Obstacle Detection Training")
    print("Strategy 1: Modular Split Architecture - Layer 1 (Vision Detector)")
    print("=" * 70)
    print("\nTarget Specifications:")
    print("  - Model Size: <20 MB")
    print("  - Inference Speed: 30+ FPS")
    print("  - mAP50 Target: >0.70")
    print("  - Hardware: RDK X5 (10 TOPS) / QCS6490 (12 TOPS)")
    print("=" * 70 + "\n")


def main():
    args = parse_args()
    print_banner()

    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Validate dataset
    logger.info("Validating dataset...")
    if not validate_dataset(DATA_YAML):
        sys.exit(1)

    # Load model
    if args.resume:
        logger.info(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    else:
        logger.info(f"Loading base model: {args.model}")
        model = YOLO(args.model)

    # Get training configuration
    train_config = get_training_config(args)

    # Print configuration
    print("\nTraining Configuration:")
    print("-" * 40)
    for key, value in train_config.items():
        print(f"  {key}: {value}")
    print("-" * 40 + "\n")

    # Start training
    logger.info("Starting training...")
    try:
        results = model.train(**train_config)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # Print results summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Model paths
    save_dir = Path(results.save_dir)
    best_model = save_dir / 'weights' / 'best.pt'
    last_model = save_dir / 'weights' / 'last.pt'

    print(f"\nResults saved to: {save_dir}")
    print(f"Best model: {best_model}")
    print(f"Last model: {last_model}")

    # Check model size
    if best_model.exists():
        size_mb = best_model.stat().st_size / (1024 * 1024)
        print(f"\nModel size: {size_mb:.2f} MB")
        if size_mb <= 20:
            print("  [PASS] Model within 20MB target")
        else:
            print("  [WARN] Model exceeds 20MB target")

    # Print metrics
    print("\nTraining Metrics:")
    print(f"  - mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  - mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  - Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  - Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")

    # Next steps
    print("\n" + "-" * 70)
    print("Next Steps:")
    print(f"  1. Validate: python validate.py --model {best_model}")
    print(f"  2. Export:   python export.py --model {best_model} --format onnx")
    print(f"  3. Test:     python inference.py --model {best_model} --source <image>")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
