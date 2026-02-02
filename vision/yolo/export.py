#!/usr/bin/env python3
"""
YOLO11 Model Export Script for Edge Deployment
Exports to ONNX, TensorRT, TFLite, and INT8 quantized formats

Target Hardware:
  - RDK X5 BPU (10 TOPS) - Uses ONNX → Horizon BIN
  - QCS6490 Hexagon NPU (12 TOPS) - Uses ONNX → QNN/SNPE

Requirements: <20MB model size, 30+ FPS inference
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / 'models'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export YOLO11 model for edge deployment'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained .pt model'
    )
    parser.add_argument(
        '--format',
        type=str,
        nargs='+',
        default=['onnx'],
        choices=['onnx', 'torchscript', 'tflite', 'engine', 'openvino'],
        help='Export formats (default: onnx)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='FP16 half precision quantization'
    )
    parser.add_argument(
        '--int8',
        action='store_true',
        help='INT8 quantization (requires calibration data)'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        default=True,
        help='Simplify ONNX model graph'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Enable dynamic batch size'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: vision/yolo/models/)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='indoor_obstacle',
        help='Output model name prefix'
    )
    return parser.parse_args()


def get_model_variant(model_path: str) -> str:
    """Extract model variant from path (nano, small, medium)."""
    path_str = str(model_path).lower()
    if 'yolo11n' in path_str or 'nano' in path_str:
        return 'nano'
    elif 'yolo11s' in path_str or 'small' in path_str:
        return 'small'
    elif 'yolo11m' in path_str or 'medium' in path_str:
        return 'medium'
    return 'custom'


def validate_model(model_path: str) -> bool:
    """Validate model file exists."""
    path = Path(model_path)
    if not path.exists():
        logger.error(f"Model not found: {model_path}")
        return False
    if not path.suffix == '.pt':
        logger.warning(f"Expected .pt file, got: {path.suffix}")
    return True


def export_model(model_path: str, export_format: str, args) -> Path:
    """Export model to specified format."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    # Build export arguments
    export_args = {
        'format': export_format,
        'imgsz': args.imgsz,
    }

    # Format-specific options
    if export_format == 'onnx':
        export_args['simplify'] = args.simplify
        export_args['opset'] = args.opset
        export_args['dynamic'] = args.dynamic

    # Quantization
    if args.half:
        export_args['half'] = True
        logger.info("Exporting with FP16 quantization")

    if args.int8:
        export_args['int8'] = True
        logger.info("Exporting with INT8 quantization")

    # Export
    logger.info(f"Exporting to {export_format.upper()}...")
    exported_path = model.export(**export_args)

    return Path(exported_path)


def check_model_size(model_path: Path, max_size_mb: float = 20.0) -> dict:
    """Check if model meets size requirements."""
    if not model_path.exists():
        return {'exists': False}

    size_bytes = model_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    return {
        'exists': True,
        'size_mb': round(size_mb, 2),
        'size_bytes': size_bytes,
        'meets_target': size_mb <= max_size_mb,
        'target_mb': max_size_mb
    }


def generate_output_name(args, export_format: str) -> str:
    """Generate descriptive output filename."""
    variant = get_model_variant(args.model)
    parts = [args.name, variant]

    if args.int8:
        parts.append('int8')
    elif args.half:
        parts.append('fp16')
    else:
        parts.append('fp32')

    return '_'.join(parts)


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("YOLO11 Model Export for Edge Deployment")
    print("=" * 70)
    print("\nTarget Specifications:")
    print("  - Model Size: <20 MB")
    print("  - Formats: ONNX (universal), TFLite, TensorRT")
    print("  - Hardware: RDK X5, QCS6490, Jetson, Coral")
    print("=" * 70 + "\n")

    # Validate input model
    if not validate_model(args.model):
        return

    model_path = Path(args.model)
    logger.info(f"Input model: {model_path}")

    # Check input model size
    input_size = check_model_size(model_path)
    if input_size['exists']:
        logger.info(f"Input model size: {input_size['size_mb']:.2f} MB")

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export results
    results = []

    for fmt in args.format:
        print(f"\n{'─' * 50}")
        print(f"Exporting to {fmt.upper()}")
        print('─' * 50)

        try:
            # Export
            exported = export_model(str(model_path), fmt, args)

            # Check size
            size_info = check_model_size(exported)

            if not size_info['exists']:
                logger.error(f"Export failed - file not created")
                continue

            # Generate output name and copy to models directory
            output_name = generate_output_name(args, fmt)
            suffix = exported.suffix
            dest_path = output_dir / f"{output_name}{suffix}"

            shutil.copy2(exported, dest_path)

            # Log result
            status = "PASS" if size_info['meets_target'] else "WARN"
            logger.info(f"Exported: {dest_path}")
            logger.info(f"Size: {size_info['size_mb']:.2f} MB [{status}]")

            results.append({
                'format': fmt,
                'path': str(dest_path),
                'size_mb': size_info['size_mb'],
                'meets_target': size_info['meets_target']
            })

        except Exception as e:
            logger.error(f"Export to {fmt} failed: {e}")
            results.append({
                'format': fmt,
                'error': str(e)
            })

    # Summary
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nExported models:")

    for r in results:
        if 'error' in r:
            print(f"  [{r['format'].upper()}] FAILED: {r['error']}")
        else:
            status = "PASS" if r['meets_target'] else "WARN (>20MB)"
            print(f"  [{r['format'].upper()}] {Path(r['path']).name}: {r['size_mb']:.2f} MB [{status}]")

    # Hardware deployment notes
    print("\n" + "-" * 70)
    print("Hardware Deployment Notes:")
    print("-" * 70)
    print("""
  RDK X5 (Horizon BPU):
    - Use ONNX model with Horizon toolchain
    - Convert: hb_mapper makertbin --config config.yaml
    - Expected: 50+ FPS with INT8

  QCS6490 (Hexagon NPU):
    - Use ONNX model with SNPE/QNN SDK
    - Convert: snpe-onnx-to-dlc --input_network <model.onnx>
    - Expected: 40+ FPS at 80-150mW

  NVIDIA Jetson:
    - Use TensorRT engine for best performance
    - Export with --format engine on Jetson device
    - Expected: 100+ FPS

  Google Coral:
    - Use TFLite INT8 with Edge TPU compiler
    - edgetpu_compiler <model.tflite>
    - Expected: 60+ FPS
""")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
