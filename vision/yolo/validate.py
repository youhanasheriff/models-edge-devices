#!/usr/bin/env python3
"""
YOLO11 Validation and Benchmarking Script

Validates model performance against edge deployment requirements:
  - mAP50 on validation set (target: >0.70)
  - Inference speed (target: 30+ FPS / <33ms)
  - Model size (target: <20MB)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR / 'config'
DATA_YAML = CONFIG_DIR / 'indoor_obstacle_data.yaml'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate YOLO11 model for edge deployment'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model (.pt or .onnx)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=str(DATA_YAML),
        help='Data YAML configuration'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cpu, 0, mps)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=1,
        help='Batch size for validation'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run speed benchmark'
    )
    parser.add_argument(
        '--benchmark-iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    return parser.parse_args()


def check_model_size(model_path: str, max_size_mb: float = 20.0) -> dict:
    """Check if model meets size requirements."""
    path = Path(model_path)
    if not path.exists():
        return {'exists': False, 'error': 'File not found'}

    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    return {
        'exists': True,
        'path': str(path),
        'size_bytes': size_bytes,
        'size_mb': round(size_mb, 2),
        'target_mb': max_size_mb,
        'meets_target': size_mb <= max_size_mb,
        'status': 'PASS' if size_mb <= max_size_mb else 'FAIL'
    }


def validate_accuracy(model_path: str, data_yaml: str, device: str, imgsz: int, batch: int) -> dict:
    """Run validation and get accuracy metrics."""
    from ultralytics import YOLO

    logger.info("Running accuracy validation...")
    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        device=device,
        imgsz=imgsz,
        batch=batch,
        verbose=True
    )

    # Extract metrics
    results = {
        'mAP50': round(float(metrics.box.map50), 4),
        'mAP50_95': round(float(metrics.box.map), 4),
        'precision': round(float(metrics.box.p.mean()), 4),
        'recall': round(float(metrics.box.r.mean()), 4),
        'f1_score': round(2 * (float(metrics.box.p.mean()) * float(metrics.box.r.mean())) /
                         (float(metrics.box.p.mean()) + float(metrics.box.r.mean()) + 1e-6), 4),
    }

    # Per-class AP50
    class_names = list(metrics.names.values())
    results['per_class_ap50'] = {
        name: round(float(ap), 4)
        for name, ap in zip(class_names, metrics.box.ap50)
    }

    # Target check
    results['target_mAP50'] = 0.70
    results['meets_target'] = results['mAP50'] >= 0.70
    results['status'] = 'PASS' if results['meets_target'] else 'FAIL'

    return results


def benchmark_speed(
    model_path: str,
    device: str,
    imgsz: int,
    iterations: int = 100,
    warmup: int = 10
) -> dict:
    """Benchmark inference speed."""
    from ultralytics import YOLO

    logger.info(f"Running speed benchmark ({iterations} iterations)...")
    model = YOLO(model_path)

    # Create dummy input (simulates real inference)
    dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    logger.info(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        model(dummy_input, device=device, verbose=False)

    # Benchmark
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        model(dummy_input, device=device, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i + 1}/{iterations}")

    # Calculate statistics
    times = np.array(times)
    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    min_ms = float(np.min(times))
    max_ms = float(np.max(times))
    p50_ms = float(np.percentile(times, 50))
    p95_ms = float(np.percentile(times, 95))
    p99_ms = float(np.percentile(times, 99))

    fps = 1000 / avg_ms

    results = {
        'iterations': iterations,
        'device': device,
        'image_size': imgsz,
        'latency_ms': {
            'mean': round(avg_ms, 2),
            'std': round(std_ms, 2),
            'min': round(min_ms, 2),
            'max': round(max_ms, 2),
            'p50': round(p50_ms, 2),
            'p95': round(p95_ms, 2),
            'p99': round(p99_ms, 2)
        },
        'fps': round(fps, 1),
        'target_fps': 30,
        'target_ms': 33.3,
        'meets_target': fps >= 30,
        'status': 'PASS' if fps >= 30 else 'FAIL'
    }

    return results


def print_report(size_results: dict, accuracy_results: dict, speed_results: dict = None):
    """Print formatted validation report."""
    print("\n" + "=" * 70)
    print("YOLO11 VALIDATION REPORT")
    print("Edge Deployment Readiness Assessment")
    print("=" * 70)

    # Model Size
    print("\n1. MODEL SIZE")
    print("-" * 50)
    print(f"   Path: {size_results.get('path', 'N/A')}")
    print(f"   Size: {size_results.get('size_mb', 'N/A')} MB")
    print(f"   Target: <{size_results.get('target_mb', 20)} MB")
    print(f"   Status: [{size_results.get('status', 'N/A')}]")

    # Accuracy
    print("\n2. ACCURACY METRICS")
    print("-" * 50)
    print(f"   mAP50: {accuracy_results.get('mAP50', 'N/A')}")
    print(f"   mAP50-95: {accuracy_results.get('mAP50_95', 'N/A')}")
    print(f"   Precision: {accuracy_results.get('precision', 'N/A')}")
    print(f"   Recall: {accuracy_results.get('recall', 'N/A')}")
    print(f"   F1 Score: {accuracy_results.get('f1_score', 'N/A')}")
    print(f"   Target mAP50: >{accuracy_results.get('target_mAP50', 0.70)}")
    print(f"   Status: [{accuracy_results.get('status', 'N/A')}]")

    # Per-class breakdown
    if 'per_class_ap50' in accuracy_results:
        print("\n   Per-Class AP50:")
        for cls_name, ap in accuracy_results['per_class_ap50'].items():
            status = "OK" if ap >= 0.5 else "LOW"
            print(f"     - {cls_name:15s}: {ap:.4f} [{status}]")

    # Speed
    if speed_results:
        print("\n3. INFERENCE SPEED")
        print("-" * 50)
        print(f"   Device: {speed_results.get('device', 'N/A')}")
        print(f"   Image Size: {speed_results.get('image_size', 'N/A')}")
        latency = speed_results.get('latency_ms', {})
        print(f"   Mean Latency: {latency.get('mean', 'N/A')} ms")
        print(f"   P95 Latency: {latency.get('p95', 'N/A')} ms")
        print(f"   P99 Latency: {latency.get('p99', 'N/A')} ms")
        print(f"   FPS: {speed_results.get('fps', 'N/A')}")
        print(f"   Target: 30+ FPS (<33.3 ms)")
        print(f"   Status: [{speed_results.get('status', 'N/A')}]")

    # Summary
    print("\n" + "=" * 70)
    print("DEPLOYMENT READINESS SUMMARY")
    print("=" * 70)

    checks = [
        ('Model Size (<20MB)', size_results.get('meets_target', False)),
        ('Accuracy (mAP50 >0.70)', accuracy_results.get('meets_target', False)),
    ]
    if speed_results:
        checks.append(('Speed (30+ FPS)', speed_results.get('meets_target', False)))

    all_pass = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"   [{status}] {check_name}")

    print("\n" + "-" * 70)
    if all_pass:
        print("   RESULT: READY FOR EDGE DEPLOYMENT")
    else:
        print("   RESULT: NOT READY - Address failing requirements")
    print("=" * 70 + "\n")


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("YOLO11 Model Validation")
    print("Target: Edge AI Deployment (RDK X5 / QCS6490)")
    print("=" * 70 + "\n")

    results = {
        'model': args.model,
        'device': args.device,
        'image_size': args.imgsz
    }

    # 1. Check model size
    logger.info("Checking model size...")
    size_results = check_model_size(args.model)
    results['model_size'] = size_results

    if not size_results['exists']:
        logger.error(f"Model not found: {args.model}")
        return

    # 2. Validate accuracy
    logger.info("Validating accuracy...")
    accuracy_results = validate_accuracy(
        args.model, args.data, args.device, args.imgsz, args.batch
    )
    results['accuracy'] = accuracy_results

    # 3. Benchmark speed (optional)
    speed_results = None
    if args.benchmark:
        logger.info("Running speed benchmark...")
        speed_results = benchmark_speed(
            args.model, args.device, args.imgsz, args.benchmark_iterations
        )
        results['speed'] = speed_results

    # Print report
    print_report(size_results, accuracy_results, speed_results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
