#!/usr/bin/env python3
"""
YOLO11 Inference Pipeline with JSON Structured Output
Layer 1 of Modular Split Architecture for Edge AI

Outputs JSON format compatible with the reasoning engine:
{
    "timestamp": "2026-01-22T10:30:00.123Z",
    "frame_id": 1234,
    "inference_ms": 18.5,
    "detections": [...],
    "summary": {...}
}

Target: 30+ FPS real-time detection for safety applications
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Safety level mapping for indoor obstacles
# High: Immediate danger, requires alert
# Medium: Potential hazard, monitor
# Low: Environmental awareness
SAFETY_LEVELS = {
    'obstacle': 'high',
    'person': 'high',
    'escalator': 'high',
    'door': 'medium',
    'closed_door': 'medium',
    'elevator': 'medium',
    'footpath': 'low',
    'wall': 'low',
}

# Class names (must match training data.yaml)
CLASS_NAMES = [
    'closed_door', 'door', 'elevator', 'escalator',
    'footpath', 'obstacle', 'person', 'wall'
]


class IndoorObstacleDetector:
    """
    YOLO11-based indoor obstacle detector with JSON structured output.

    Designed for real-time safety applications with structured output
    compatible with the reasoning engine (Layer 3).
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cpu',
        img_size: int = 640,
        class_names: List[str] = None
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLO11 model (.pt or .onnx)
            conf_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Inference device ('cpu', '0', 'mps', 'cuda:0')
            img_size: Input image size (default: 640)
            class_names: Custom class names (default: indoor obstacle classes)
        """
        from ultralytics import YOLO

        self.model_path = model_path
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size
        self.class_names = class_names or CLASS_NAMES
        self.frame_count = 0

        # Performance tracking
        self.inference_times: List[float] = []
        self.max_history = 100

        logger.info(f"Detector initialized:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Confidence threshold: {conf_threshold}")
        logger.info(f"  IoU threshold: {iou_threshold}")
        logger.info(f"  Image size: {img_size}")

    def detect(
        self,
        source: Union[str, np.ndarray],
        return_image: bool = False
    ) -> Dict:
        """
        Run detection on image and return JSON-structured output.

        Args:
            source: Image path or numpy array (BGR format from OpenCV)
            return_image: If True, include annotated image in output

        Returns:
            JSON-compatible dictionary with detections
        """
        start_time = time.perf_counter()

        # Run YOLO inference
        results = self.model(
            source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )[0]

        # Calculate inference time
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        self.frame_count += 1

        # Track performance
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.max_history:
            self.inference_times.pop(0)

        # Get image dimensions
        img_height, img_width = results.orig_shape
        img_area = img_height * img_width

        # Process detections
        detections = []
        high_priority = []
        classes_detected = set()

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes

            for i in range(len(boxes)):
                # Bounding box (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

                # Confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Get class name (with bounds checking)
                if cls_id < len(self.class_names):
                    cls_name = self.class_names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"

                classes_detected.add(cls_name)

                # Calculate metrics
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                area_ratio = box_area / img_area
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Safety level
                safety_level = SAFETY_LEVELS.get(cls_name, 'low')
                if safety_level == 'high':
                    high_priority.append(cls_name)

                # Build detection object
                detection = {
                    'class': cls_name,
                    'class_id': cls_id,
                    'confidence': round(conf, 3),
                    'bbox': {
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'width': int(box_width),
                        'height': int(box_height)
                    },
                    'center': {
                        'x': int(center_x),
                        'y': int(center_y)
                    },
                    'area_ratio': round(area_ratio, 4),
                    'safety_level': safety_level
                }
                detections.append(detection)

        # Sort by safety level (high first) then confidence (descending)
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        detections.sort(
            key=lambda d: (priority_order[d['safety_level']], -d['confidence'])
        )

        # Determine scene complexity
        n_detections = len(detections)
        if n_detections == 0:
            scene_complexity = 'clear'
        elif n_detections <= 2:
            scene_complexity = 'simple'
        elif n_detections <= 5:
            scene_complexity = 'moderate'
        else:
            scene_complexity = 'complex'

        # Calculate FPS
        avg_inference = sum(self.inference_times) / len(self.inference_times)
        current_fps = 1000 / inference_time if inference_time > 0 else 0
        avg_fps = 1000 / avg_inference if avg_inference > 0 else 0

        # Build output
        output = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'frame_id': self.frame_count,
            'inference_ms': round(inference_time, 2),
            'fps': {
                'current': round(current_fps, 1),
                'average': round(avg_fps, 1)
            },
            'image_size': {
                'width': img_width,
                'height': img_height
            },
            'detections': detections,
            'summary': {
                'total_objects': n_detections,
                'high_priority': list(set(high_priority)),
                'scene_complexity': scene_complexity,
                'classes_detected': list(classes_detected)
            }
        }

        # Optionally include annotated image
        if return_image:
            output['annotated_image'] = results.plot()

        return output

    def detect_batch(
        self,
        sources: List[Union[str, np.ndarray]]
    ) -> List[Dict]:
        """Run detection on multiple images."""
        return [self.detect(src) for src in sources]

    def detect_stream(
        self,
        source: Union[str, int] = 0,
        output_callback: Optional[Callable[[Dict], None]] = None,
        max_frames: Optional[int] = None,
        show: bool = False,
        save_json: Optional[str] = None
    ):
        """
        Run detection on video stream with JSON output per frame.

        Args:
            source: Video path or camera index (0 for webcam)
            output_callback: Function called with each detection JSON
            max_frames: Maximum frames to process (None for infinite)
            show: Display annotated frames
            save_json: Path to save all detections as JSONL
        """
        import cv2

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {source}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Stream opened: {source}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}")

        # Open output file if saving
        json_file = None
        if save_json:
            json_file = open(save_json, 'w')

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of stream")
                    break

                # Detect
                output = self.detect(frame, return_image=show)

                # Callback
                if output_callback:
                    output_callback(output)
                else:
                    # Default: print summary line
                    summary = output['summary']
                    print(
                        f"Frame {output['frame_id']:5d}: "
                        f"{summary['total_objects']:2d} objects, "
                        f"{output['inference_ms']:6.1f}ms, "
                        f"{output['fps']['current']:5.1f} FPS | "
                        f"Priority: {', '.join(summary['high_priority']) or 'None'}"
                    )

                # Save to file
                if json_file:
                    # Remove image data before saving
                    save_output = {k: v for k, v in output.items() if k != 'annotated_image'}
                    json_file.write(json.dumps(save_output) + '\n')

                # Display
                if show and 'annotated_image' in output:
                    cv2.imshow('Indoor Obstacle Detection', output['annotated_image'])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User quit")
                        break

                # Check frame limit
                if max_frames and self.frame_count >= max_frames:
                    logger.info(f"Reached frame limit: {max_frames}")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()
            if json_file:
                json_file.close()
                logger.info(f"Saved detections to: {save_json}")

            # Print statistics
            if self.inference_times:
                avg_ms = sum(self.inference_times) / len(self.inference_times)
                min_ms = min(self.inference_times)
                max_ms = max(self.inference_times)
                avg_fps = 1000 / avg_ms

                print("\n" + "-" * 50)
                print("Performance Statistics:")
                print(f"  Frames processed: {self.frame_count}")
                print(f"  Average latency: {avg_ms:.1f} ms")
                print(f"  Min latency: {min_ms:.1f} ms")
                print(f"  Max latency: {max_ms:.1f} ms")
                print(f"  Average FPS: {avg_fps:.1f}")
                print(f"  Target (30 FPS): {'PASS' if avg_fps >= 30 else 'FAIL'}")
                print("-" * 50)

    def get_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {'frames': 0}

        times = self.inference_times
        return {
            'frames': self.frame_count,
            'avg_ms': round(sum(times) / len(times), 2),
            'min_ms': round(min(times), 2),
            'max_ms': round(max(times), 2),
            'avg_fps': round(1000 / (sum(times) / len(times)), 1)
        }


def main():
    parser = argparse.ArgumentParser(
        description='YOLO11 Indoor Obstacle Detection with JSON Output'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model path (.pt or .onnx)'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Image/video path or camera index (0 for webcam)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS'
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
        default='cpu',
        help='Device (cpu, 0, mps, cuda:0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for single image or JSONL for video'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display annotated results'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames for video/stream'
    )
    args = parser.parse_args()

    # Initialize detector
    detector = IndoorObstacleDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        img_size=args.imgsz
    )

    # Determine source type
    source = args.source
    is_stream = source.isdigit() or source.endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_stream:
        # Video or webcam stream
        stream_source = int(source) if source.isdigit() else source
        detector.detect_stream(
            source=stream_source,
            max_frames=args.max_frames,
            show=args.show,
            save_json=args.output
        )
    else:
        # Single image
        print("\n" + "=" * 60)
        print("Indoor Obstacle Detection - Single Image")
        print("=" * 60 + "\n")

        output = detector.detect(source, return_image=args.show)

        # Print JSON output
        print("Detection Results (JSON):")
        print("-" * 40)
        # Remove image data for printing
        print_output = {k: v for k, v in output.items() if k != 'annotated_image'}
        print(json.dumps(print_output, indent=2))

        # Save if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(print_output, f, indent=2)
            logger.info(f"Saved to: {args.output}")

        # Display if requested
        if args.show and 'annotated_image' in output:
            import cv2
            cv2.imshow('Detection Result', output['annotated_image'])
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
