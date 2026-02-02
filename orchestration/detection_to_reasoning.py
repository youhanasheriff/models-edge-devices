#!/usr/bin/env python3
"""
Detection to Reasoning Bridge
Part of Strategy 1: Modular Split Architecture

Bridges Layer 1 (YOLO Detection) to Layer 3 (LLM Reasoning)
Implements Layer 2 (Safety Logic Validation)

Pipeline:
  Detection JSON → Validation → Reasoning Trigger → LLM Prompt
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    """Safety priority levels for detected objects."""
    HIGH = 0      # Immediate danger - always trigger reasoning
    MEDIUM = 1    # Potential hazard - trigger if confidence high
    LOW = 2       # Environmental awareness - optional reasoning


@dataclass
class DetectionEvent:
    """Represents a significant detection event requiring reasoning."""
    trigger_reason: str
    priority: SafetyLevel
    detections: List[Dict]
    context: str


class DetectionToReasoningBridge:
    """
    Bridges YOLO detection output to LLM reasoning engine.

    Responsibilities:
    1. Validate detections (Layer 2 safety logic)
    2. Determine if reasoning should be triggered
    3. Generate structured prompts for the LLM
    4. Track temporal context across frames
    """

    # System prompt for the reasoning model
    SYSTEM_PROMPT = """You are a safety advisor for users with visual impairments navigating indoor environments.

Your role:
- Analyze real-time obstacle detection data
- Provide clear, actionable safety guidance
- Prioritize immediate hazards over general information
- Be concise but thorough about safety-critical details

IMPORTANT RULES:
- Base responses ONLY on the detected objects provided
- Do NOT hallucinate or assume objects not in the data
- If no high-priority objects detected, confirm the path is clear
- Use spatial terms (left, right, ahead, nearby) based on detection positions"""

    def __init__(
        self,
        min_confidence_for_reasoning: float = 0.5,
        max_frame_history: int = 10,
        reasoning_cooldown_frames: int = 5
    ):
        """
        Initialize the bridge.

        Args:
            min_confidence_for_reasoning: Minimum confidence to include in prompt
            max_frame_history: Number of frames to track for temporal analysis
            reasoning_cooldown_frames: Minimum frames between reasoning triggers
        """
        self.min_confidence = min_confidence_for_reasoning
        self.max_history = max_frame_history
        self.cooldown_frames = reasoning_cooldown_frames

        self.frame_history: List[Dict] = []
        self.last_reasoning_frame: int = -self.cooldown_frames
        self.previous_classes: set = set()

    def validate_detections(self, detection_json: Dict) -> Dict:
        """
        Layer 2: Validate and filter detections.

        Applies:
        - Confidence filtering
        - Sanity checks (impossible scenarios)
        - Temporal consistency
        """
        validated = detection_json.copy()

        # Filter by minimum confidence for reasoning
        filtered_detections = [
            d for d in validated['detections']
            if d['confidence'] >= self.min_confidence
        ]

        # Additional validation rules
        validated_detections = []
        for det in filtered_detections:
            # Rule: Very small detections (< 1% of image) might be noise
            if det['area_ratio'] < 0.005:
                continue

            # Rule: Detections at extreme edges might be partial
            center = det['center']
            img_size = validated['image_size']
            edge_margin = 0.05  # 5% from edge

            if (center['x'] < img_size['width'] * edge_margin or
                center['x'] > img_size['width'] * (1 - edge_margin) or
                center['y'] < img_size['height'] * edge_margin or
                center['y'] > img_size['height'] * (1 - edge_margin)):
                # Mark as edge detection but still include
                det['edge_detection'] = True

            validated_detections.append(det)

        validated['detections'] = validated_detections
        validated['summary']['total_objects'] = len(validated_detections)
        validated['summary']['validated'] = True

        # Update high priority list after filtering
        validated['summary']['high_priority'] = [
            d['class'] for d in validated_detections
            if d['safety_level'] == 'high'
        ]

        # Update classes detected
        validated['summary']['classes_detected'] = list(set(
            d['class'] for d in validated_detections
        ))

        # Add to history
        self.frame_history.append(validated)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)

        return validated

    def should_trigger_reasoning(self, validated_json: Dict) -> tuple[bool, str]:
        """
        Determine if the reasoning engine should be activated.

        Returns:
            Tuple of (should_trigger, reason)
        """
        frame_id = validated_json['frame_id']
        summary = validated_json['summary']

        # Check cooldown
        frames_since_last = frame_id - self.last_reasoning_frame
        if frames_since_last < self.cooldown_frames:
            return False, "cooldown_active"

        # Always trigger for high-priority objects
        if summary['high_priority']:
            return True, f"high_priority_detected: {summary['high_priority']}"

        # Trigger for complex scenes that need interpretation
        if summary['scene_complexity'] == 'complex':
            return True, "complex_scene"

        # Trigger if new class appeared
        current_classes = set(summary['classes_detected'])
        new_classes = current_classes - self.previous_classes
        self.previous_classes = current_classes

        if new_classes:
            return True, f"new_objects_appeared: {list(new_classes)}"

        # Trigger for moderate scenes with medium-priority objects
        medium_priority = [
            d for d in validated_json['detections']
            if d['safety_level'] == 'medium' and d['confidence'] >= 0.7
        ]
        if medium_priority and summary['scene_complexity'] in ['moderate', 'complex']:
            return True, "medium_priority_in_moderate_scene"

        return False, "no_trigger_condition_met"

    def describe_position(self, center: Dict, image_size: Dict) -> str:
        """Convert pixel coordinates to human-readable spatial description."""
        x_ratio = center['x'] / image_size['width']
        y_ratio = center['y'] / image_size['height']

        # Horizontal position
        if x_ratio < 0.33:
            h_pos = "to your left"
        elif x_ratio > 0.66:
            h_pos = "to your right"
        else:
            h_pos = "directly ahead"

        # Vertical position (higher y = closer to camera = nearby)
        if y_ratio < 0.33:
            distance = "far ahead"
        elif y_ratio > 0.66:
            distance = "very close"
        else:
            distance = "at moderate distance"

        return f"{h_pos}, {distance}"

    def generate_reasoning_prompt(self, validated_json: Dict, trigger_reason: str) -> str:
        """
        Generate a structured prompt for the reasoning model.

        The prompt includes:
        - Detected objects with positions
        - Scene summary
        - Specific guidance request
        """
        detections = validated_json['detections']
        summary = validated_json['summary']
        image_size = validated_json['image_size']

        # Build object descriptions
        if detections:
            object_lines = []
            for i, det in enumerate(detections, 1):
                position = self.describe_position(det['center'], image_size)
                confidence_pct = int(det['confidence'] * 100)
                safety_marker = "!" if det['safety_level'] == 'high' else ""

                object_lines.append(
                    f"  {i}. {safety_marker}{det['class']} ({confidence_pct}% confident) - {position}"
                )

            objects_text = "\n".join(object_lines)
        else:
            objects_text = "  No objects detected in current frame."

        # Build prompt
        prompt = f"""CURRENT CAMERA VIEW ANALYSIS

DETECTED OBJECTS:
{objects_text}

SCENE SUMMARY:
- Total objects: {summary['total_objects']}
- High priority alerts: {', '.join(summary['high_priority']) if summary['high_priority'] else 'None'}
- Scene complexity: {summary['scene_complexity']}
- Trigger reason: {trigger_reason}

Based on this detection data, please provide:
1. A brief safety assessment of the current view
2. Any immediate warnings about obstacles or hazards
3. Navigation guidance (if movement is detected or path is unclear)

Keep response concise (2-4 sentences) and actionable."""

        return prompt

    def format_for_llm(self, validated_json: Dict) -> Dict:
        """
        Package everything for the reasoning model.

        Returns dict with:
        - system_prompt: Context for the LLM
        - user_prompt: The actual query with detection data
        - should_reason: Whether to actually call the LLM
        - trigger_reason: Why reasoning was/wasn't triggered
        - raw_detections: Original detection data for reference
        """
        should_trigger, reason = self.should_trigger_reasoning(validated_json)

        if should_trigger:
            self.last_reasoning_frame = validated_json['frame_id']

        return {
            'system_prompt': self.SYSTEM_PROMPT,
            'user_prompt': self.generate_reasoning_prompt(validated_json, reason),
            'should_reason': should_trigger,
            'trigger_reason': reason,
            'frame_id': validated_json['frame_id'],
            'timestamp': validated_json['timestamp'],
            'raw_detections': validated_json
        }

    def process_detection(self, detection_json: Dict) -> Dict:
        """
        Main entry point: Process a detection frame through the bridge.

        Args:
            detection_json: Output from IndoorObstacleDetector.detect()

        Returns:
            LLM-ready dict with reasoning prompt if triggered
        """
        # Step 1: Validate detections (Layer 2)
        validated = self.validate_detections(detection_json)

        # Step 2: Format for LLM (Layer 3 input)
        llm_input = self.format_for_llm(validated)

        return llm_input


# Standalone usage example
def main():
    """Demo the bridge with sample detection data."""

    # Sample detection output (simulating YOLO output)
    sample_detection = {
        "timestamp": "2026-01-22T10:30:00.123Z",
        "frame_id": 100,
        "inference_ms": 18.5,
        "fps": {"current": 54.1, "average": 52.3},
        "image_size": {"width": 640, "height": 640},
        "detections": [
            {
                "class": "door",
                "class_id": 1,
                "confidence": 0.92,
                "bbox": {"x1": 200, "y1": 100, "x2": 400, "y2": 500, "width": 200, "height": 400},
                "center": {"x": 300, "y": 300},
                "area_ratio": 0.195,
                "safety_level": "medium"
            },
            {
                "class": "obstacle",
                "class_id": 5,
                "confidence": 0.85,
                "bbox": {"x1": 50, "y1": 450, "x2": 150, "y2": 600, "width": 100, "height": 150},
                "center": {"x": 100, "y": 525},
                "area_ratio": 0.037,
                "safety_level": "high"
            },
            {
                "class": "person",
                "class_id": 6,
                "confidence": 0.78,
                "bbox": {"x1": 500, "y1": 200, "x2": 600, "y2": 550, "width": 100, "height": 350},
                "center": {"x": 550, "y": 375},
                "area_ratio": 0.085,
                "safety_level": "high"
            }
        ],
        "summary": {
            "total_objects": 3,
            "high_priority": ["obstacle", "person"],
            "scene_complexity": "moderate",
            "classes_detected": ["door", "obstacle", "person"]
        }
    }

    # Initialize bridge
    bridge = DetectionToReasoningBridge(
        min_confidence_for_reasoning=0.5,
        reasoning_cooldown_frames=5
    )

    # Process detection
    llm_input = bridge.process_detection(sample_detection)

    # Display results
    print("=" * 70)
    print("DETECTION TO REASONING BRIDGE - DEMO")
    print("=" * 70)

    print(f"\nFrame ID: {llm_input['frame_id']}")
    print(f"Should Trigger Reasoning: {llm_input['should_reason']}")
    print(f"Trigger Reason: {llm_input['trigger_reason']}")

    print("\n" + "-" * 70)
    print("SYSTEM PROMPT (for LLM context):")
    print("-" * 70)
    print(llm_input['system_prompt'])

    print("\n" + "-" * 70)
    print("USER PROMPT (sent to LLM):")
    print("-" * 70)
    print(llm_input['user_prompt'])

    print("\n" + "=" * 70)
    print("This prompt would be sent to the reasoning engine (SmolLM2/Phi-3)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
