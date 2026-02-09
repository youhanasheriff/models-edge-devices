# Pocket AI Guardian: Vision Model Training Report

**Date:** 2026-02-05  
**Model:** YOLO11 Nano (yolo11n)  
**Dataset:** IndoorObstacle Detection.v9i.yolov11  
**Author:** Youhana Sheriff  
**Status:** Training In Progress (Epoch 60/150)

---

## 1. Executive Summary

I have successfully initialized and resumed the training of the Layer 1 perception model for the Pocket AI Guardian system. The selected architecture (YOLO11n) continues to meet the strict edge-device constraints (sub-20MB footprint) while delivering excellent baseline performance for obstacle and person detection.

This report covers the **re-training session** initiated following the accidental deletion of previous model weights. The training was stabilized by optimizing the batch size to address memory constraints on the Apple Silicon hardware. Preliminary results from the re-training show significant performance improvements over previous baselines, with mAP @ 50 exceding 70%.

## 2. Training Configuration

| Parameter | Value | Notes |
|:---|:---|:---|
| **Architecture** | YOLO11n (Nano, Edge-Optimized) | |
| **Hardware** | Apple Silicon (M-series) | |
| **Acceleration** | Metal Performance Shaders (MPS) | |
| **Input Resolution** | 640 × 640 | |
| **Batch Size** | **8** | Optimized from 16 to prevent OOM |
| **Epochs** | Target: 150 (Current: 59) | Resumed from Epoch 54 |

**Configuration Update:**  
Initial attempts with **Batch Size 16** resulted in GPU memory overload and process termination. The batch size was reduced to **8**, which stabilized the training process and allowed for successful continuation without memory errors.

## 3. Performance Metrics (Interim)

**status:** Validation at Epoch 57 (Best observed in recent run).

| Metric | Previous Baseline (Feb 2) | Current Value (Feb 5) | Target | Status |
|:---|:---|:---|:---|:---|
| **mAP @ 50** | 44.1% | **75.4%** | >70% | **PASS** (Exceeds Target) |
| **mAP @ 50-95**| N/A | **43.9%** | - | Strong Baseline |
| **Precision** | 30.4% | **85.8%** | >50% | **PASS** |
| **Recall** | 56.6% | **61.5%** | >60% | **PASS** |
| **Model Size** | 5.22 MB | ~5.2 MB | <20 MB | **PASS** |

> **Note:** The current training run demonstrates a substantial leap in accuracy compared to the initial validation, with mAP @ 50 Crossing the 70% threshold.

## 4. Technical Challenges & Resolution

### Incident: Memory Overload & Crash
- **Issue:** Training process terminated unexpectedly during initial epochs.
- **Root Cause:** Batch size of 16 exceeded available GPU memory on the M-series chip when combined with the model complexity and image resolution.
- **Resolution:** 
    1. Cleared previous incomplete run artifacts.
    2. Reduced batch size from 16 to 8.
    3. Resumed training successfully; process is now stable.

## 5. Failure Analysis: “The Walking Blind Spot” (Context)

*The following issues identified in the Feb 2nd report are the primary focus of this extended training run:*

- **Issue:** The model previously failed to reliably detect persons walking directly toward the camera (ego-centric view).
- **Hypothesis:** Dataset bias towards side-profile static poses.
- **Mitigation Strategy:** The current 150-epoch run is intended to maximize feature extraction from the existing data, while parallel data curation efforts (JRDB/SCAND datasets) are underway to address the semantic gap.

## 6. Strategic Plan: Targeted Data Curation

To further eliminate the “Walking Blind Spot,” a targeted ego-centric data enrichment phase has been initiated:

1. **Priority Data Sources:**
   - **JRDB (JackRabbot Dataset):** Robot-height perspective (~1m).
   - **SCAND Dataset:** Corridor and crowd-based walking scenarios.

2. **Action Plan:**
   - **Curate:** Automated AI agents downloading/subsetting datasets (In Progress).
   - **Filter:** Isolate frontal-view persons and approaching motion patterns.
   - **Retrain:** Future fine-tuning will incorporate these high-value samples.

## 7. Next Milestones

- [ ] Complete full 150-epoch training run.
- [ ] Evaluate "Walking Blind Spot" performance with the improved model (75% mAP).
- [ ] Integrate depth-assisted distance warning.
- [ ] Audio feedback calibration.
