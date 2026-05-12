# Phase 3 — Incremental Class Addition & Interactive Dashboard

## Overview

Phase 2 produced a working 75-class model. Phase 3 answers the final question: **can the live system learn a brand-new class through the UI, without restarting, without full retraining, and without forgetting the existing 75 classes?**

This phase also wraps the complete pipeline — YOLO detection, CNN accident classification, violation logging, and continual learning — into a single Gradio web dashboard.

Two notebooks are included.

---

## 📒 Notebooks

### `01_end_to_end_inference_pipeline.ipynb`
**Unified Inference + Accident Detection Integration**

Assembles and validates the complete dual-model inference pipeline before the dashboard is built.

**YOLO Detection Branch:**
- Loads `yolo_unified.pt` (75 classes)
- Runs inference on image, video, and webcam inputs
- Routes detections into three logical buckets: Vehicles, Signs, Helmets
- Applies **spatial overlap check** for helmet violation flagging:
  - A violation is logged only when a `no_helmet` bounding box spatially intersects a `motorcycle` bounding box
  - Reduces false positives from 21.3% (label-only) → **8.7%** (overlap method)

**Accident CNN Branch (parallel, independent):**
- Loads `accident_cnn.h5` (binary: accident / no_accident)
- **Conditional activation:** CNN only runs when 2+ vehicles are detected in the frame
- **Temporal smoothing:** maintains a `deque(maxlen=5)` of recent frame confidences; alert fires only when 3 of 5 consecutive frames exceed the threshold
- Reduces false positive rate from **18.4% → 6.1%**

**Session Logging:**
- Violation log: timestamp, detection type, confidence, video timestamp
- Accident log: timestamp, frame number, video time (seconds), confidence
- Both written to persistent CSV files in `logs/`
- **Ruthless Wipe reset:** clears RAM counters and CSV files together to keep display and storage in sync

---

### `02_interactive_gradio_dashboard.ipynb`
**Full Gradio Dashboard with Hot-Reload Continual Learning**

The main deployable interface. Integrates both model branches into a tabbed Gradio web application.

**Dashboard Tabs:**

| Tab | Function |
|---|---|
| Image Detection | Upload any Indian road image → annotated output + detection stats |
| Video Detection | Upload video → frame-by-frame processing → annotated output + full report |
| Webcam Detection | Live camera feed → real-time bounding boxes + compliance counters |
| Violation Dashboard | Session totals + live violation log table + accident log table |
| Continual Learning | Upload new class images + labels → incremental train → hot reload |
| Model Info | Current model class list, weight path, and session configuration |

**Continual Learning Module — How It Works:**

When an operator types a new class name (e.g. `ambulance`) and uploads images + YOLO labels:

1. Validates and stages the uploaded image-label pairs
2. Samples the replay buffer at a **1:2 ratio** (for every new image, 2 old-class images are mixed in)
3. Applies **Confuse Limits** — caps visually similar old-class samples to prevent the new class from being drowned out
4. Locks backbone with **`freeze=10`** to protect foundational geometric feature representations
5. Fine-tunes upper detection layers for 20 epochs at a low learning rate
6. Saves updated `best.pt` weights
7. **Hot-reloads** new weights into the running dashboard — detects the new class immediately with no server restart

**Demonstrated Result:**

| | Value |
|---|---|
| Starting classes | 75 |
| After incremental training | **76 (ambulance added)** |
| Ambulance detection confidence | **0.91** |
| Prior 75-class performance | **Fully retained** |

---

## 🛡️ CL Guardrail Summary

| Mechanism | Setting | Purpose |
|---|---|---|
| Replay Buffer | 1:2 ratio | Anchors old class knowledge |
| Confuse Limit | Dynamic cap | Prevents new class from being drowned out |
| Backbone Freeze | `freeze=10` | Protects geometric feature representations |
| Hot Reload | `best.pt` injection | Zero-downtime model update |
| Temporal Smoothing | 3 of 5 frames | Accident false positive suppression |
| Conditional CNN | 2+ vehicles required | Compute efficiency |

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
```

Open `02_interactive_gradio_dashboard.ipynb` and run all cells.
The Gradio link appears at the bottom — open it in your browser.

Update model paths before running:

```python
YOLO_PATH = "models/yolo_unified.pt"   # ← UPDATE THIS
ACC_PATH  = "models/accident_cnn.h5"   # ← UPDATE THIS
```

Use `share=True` in `demo.launch()` to generate a public Gradio link.

---

## 🖼️ Sample Outputs

| Screenshot | Description |
|---|---|
| Vehicle + no_helmet + cross_road sign | Multi-class simultaneous detection |
| Helmet + no_helmet side by side | Compliant vs. violating rider |
| ACCIDENT DETECTED 76% | CNN triggered with temporal confirmation |
| Ambulance 0.91 | New class detected post-incremental training |
| Violation log table | Timestamped CSV rendered live in dashboard |

All sample screenshots are available in [`sample_output/`](../sample_output/).
