# 🚦 Smart City Surveillance — Adaptive Continual Learning System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://ultralytics.com)
[![Gradio](https://img.shields.io/badge/Dashboard-Gradio-yellow.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A real-time, adaptive traffic surveillance system that detects vehicles, traffic signs, helmet violations, and road accidents — and learns new object classes incrementally without forgetting previously learned ones.

---

## 🎯 Problem

Static deep learning models require **full retraining from scratch** every time a new object category is introduced. In continuously evolving urban environments, this is expensive, slow, and impractical. Sequential training without memory safeguards causes **catastrophic forgetting** — the model overwrites prior knowledge while learning new data.

---

## 💡 Solution

A dual-model architecture built around an **Adaptive Continual Learning Engine**:

- **Unified YOLOv8** — detects 75 classes (vehicles, traffic signs, helmet compliance) in a single forward pass
- **Standalone Binary CNN** — parallel scene-level accident classification with temporal smoothing
- **Experience Replay Buffer** — mixes old-class samples into new training data to prevent forgetting
- **Hot-Reload Deployment** — new classes added via dashboard are live instantly with no server restart

---

## 📊 Results

| Metric | Value |
|---|---|
| Final mAP@0.5 — 75 classes | **85.9%** |
| Forgetting amount without CL | 0.4762 |
| Forgetting amount with CL | **0.0888** |
| Traffic sign retention after CL update | **39.89%** vs 1.14% baseline |
| Accident detection precision | **0.9314** |
| Accident false positive rate (with smoothing) | **6.1%** |
| Helmet violation precision (overlap method) | **0.89** |

---
# 🚦 Smart City Surveillance System

An AI-powered smart city surveillance project focused on:

- 🚗 Vehicle & traffic monitoring
- 🪖 Helmet compliance detection
- 🚨 Accident detection
- 🧠 Continual learning for incremental class updates
- 📊 Interactive Gradio dashboard for real-time inference

---

# 📁 Project Structure

```bash
smart-city-surveillance/
│
├── app.py.ipynb
│   └── Entry point for launching the Gradio dashboard
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│   └── Git ignored files and folders
│
├── config/
│   │
│   └── data.yaml
│       └── YOLO dataset configuration for 75 classes
│
├── phase1_catastrophic_forgetting/
│   └── Demonstrates catastrophic forgetting and continual learning
│   │
│   ├── README.md
│   ├── 01_catastrophic_forgetting_demo.ipynb
│   └── 02_continual_learning_implementation.ipynb
│
├── phase2_unified_model/
│   └── Building the unified multitask surveillance model
│   │
│   ├── README.md
│   ├── 01_helmet_compliance_baseline.ipynb
│   └── 02_unified_multitask_model_training.ipynb
│
├── phase3_incremental_dashboard/
│   └── Real-time inference and dashboard system
│   │
│   ├── README.md
│   ├── 01_end_to_end_inference_pipeline.ipynb
│   └── 02_interactive_gradio_dashboard.ipynb
│
├── Accident-Detection-System/
│   └── Standalone CNN-based accident detection pipeline
│   │
│   └── README.md
│
└── sample_output/
    └── Detection screenshots and inference results
```

---
## 🔄 How It Works

### Phase 1 — Catastrophic Forgetting Analysis
Experimentally proves that sequential YOLO training collapses prior knowledge (mAP: 48.77% → 1.14%), then validates that Replay Buffer + EWC + LwF strategies recover retention to 39.89% with a forgetting amount of just 0.0888.

### Phase 2 — Unified 75-Class Model
Constructs a merged Indian road dataset from multiple Roboflow sources. Trains a single YOLOv8 model through three incremental CL tasks — signs → signs + vehicles → signs + vehicles + helmet — using a balanced replay buffer at each step.

### Phase 3 — Live Incremental Class Addition
Deploys the complete pipeline as an interactive Gradio dashboard. A new class (e.g. ambulance) is added via the UI using a 1:2 replay ratio, confuse limits, and backbone freeze=10 — then hot-reloaded into the live session with zero downtime.

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/smart-city-surveillance
cd smart-city-surveillance
pip install -r requirements.txt
```

> **Model weights** are not included in this repository due to file size.
> See [`models/README.md`] for Git LFS setup or Google Drive download links.

---

## 🚀 Quick Start

```bash
# Open the dashboard notebook and run all cells
# Gradio link appears at the bottom — open in browser
```

Update model paths before launching:

```python
YOLO_PATH = "models/yolo_unified.pt"
ACC_PATH  = "models/accident_cnn.h5"
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Accident Classification | Binary CNN (Keras / TensorFlow) |
| Dashboard | Gradio |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| Augmentation | Albumentations |
| Dataset Management | Roboflow |
| Data Processing | Pandas, NumPy |

---

## 🖼️ Sample Output

| Detection | Description |
|---|---|
| Vehicle + no_helmet + traffic sign | Multi-class simultaneous detection |
| ACCIDENT DETECTED 76% | CNN triggered with temporal confirmation |
| Ambulance 0.91 | New class detected after incremental training |
| Violation log table | Timestamped CSV rendered live in dashboard |

---

## 📄 License

This project is released under the [MIT License](LICENSE).Sonnet 4.6Adaptive
