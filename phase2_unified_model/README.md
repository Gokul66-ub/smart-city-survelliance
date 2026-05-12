# Phase 2 — Building the Unified 75-Class Model

## Overview

Phase 1 proved that continual learning works in principle. Phase 2 applies it at full scale to build a single, production-ready YOLOv8 model that simultaneously detects **vehicles, traffic signs, and helmet compliance** across 75 unified classes — without catastrophic forgetting.

Two notebooks are included. Run them in order.

---

## 📒 Notebooks

### `01_helmet_compliance_baseline.ipynb`
**Helmet Dataset Preparation and Baseline Training**

This notebook has two responsibilities:

**Part A — Dataset Preparation (Task 3 setup)**

Constructs the Task 3 training dataset by merging the motorcycle helmet dataset with a replay buffer of traffic sign and vehicle images from prior tasks.

- Source: Motorcycle Helmet dataset (`helmet_split/`) in YOLO format
- Helmet class IDs are offset by +73 to align with the unified 75-class taxonomy:
  - `helmet` → class 73
  - `no_helmet` → class 74
- A sign replay buffer (600 images from Roboflow) is remapped to unified class IDs using `SIGN_REMAP`
- Output: merged `task3_dataset/` with train/valid/test splits and a `task3.yaml` config file

**Part B — Baseline Independent Training**

Trains three separate, isolated YOLOv8 models — one per task — to establish clean accuracy floors before merging:

| Model | Classes | Purpose |
|---|---|---|
| Traffic Sign Model | 64 classes | Baseline accuracy for signs alone |
| Vehicle Model | 9 classes | Baseline accuracy for vehicles alone |
| Helmet Model | 2 classes | Baseline accuracy for helmet alone |

> These baselines confirm each dataset is clean and each task is individually learnable before CL merging begins.

---

### `02_unified_multitask_model_training.ipynb`
**Three-Step Continual Learning Unification**

Trains the final unified model through three sequential CL tasks, each building on the previous:

**CL Task 1 — Traffic Signs (Foundation)**
Trains YOLOv8 on the Traffic Sign dataset from scratch. This becomes the starting point for all subsequent CL updates.

**CL Task 2 — Signs + Vehicles (Initial Merge)**
Fine-tunes the Task 1 model on a combined dataset: full vehicle training data + a replay buffer of traffic sign samples. The Experience Replay Buffer prevents the model from forgetting sign classes while it learns the 9 new vehicle classes.

**CL Task 3 — Signs + Vehicles + Helmet (Full Unification)**
Fine-tunes the Task 2 model on the Task 3 dataset built in notebook 01: helmet images + replay samples of both signs and vehicles. The final output is a single YOLOv8 model covering all 75 classes.

**Final Unified Class Layout:**

| Range | Category | Count |
|---|---|---|
| 0 – 8 | Vehicles (car, motorcycle, auto, bus, truck, bicycle, person, animal, cart) | 9 |
| 9 – 72 | Indian Traffic Signs | 64 |
| 73 – 74 | Helmet / No-Helmet | 2 |
| **Total** | | **75** |

---

## 📊 Performance — Unified Model

| Model Stage | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Traffic Sign only | 0.913 | 0.905 | 0.948 | 0.816 |
| Traffic Sign + Vehicle | 0.836 | 0.798 | 0.864 | 0.772 |
| Traffic Sign + Vehicle + Helmet | 0.752 | 0.859 | **0.859** | 0.696 |

---

## 📦 Datasets Used

| Dataset | Classes | Source |
|---|---|---|
| India Traffic Sign (Roboflow) | 64 sign classes | `college-opfvn / india-traffic-sign v9` |
| Vehicle Detection | 9 vehicle classes | Roboflow |
| Motorcycle Helmet | helmet, no_helmet | Local (`helmet_split/`) |

---

## ▶️ How to Run

- `01_helmet_compliance_baseline.ipynb` → run **locally** (needs `helmet_split/` on your machine)
- `02_unified_multitask_model_training.ipynb` → run on **Google Colab** (GPU required)

Update the local paths at the top of notebook 01 before running:

```python
H1_PATH     = r"C:\path\to\your\helmet_split"    
OUTPUT_PATH = r"C:\path\to\your\task3_dataset"  
YAML_OUT    = r"C:\path\to\your\task3.yaml"      
```

**Output after both notebooks:**
- `models/yolo_unified.pt` — final 75-class YOLOv8 weights
- `config/data.yaml` — unified class config (copy from `task3.yaml`, set `path: ../data`)
