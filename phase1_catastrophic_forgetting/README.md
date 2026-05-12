# Phase 1 — Catastrophic Forgetting: The Problem and The Fix

## Overview

This phase is purely **experimental**. The goal is not to build a working product yet — it is to scientifically prove that sequential training destroys prior knowledge, and then demonstrate that continual learning strategies can recover it.

Two notebooks are included. Run them in order.

---

## 📒 Notebooks

### `01_catastrophic_forgetting_demo.ipynb`
**Sequential Training — Without Continual Learning**

Trains a YOLO model sequentially on two datasets with no memory safeguards:

1. Train on the **Traffic Sign dataset** → evaluate → record baseline mAP
2. Continue training the same model on the **Vehicle dataset** (no replay, no protection)
3. Re-evaluate on Traffic Sign classes

**Observed Result:**

| Metric | Value |
|---|---|
| Traffic mAP@50 before vehicle training | 48.77% |
| Traffic mAP@50 after vehicle training | **1.14%** |
| Vehicle mAP@50 | 5.01% |
| Forgetting Amount | **0.4762** |

> **Conclusion:** The model almost completely overwrites its traffic sign knowledge to accommodate vehicle classes. Both tasks end up performing poorly — a textbook case of catastrophic forgetting.

---

### `02_continual_learning_implementation.ipynb`
**Incremental Learning — With Continual Learning Strategies**

Repeats the same sequential experiment but introduces three CL strategies:

**Strategy 1 — Experience Replay Buffer**
A controlled sample of original traffic sign images is actively mixed into the new vehicle training data. This anchors the model's foundational weights so gradient updates from the vehicle task cannot fully overwrite sign representations.

**Strategy 2 — Elastic Weight Consolidation (EWC)**
Computes a Fisher Information matrix after the first task. Adds a regularization penalty to the loss function that slows down updates to weights that were important for traffic sign detection.

**Strategy 3 — Learning without Forgetting (LwF)**
Runs the old model on new vehicle images to generate soft targets (distillation outputs). The new model is trained to match both the ground-truth vehicle labels and the old model's soft predictions, preserving prior knowledge through knowledge distillation.

**Result after applying all three strategies:**

| Metric | Without CL | With CL |
|---|---|---|
| Traffic mAP@50 after vehicle training | 1.14% | **39.89%** |
| Vehicle mAP@50 | 5.01% | **22.21%** |
| Forgetting Amount | 0.4762 | **0.0888** |

> **Conclusion:** The combination of Replay Buffer + EWC + LwF reduces forgetting by **81.3%** while simultaneously improving vehicle detection — proving that continual learning is effective and necessary for the full system.

---

## 📦 Datasets Used

| Dataset | Classes | Source |
|---|---|---|
| Traffic Sign Detection Benchmark | 64 Indian traffic sign classes | Roboflow |
| Vehicle Detection Dataset | 9 vehicle classes | Roboflow |

---

## ▶️ How to Run

Open both notebooks in **Google Colab** (GPU recommended):

1. Run `01_catastrophic_forgetting_demo.ipynb` fully — observe the mAP collapse
2. Run `02_continual_learning_implementation.ipynb` — observe the recovery

Both notebooks are self-contained and download their datasets automatically via the Roboflow API.
