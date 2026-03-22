# 🦁 Eco-Tracker Animal Detector

**Real-time wildlife detection in complex natural environments, powered by YOLOv8.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white)](https://docs.ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![mAP@50](https://img.shields.io/badge/mAP%4050-95.9%25-brightgreen)]()

---

## 📋 Overview

Eco-Tracker Animal Detector is a computer vision system designed to identify and localize **Lions, Rhinos, Elephants, and Bears** in real-time across challenging natural habitats — savanna grasslands, dense woodlands, and mixed terrain.

Built on **YOLOv8 Nano** (Ultralytics), the model is optimized for inference on edge devices, including Apple Silicon Macs with **MPS acceleration**, making it suitable for deployment in field monitoring stations and portable research setups.

> **Core philosophy:** This project follows a **Data-Centric AI** approach. The performance leap from v1 to v2 was not achieved by scaling the architecture, but by *deeply understanding the failure modes* and surgically curating the training data.

---

## 🎯 Key Results

| Metric | Score |
|:--|:--|
| **mAP@50** | **0.959 (96%)** |
| **Precision** | 0.939 |
| **Recall** | 0.901 |

| Class | Highlight |
|:--|:--|
| 🦁 Lion | 97% accuracy — near-perfect detection even in tall grass |
| 🦏 Rhino | Background false positives reduced from **50% → 16%** |
| 🐘 Elephant | Robust detection across varying lighting conditions |
| 🐻 Bear | Reliable identification in woodland environments |

---

## 🔬 The Optimization Story: From v1 to v2

This is where the real engineering happened. Raw metrics only tell half the story — here's how they were earned.

### 🚨 The Problem (v1)

The first model looked good on paper but broke down in the field:

- **Class confusion:** Lions misclassified as Rhinos and vice versa, especially in low-contrast scenes.
- **50% false positive rate on background:** Rocks, termite mounds, and shadows were consistently flagged as Rhinos. The model had learned texture shortcuts rather than actual animal morphology.

A standard response would have been to throw more data at the model or scale up the architecture. Instead, I went looking at *what the model was actually seeing*.

### 🛠️ The Solution: Hard Negative Mining

After a systematic error analysis — manually reviewing every false positive — a clear pattern emerged: the model had no concept of "this is just terrain." It had never been explicitly taught what an animal is *not*.

**The fix was surgical, not brute-force:**

1. **Curated ~100 Background Negative Samples** — images of rocks, tall grass, shadows, termite mounds, and dry riverbeds. No annotations. Just pure background context.
2. **Integrated them into the training pipeline** so the model could learn to suppress activations on terrain features that superficially resemble animal silhouettes.

### ⚙️ Fine-Tuning Strategy

Rather than retraining from scratch (and risking catastrophic forgetting), the v2 model was **fine-tuned from the best v1 checkpoint**:

- **Starting point:** `best.pt` weights from v1
- **Learning rate:** `lr0=0.001` — deliberately low to stabilize gradient updates and preserve existing feature representations while integrating new background knowledge
- **Augmentation pipeline:** Mosaic + MixUp to force the model to handle cluttered, multi-object scenes

This approach embodies a core principle: **understand the data before touching the model**.

---

## 📂 Repository Structure

```
eco-tracker-animal-detector/
├── data/
│   ├── images/
│   │   ├── train/          # Training images (including negative samples)
│   │   └── val/            # Validation images
│   ├── labels/
│   │   ├── train/          # YOLO-format annotations
│   │   └── val/
│   └── data.yaml           # Dataset configuration
├── runs/
│   ├── v1/                 # Baseline model artifacts
│   └── v2/                 # Optimized model artifacts
├── weights/
│   └── best.pt             # Final production weights (v2)
├── notebooks/
│   └── analysis.ipynb      # Error analysis & metric visualization
├── train.py                # Training script
├── detect.py               # Inference script
└── README.md
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/eco-tracker-animal-detector.git
cd eco-tracker-animal-detector

# Install dependencies
pip install ultralytics
```

> **Hardware note:** On Apple Silicon (M1/M2/M3), YOLOv8 will automatically leverage **MPS acceleration**. No additional configuration required.

---

## 🏋️ Training

### Dataset Configuration

Create a `data.yaml` file that maps your dataset structure:

```yaml
# data.yaml
path: ./data
train: images/train
val: images/val

names:
  0: Lion
  1: Rhino
  2: Elephant
  3: Bear
```

> **Important:** Background negative samples (rocks, grass, shadows) are placed in `images/train/` with **no corresponding label files**. YOLOv8 interprets missing annotations as fully-negative images, teaching the model to suppress false activations.

### Training — v1 (Baseline)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Nano variant for edge deployment

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    augment=True,       # Mosaic + MixUp enabled
    device="mps",       # Apple Silicon acceleration
)
```

### Training — v2 (Fine-Tuned with Hard Negatives)

```python
from ultralytics import YOLO

# Resume from the best v1 checkpoint
model = YOLO("runs/v1/weights/best.pt")

model.train(
    data="data.yaml",   # Now includes ~100 negative samples
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.001,          # Low LR to preserve learned features
    augment=True,
    device="mps",
)
```

---

## 🔍 Inference

### Single Image

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")

results = model.predict(
    source="path/to/image.jpg",
    conf=0.5,            # Confidence threshold
    save=True,           # Save annotated output
    device="mps",
)
```

### Batch / Directory

```python
results = model.predict(
    source="path/to/images/",
    conf=0.5,
    save=True,
    device="mps",
)
```

### Webcam / Video Stream

```python
results = model.predict(
    source=0,            # Webcam index
    conf=0.5,
    show=True,           # Live display
    device="mps",
)
```

---

## 📊 Performance Analysis

**Key observations:**

- **Lion** achieves 97% accuracy with minimal cross-class confusion — the model has learned to distinguish mane texture and body posture from other large quadrupeds.
- **Rhino** background false positives dropped from 50% to 16% after integrating hard negatives. Remaining errors concentrate on distant, partially occluded subjects — an expected limitation at Nano resolution.
- **Elephant** and **Bear** maintain strong per-class performance with clean separation in feature space.

## 🧠 Lessons Learned

1. **Data quality > model complexity.** A Nano architecture with well-curated data outperformed larger models trained on noisy datasets.
2. **Error analysis is not optional.** The 50% false positive rate on Rhinos was invisible in aggregate metrics — it only surfaced through manual inspection of predictions.
3. **Hard Negative Mining is underrated.** Teaching a model what something is *not* can be as powerful as teaching it what something *is*.
4. **Fine-tuning beats retraining.** A controlled learning rate preserved months of learned feature representations while integrating targeted corrections.

---

## 🗺️ Roadmap

- [ ] Expand species coverage (Giraffes, Zebras, Leopards)
- [ ] Export to ONNX / CoreML for mobile deployment
- [ ] Integrate tracking (ByteTrack) for individual animal re-identification
- [ ] Build a Gradio demo for browser-based inference
- [ ] Night-vision / thermal imaging support

---

## 🤝 Contributing

Contributions are welcome. If you have access to wildlife imagery datasets or ideas for reducing edge-case false positives, open an issue or submit a pull request.

---

<p align="center">
  Built with patience, data, and a healthy obsession with confusion matrices.
</p>
