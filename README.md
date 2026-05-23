---
permalink: /
---

# ML Coding 101 — Package-by-Package Reference

> *"The best code is the code you don't have to write twice."*

A code-heavy, copy-paste-ready reference for the most common APIs and workflows in computer vision & ML engineering. Each chapter is a standalone package guide — real patterns you'll use daily, not toy examples.

---

## Quick Start Paths

**Path A: "I need to process images fast"**
→ [NumPy](./01_numpy.md) → [OpenCV](./02_opencv.md) → [Matplotlib](./03_matplotlib.md)

**Path B: "I need to train a vision model"**
→ [TorchVision](./05_torchvision.md) → [TorchMetrics](./07_torchmetrics.md) → [W&B](./08_wandb.md)

**Path C: "I need to detect/segment objects"**
→ [Ultralytics](./10_ultralytics.md) → [OpenCV](./02_opencv.md) → [TensorRT](./11_tensorrt.md)

**Path D: "I need to optimize & deploy"**
→ [TorchAO](./06_torchao.md) → [TensorRT](./11_tensorrt.md) → [MLflow](./09_mlflow.md)

---

## Chapters

| # | Package | What You'll Learn |
|---|---------|-------------------|
| **Data & Visualization** | | |
| 1 | [NumPy](./01_numpy.md) | Array ops, broadcasting, indexing, memory layout, random, linear algebra |
| 2 | [OpenCV](./02_opencv.md) | Image I/O, color spaces, transforms, drawing, video, contours, DNN module |
| 3 | [Matplotlib](./03_matplotlib.md) | Figures, subplots, image display, annotations, styling, animation |
| 4 | [tqdm](./04_tqdm.md) | Progress bars, nested loops, pandas integration, custom formatting |
| **Training & Models** | | |
| 5 | [TorchVision](./05_torchvision.md) | Transforms v2, datasets, pretrained models, feature extraction, detection |
| 6 | [TorchAO](./06_torchao.md) | Quantization, sparsity, int8/int4, autoquant, kernel fusion |
| 7 | [TorchMetrics](./07_torchmetrics.md) | Classification, detection, segmentation metrics, custom metrics, DDP |
| **Experiment Tracking** | | |
| 8 | [Weights & Biases](./08_wandb.md) | Logging, sweeps, artifacts, tables, model registry, reports |
| 9 | [MLflow](./09_mlflow.md) | Tracking, model registry, serving, projects, recipes |
| **Detection & Deployment** | | |
| 10 | [Ultralytics](./10_ultralytics.md) | YOLOv8/11, train, val, predict, export, track, segment, pose |
| 11 | [TensorRT](./11_tensorrt.md) | ONNX conversion, engine building, INT8 calibration, dynamic shapes |

---

## Prerequisites

- Python 3.9+
- Basic PyTorch familiarity (`torch.Tensor`, autograd, `nn.Module`)
- A GPU (recommended for chapters 5–11)
- `pip install numpy opencv-python matplotlib tqdm`

---

## Conventions Used

- 🔥 **Pro Tip** — a better approach or non-obvious shortcut
- ⚠️ **Pitfall** — common mistake that wastes hours
- ✅ **Prefer** / ❌ **Avoid** — side-by-side better vs worse patterns
- All code is Python 3.10+ with type hints where helpful
- Tensor shapes shown in comments: `# [B, C, H, W]`

---

*Last updated: May 2026*
