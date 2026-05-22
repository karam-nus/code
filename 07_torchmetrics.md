---
title: "Chapter 7 — TorchMetrics"
---

[← Back to Table of Contents](./README.md)

# TorchMetrics — Reliable Metrics for PyTorch

> *"If you can't measure it, you can't improve it."*

## Installation

```bash
pip install torchmetrics
# For detection metrics:
pip install torchmetrics[detection]
```

## Import Convention

```python
import torch
import torchmetrics
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
    MulticlassF1Score, MulticlassConfusionMatrix, MulticlassAUROC,
)
from torchmetrics.detection import MeanAveragePrecision
```

## Basic Classification Metrics

```python
# Single metric
accuracy = MulticlassAccuracy(num_classes=10, average='macro')

# Update with batches (accumulates state)
preds = torch.randn(32, 10)   # [B, num_classes] logits
target = torch.randint(0, 10, (32,))  # [B,] class indices

accuracy.update(preds, target)
accuracy.update(preds, target)   # accumulate over multiple batches

# Compute final result
result = accuracy.compute()
print(f"Accuracy: {result:.4f}")  # Accuracy: 0.1234

# Reset for next epoch
accuracy.reset()
```

> 🔥 **Pro Tip**: TorchMetrics accumulates state across batches, so you get exact metrics over the full dataset. Don't average per-batch metrics manually — that gives wrong results!

## MetricCollection (Multiple Metrics at Once)

```python
from torchmetrics import MetricCollection

metrics = MetricCollection({
    'accuracy': MulticlassAccuracy(num_classes=10, average='macro'),
    'precision': MulticlassPrecision(num_classes=10, average='macro'),
    'recall': MulticlassRecall(num_classes=10, average='macro'),
    'f1': MulticlassF1Score(num_classes=10, average='macro'),
})

# Move to device (critical for GPU training!)
metrics = metrics.to(device)

# In training/eval loop
for images, targets in dataloader:
    images, targets = images.to(device), targets.to(device)
    preds = model(images)
    metrics.update(preds, targets)

# Get all metrics at once
results = metrics.compute()
for name, value in results.items():
    print(f"{name}: {value:.4f}")
# accuracy: 0.9234
# precision: 0.9156
# recall: 0.9178
# f1: 0.9167

metrics.reset()
```

> ⚠️ **Pitfall**: Always call `metrics.to(device)` before using in a GPU training loop. Otherwise you'll get device mismatch errors on the first `.update()` call.

## Training Loop Integration

```python
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

def create_metrics(num_classes, device):
    metrics = MetricCollection({
        'acc': MulticlassAccuracy(num_classes=num_classes, average='macro'),
        'f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
    })
    return metrics.to(device)

# Separate metric instances for train and val
train_metrics = create_metrics(num_classes=10, device='cuda')
val_metrics = create_metrics(num_classes=10, device='cuda')

# Training
model.train()
for images, targets in train_loader:
    images, targets = images.cuda(), targets.cuda()
    preds = model(images)
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_metrics.update(preds, targets)

train_results = train_metrics.compute()
print(f"Train — Acc: {train_results['acc']:.4f}, F1: {train_results['f1']:.4f}")
train_metrics.reset()

# Validation
model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images, targets = images.cuda(), targets.cuda()
        preds = model(images)
        val_metrics.update(preds, targets)

val_results = val_metrics.compute()
print(f"Val — Acc: {val_results['acc']:.4f}, F1: {val_results['f1']:.4f}")
val_metrics.reset()
```

## Object Detection Metrics (mAP)

```python
from torchmetrics.detection import MeanAveragePrecision

map_metric = MeanAveragePrecision(
    iou_type='bbox',          # 'bbox' or 'segm'
    iou_thresholds=[0.5, 0.75],  # COCO uses 0.5:0.95 by default
)

# Predictions format: list of dicts per image
preds = [
    {
        'boxes': torch.tensor([[10, 20, 100, 200], [150, 50, 300, 400]]),  # [N, 4] xyxy
        'scores': torch.tensor([0.95, 0.87]),                               # [N,]
        'labels': torch.tensor([1, 3]),                                     # [N,]
    }
]

# Targets format: same structure without scores
targets = [
    {
        'boxes': torch.tensor([[12, 18, 98, 195], [155, 55, 305, 395]]),
        'labels': torch.tensor([1, 3]),
    }
]

map_metric.update(preds, targets)
results = map_metric.compute()

print(f"mAP@0.5: {results['map_50']:.4f}")     # 0.9500
print(f"mAP@0.75: {results['map_75']:.4f}")    # 0.8700
print(f"mAP@[.5:.95]: {results['map']:.4f}")   # 0.7800
```

> ⚠️ **Pitfall**: Detection metrics expect boxes in `xyxy` format (x1, y1, x2, y2), NOT xywh. Convert with `boxes[:, 2:] += boxes[:, :2]` if needed.

## Segmentation Metrics

```python
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import MulticlassJaccardIndex

# Per-class IoU (Jaccard Index)
iou = MulticlassJaccardIndex(num_classes=21, average='none')  # per-class

# preds: [B, num_classes, H, W] logits or probabilities
# target: [B, H, W] integer class labels
preds = torch.randn(4, 21, 256, 256)   # logits
target = torch.randint(0, 21, (4, 256, 256))

iou.update(preds, target)
per_class_iou = iou.compute()   # [21,] — IoU for each class
mean_iou = per_class_iou.mean()

print(f"mIoU: {mean_iou:.4f}")
for cls_idx, cls_iou in enumerate(per_class_iou):
    print(f"  Class {cls_idx}: {cls_iou:.4f}")
```

## Distributed Training (DDP)

```python
# TorchMetrics handles DDP sync automatically!
import torch.distributed as dist

# Metrics automatically sync across processes on .compute()
metrics = MetricCollection({
    'acc': MulticlassAccuracy(num_classes=10, average='macro'),
    'f1': MulticlassF1Score(num_classes=10, average='macro'),
}).to(device)

# Each rank updates with its own batch
metrics.update(local_preds, local_targets)

# .compute() internally does all_gather — gives global result
# No manual dist.all_reduce needed!
global_results = metrics.compute()
```

> 🔥 **Pro Tip**: TorchMetrics automatically synchronizes across DDP ranks on `.compute()`. You don't need any manual `dist.all_reduce()` calls — just use it normally.

## Custom Metrics

```python
from torchmetrics import Metric

class TopKAccuracy(Metric):
    """Custom top-k accuracy metric"""
    full_state_update: bool = False

    def __init__(self, k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """preds: [B, C] logits, target: [B,] indices"""
        topk_preds = preds.topk(self.k, dim=1).indices  # [B, k]
        correct = (topk_preds == target.unsqueeze(1)).any(dim=1)
        self.correct += correct.sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

# Usage
top5 = TopKAccuracy(k=5).to(device)
top5.update(preds, targets)
print(f"Top-5 Accuracy: {top5.compute():.4f}")
```

## Common Patterns

```python
# --- Complete evaluation function ---
def evaluate_classification(model, dataloader, num_classes, device):
    """Full evaluation with all standard metrics"""
    metrics = MetricCollection({
        'accuracy': MulticlassAccuracy(num_classes=num_classes, average='macro'),
        'accuracy_top5': MulticlassAccuracy(num_classes=num_classes, average='macro', top_k=5),
        'precision': MulticlassPrecision(num_classes=num_classes, average='macro'),
        'recall': MulticlassRecall(num_classes=num_classes, average='macro'),
        'f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
        'per_class_f1': MulticlassF1Score(num_classes=num_classes, average='none'),
    }).to(device)

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            metrics.update(preds, targets)

    results = metrics.compute()
    return results

# --- Log metrics to wandb ---
import wandb

def log_metrics(metrics_dict, step, prefix='val'):
    """Log TorchMetrics results to wandb"""
    for name, value in metrics_dict.items():
        if value.dim() == 0:  # scalar
            wandb.log({f'{prefix}/{name}': value.item()}, step=step)
        else:  # per-class
            for i, v in enumerate(value):
                wandb.log({f'{prefix}/{name}_class_{i}': v.item()}, step=step)

# --- Early stopping based on metric ---
class EarlyStopping:
    def __init__(self, patience=10, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0

    def __call__(self, metric_value):
        if self.mode == 'max':
            improved = metric_value > self.best
        else:
            improved = metric_value < self.best

        if improved:
            self.best = metric_value
            self.counter = 0
            return False  # don't stop
        else:
            self.counter += 1
            return self.counter >= self.patience  # stop if patience exceeded

early_stop = EarlyStopping(patience=10, mode='max')
for epoch in range(100):
    val_results = evaluate_classification(model, val_loader, num_classes, device)
    if early_stop(val_results['f1'].item()):
        print(f"Early stopping at epoch {epoch}")
        break
```

## Prefer / Avoid

| ❌ Avoid | ✅ Prefer | Why |
|----------|-----------|-----|
| Manual accuracy accumulation | `torchmetrics.Accuracy` | Handles edge cases, DDP |
| Averaging per-batch metrics | `.update()` + `.compute()` | Mathematically correct |
| `sklearn` metrics in training loop | TorchMetrics (GPU-native) | No CPU transfer overhead |
| Forget `.to(device)` | Always `.to(device)` first | Device mismatch errors |
| One metric instance for train+val | Separate instances | State contamination |
| Manual DDP metric sync | TorchMetrics auto-sync | Less boilerplate, correct |

---

[← Previous: Chapter 6 — TorchAO](./06_torchao.md) · **Next: [Chapter 8 — Weights & Biases →](./08_wandb.md)**

---
*Last updated: May 2026*
