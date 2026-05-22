---
title: "Chapter 5 — TorchVision"
---

[← Back to Table of Contents](./README.md)

# TorchVision — Models, Transforms & Datasets

> *"Don't rewrite the data pipeline. TorchVision already did it better."*

## Installation

```bash
pip install torchvision
```

## Import Convention

```python
import torch
import torchvision
from torchvision import transforms, models, datasets
from torchvision.transforms import v2  # ← new API, always prefer this
from torchvision import tv_tensors
from torchvision.utils import draw_bounding_boxes, make_grid
```

## Transforms V2 (Modern API)

```python
from torchvision.transforms import v2

# --- Classification transforms ---
train_transform = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.08, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    v2.RandomErasing(p=0.25),
    v2.ToDtype(torch.float32, scale=True),   # uint8 → float [0, 1]
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

val_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# --- Detection / Segmentation transforms (joint!) ---
# V2 transforms work on BOTH images AND targets simultaneously
detection_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomPhotometricDistort(p=0.5),
    v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104)}),
    v2.RandomIoUCrop(),
    v2.SanitizeBoundingBoxes(),   # remove degenerate boxes after crop
    v2.Resize((640, 640)),
    v2.ToDtype(torch.float32, scale=True),
])

# Apply to both image and target
from torchvision import tv_tensors

img = tv_tensors.Image(torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8))
boxes = tv_tensors.BoundingBoxes(
    torch.tensor([[10, 20, 100, 200], [150, 50, 300, 400]]),
    format="XYXY", canvas_size=(480, 640)
)
labels = torch.tensor([1, 3])

# Single call transforms BOTH image and boxes correctly!
transformed = detection_transform({"image": img, "boxes": boxes, "labels": labels})
```

> 🔥 **Pro Tip**: Always use transforms v2. It handles images, bounding boxes, masks, and keypoints jointly — no more manual coordinate transforms!

> ⚠️ **Pitfall**: `v2.ToDtype(torch.float32, scale=True)` replaces the old `ToTensor()`. The old `transforms.ToTensor()` is deprecated and should not be used in new code.

## Pretrained Models

```python
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    vit_b_16, ViT_B_16_Weights,
)

# --- Load with latest weights (new API) ---
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # specific version

# Access preprocessing from weights
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()   # returns the correct transform pipeline
categories = weights.meta["categories"]  # class names list

# --- Feature extraction (remove head) ---
from torchvision.models.feature_extraction import create_feature_extractor

backbone = create_feature_extractor(model, return_nodes={
    'layer2': 'feat_8x',     # 1/8 resolution
    'layer3': 'feat_16x',    # 1/16 resolution
    'layer4': 'feat_32x',    # 1/32 resolution
})

x = torch.randn(1, 3, 224, 224)
features = backbone(x)
print(features['feat_8x'].shape)    # [1, 512, 28, 28]
print(features['feat_16x'].shape)   # [1, 1024, 14, 14]
print(features['feat_32x'].shape)   # [1, 2048, 7, 7]

# --- Transfer learning (freeze + replace head) ---
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, num_classes),
)

# Only train the new head
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

> 🔥 **Pro Tip**: Use `weights.transforms()` to get the exact preprocessing the model was trained with. No more guessing normalization values!

## Detection Models

```python
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
)

# Load pretrained detection model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
model.eval()

# Inference
img = torch.randint(0, 255, (3, 800, 1200), dtype=torch.uint8)
preprocess = weights.transforms()
batch = [preprocess(img)]

with torch.no_grad():
    predictions = model(batch)

# predictions[0] = {'boxes': [N, 4], 'labels': [N], 'scores': [N]}
boxes = predictions[0]['boxes']     # [N, 4] xyxy format
scores = predictions[0]['scores']   # [N,]
labels = predictions[0]['labels']   # [N,]

# Filter by confidence
keep = scores > 0.7
boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

# Custom detection model (different number of classes)
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)
```

## Segmentation Models

```python
from torchvision.models.segmentation import (
    deeplabv3_resnet101, DeepLabV3_ResNet101_Weights,
    fcn_resnet50, FCN_ResNet50_Weights,
)

weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
model.eval()

preprocess = weights.transforms()
img = torch.randint(0, 255, (3, 520, 780), dtype=torch.uint8)
batch = preprocess(img).unsqueeze(0)  # [1, 3, 520, 780]

with torch.no_grad():
    output = model(batch)['out']      # [1, 21, 520, 780] — 21 PASCAL VOC classes
    masks = output.argmax(dim=1)      # [1, 520, 780] — class per pixel
```

## Datasets & DataLoaders

```python
# Built-in datasets
train_dataset = datasets.ImageFolder(
    root='data/train',
    transform=train_transform,
)
# Folder structure: data/train/class_name/image.jpg

val_dataset = datasets.ImageFolder(
    root='data/val',
    transform=val_transform,
)

# DataLoader with optimal settings
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,           # use all CPU cores
    pin_memory=True,         # faster GPU transfer
    persistent_workers=True, # don't respawn workers each epoch
    prefetch_factor=2,       # prefetch 2 batches per worker
    drop_last=True,          # consistent batch sizes for BN
)

# COCO-style detection dataset
from torchvision.datasets import CocoDetection

coco_dataset = CocoDetection(
    root='data/coco/images/train2017',
    annFile='data/coco/annotations/instances_train2017.json',
    transforms=detection_transform,
)
```

> 🔥 **Pro Tip**: Set `num_workers = min(8, os.cpu_count())`, `pin_memory=True`, and `persistent_workers=True`. This alone can 2-3x your training throughput.

> ⚠️ **Pitfall**: Don't use `num_workers > 0` on Windows without `if __name__ == '__main__'` guard. It will crash or hang.

## Visualization Utilities

```python
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid
import matplotlib.pyplot as plt

# Draw bounding boxes on image
img = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8)
boxes = torch.tensor([[50, 50, 200, 200], [300, 100, 500, 400]], dtype=torch.float32)
labels = ['cat 0.95', 'dog 0.87']

drawn = draw_bounding_boxes(img, boxes, labels=labels,
                            colors='lime', width=3, font_size=15)
plt.imshow(drawn.permute(1, 2, 0))  # [C, H, W] → [H, W, C]
plt.axis('off')
plt.savefig('detections.png')
plt.close()

# Draw segmentation masks
masks = torch.randint(0, 2, (3, 480, 640), dtype=torch.bool)  # 3 binary masks
drawn = draw_segmentation_masks(img, masks, alpha=0.5,
                                colors=['red', 'green', 'blue'])

# Make grid from batch
batch = torch.randint(0, 255, (16, 3, 64, 64), dtype=torch.uint8)
grid = make_grid(batch, nrow=8, padding=2, normalize=True)  # [3, H, W]
plt.imshow(grid.permute(1, 2, 0))
plt.savefig('batch_grid.png')
plt.close()
```

## Common Patterns

```python
# --- Complete training pipeline ---
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = ResNet50_Weights.DEFAULT

train_transform = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.TrivialAugmentWide(),       # strong augmentation, zero hyperparams
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageFolder('data/train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                         num_workers=8, pin_memory=True)

model = models.resnet50(weights=weights)
model.fc = nn.Linear(2048, len(train_dataset.classes))
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# --- Extract embeddings for similarity search ---
def extract_embeddings(model, dataloader, device):
    """Get feature vectors from penultimate layer"""
    model.eval()
    # Remove classification head
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    backbone = backbone.to(device)

    embeddings = []
    with torch.no_grad():
        for images, _ in dataloader:
            features = backbone(images.to(device))  # [B, 2048, 1, 1]
            features = features.flatten(1)           # [B, 2048]
            embeddings.append(features.cpu())

    return torch.cat(embeddings, dim=0)              # [N, 2048]

# --- Inference with TTA (Test-Time Augmentation) ---
def predict_with_tta(model, img, n_augments=5):
    """Average predictions across augmented versions"""
    model.eval()
    tta_transform = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preds = []
    with torch.no_grad():
        for _ in range(n_augments):
            aug_img = tta_transform(img).unsqueeze(0)  # [1, 3, 224, 224]
            pred = torch.softmax(model(aug_img), dim=1)
            preds.append(pred)

    return torch.stack(preds).mean(dim=0)  # [1, num_classes]
```

## Prefer / Avoid

| ❌ Avoid | ✅ Prefer | Why |
|----------|-----------|-----|
| `transforms.ToTensor()` | `v2.ToDtype(torch.float32, scale=True)` | Old API deprecated |
| `models.resnet50(pretrained=True)` | `resnet50(weights=ResNet50_Weights.DEFAULT)` | New weights API |
| Guessing normalize values | `weights.transforms()` | Exact training preprocess |
| Manual box transforms in augmentation | `v2` with `tv_tensors` | Joint transforms |
| `num_workers=0` | `num_workers=8, pin_memory=True` | 2-3x faster loading |
| `ImageFolder` with no augmentation | `TrivialAugmentWide()` | Free accuracy boost |

---

[← Previous: Chapter 4 — tqdm](./04_tqdm.md) · **Next: [Chapter 6 — TorchAO →](./06_torchao.md)**

---
*Last updated: May 2026*
