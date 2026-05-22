---
title: "Chapter 10 — Ultralytics"
---

[← Back to Table of Contents](./README.md)

# Ultralytics — YOLO Made Simple

> *"From zero to object detection in 3 lines of code."*

## Installation

```bash
pip install ultralytics
```

## Import Convention

```python
from ultralytics import YOLO, settings
```

## Quick Start — Inference in 3 Lines

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')           # nano (fastest) — auto-downloads
results = model('image.jpg')          # inference
results[0].show()                     # display with annotations

# Model variants by size:
# yolo11n.pt  — Nano   (fastest, least accurate)
# yolo11s.pt  — Small
# yolo11m.pt  — Medium
# yolo11l.pt  — Large
# yolo11x.pt  — XLarge (slowest, most accurate)
```

## Prediction (Detect, Segment, Classify, Pose)

```python
from ultralytics import YOLO

# --- Object Detection ---
model = YOLO('yolo11n.pt')
results = model.predict(
    source='image.jpg',       # str, Path, np.ndarray, torch.Tensor, list
    conf=0.25,                # confidence threshold
    iou=0.7,                  # NMS IoU threshold
    max_det=300,              # max detections per image
    classes=[0, 1, 2],        # filter specific classes (0=person, 1=bicycle, 2=car)
    device='cuda:0',          # or 'cpu', '0,1' for multi-GPU
    save=True,                # save annotated images
    save_txt=True,            # save labels as .txt
)

# Access results
for result in results:
    boxes = result.boxes              # Boxes object
    print(boxes.xyxy)                 # [N, 4] bounding boxes (x1, y1, x2, y2)
    print(boxes.conf)                 # [N,] confidence scores
    print(boxes.cls)                  # [N,] class indices
    print(boxes.xywh)                 # [N, 4] center-x, center-y, width, height

    # Convert to numpy
    boxes_np = boxes.xyxy.cpu().numpy()    # [N, 4]
    scores_np = boxes.conf.cpu().numpy()   # [N,]

# --- Instance Segmentation ---
seg_model = YOLO('yolo11n-seg.pt')
results = seg_model.predict('image.jpg')
for result in results:
    masks = result.masks               # Masks object
    print(masks.data.shape)            # [N, H, W] binary masks
    print(masks.xy)                    # list of polygon coordinates

# --- Pose Estimation ---
pose_model = YOLO('yolo11n-pose.pt')
results = pose_model.predict('image.jpg')
for result in results:
    keypoints = result.keypoints       # Keypoints object
    print(keypoints.xy.shape)          # [N, 17, 2] keypoint coordinates
    print(keypoints.conf.shape)        # [N, 17] keypoint confidence

# --- Classification ---
cls_model = YOLO('yolo11n-cls.pt')
results = cls_model.predict('image.jpg')
for result in results:
    probs = result.probs               # Probs object
    print(probs.top5)                  # top-5 class indices
    print(probs.top5conf)              # top-5 confidences
```

> 🔥 **Pro Tip**: `model.predict()` accepts almost anything as source: file path, directory, URL, numpy array, torch tensor, PIL image, video, webcam (0), or even a list of mixed types.

## Training Custom Models

```python
from ultralytics import YOLO

# --- Train detection model ---
model = YOLO('yolo11n.pt')  # start from pretrained

results = model.train(
    data='coco128.yaml',      # dataset config (or path to custom yaml)
    epochs=100,
    imgsz=640,
    batch=16,                 # batch size (-1 for auto)
    device='0',               # GPU device
    workers=8,
    patience=50,              # early stopping patience
    optimizer='AdamW',
    lr0=0.01,                 # initial learning rate
    lrf=0.01,                 # final LR = lr0 * lrf
    augment=True,
    mosaic=1.0,               # mosaic augmentation probability
    mixup=0.1,                # mixup augmentation probability
    copy_paste=0.1,           # copy-paste augmentation
    close_mosaic=10,          # disable mosaic for last N epochs
    name='my_experiment',     # save to runs/detect/my_experiment/
    resume=False,             # resume from last.pt
)

# --- Custom dataset YAML ---
# my_dataset.yaml:
# path: /data/my_dataset
# train: images/train
# val: images/val
# test: images/test  # optional
#
# nc: 3  # number of classes
# names: ['cat', 'dog', 'bird']

# --- Resume training ---
model = YOLO('runs/detect/my_experiment/weights/last.pt')
model.train(resume=True)
```

> ⚠️ **Pitfall**: Your dataset YAML paths must be relative to the `path` field or absolute. Relative paths from the YAML file location are NOT supported — this is the #1 training error.

## Validation

```python
# Validate on dataset
model = YOLO('runs/detect/my_experiment/weights/best.pt')
metrics = model.val(
    data='my_dataset.yaml',
    imgsz=640,
    batch=32,
    conf=0.001,               # low conf for mAP calculation
    iou=0.6,
    device='0',
    split='test',             # 'val' or 'test'
)

# Access metrics
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# Per-class metrics
for i, name in enumerate(model.names.values()):
    print(f"{name}: mAP50={metrics.box.maps[i]:.4f}")
```

## Export (ONNX, TensorRT, CoreML, etc.)

```python
model = YOLO('best.pt')

# Export to ONNX
model.export(
    format='onnx',
    imgsz=640,
    dynamic=True,          # dynamic batch size
    simplify=True,         # ONNX simplifier
    opset=17,
    half=False,            # FP16
)
# Produces: best.onnx

# Export to TensorRT
model.export(
    format='engine',       # TensorRT
    imgsz=640,
    half=True,             # FP16 (recommended for TRT)
    device='0',
    dynamic=True,
    batch=8,               # max batch size for dynamic
)
# Produces: best.engine

# Export to other formats
model.export(format='torchscript')   # TorchScript
model.export(format='coreml')        # CoreML (iOS)
model.export(format='tflite')        # TFLite (mobile)
model.export(format='openvino')      # OpenVINO (Intel)

# Run inference with exported model (same API!)
onnx_model = YOLO('best.onnx')
results = onnx_model.predict('image.jpg')
```

> 🔥 **Pro Tip**: Export with `half=True` for TensorRT. FP16 is 2x faster with negligible accuracy loss on modern GPUs.

## Object Tracking

```python
# Built-in tracking (BoT-SORT or ByteTrack)
model = YOLO('yolo11n.pt')

# Track objects in video
results = model.track(
    source='video.mp4',
    tracker='botsort.yaml',    # or 'bytetrack.yaml'
    conf=0.3,
    iou=0.5,
    show=True,                 # display live
    save=True,                 # save annotated video
    persist=True,              # persist tracks between frames
)

# Process tracking results
for result in results:
    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().numpy()
        for track_id, box in zip(track_ids, boxes):
            print(f"Track {track_id}: {box}")
```

## Batch Processing & Streaming

```python
# Process directory of images
results = model.predict(source='images/', save=True)

# Stream video (memory-efficient)
results = model.predict(source='video.mp4', stream=True)
for result in results:
    # Process one frame at a time (generator)
    boxes = result.boxes.xyxy
    # ... process

# Webcam
results = model.predict(source=0, show=True, stream=True)

# RTSP stream
results = model.predict(source='rtsp://admin:pass@192.168.1.1/stream', stream=True)

# Process numpy arrays directly
import numpy as np
frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
results = model.predict(source=frame, verbose=False)

# Batch of numpy arrays
frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(4)]
results = model.predict(source=frames, batch=4)
```

> ⚠️ **Pitfall**: Always use `stream=True` for video processing. Without it, all frames are stored in memory — a 10-minute video will OOM.

## Common Patterns

```python
# --- Inference with confidence filtering ---
def detect_objects(model, image, conf=0.5, classes=None):
    """Clean detection function returning numpy arrays"""
    results = model.predict(image, conf=conf, classes=classes, verbose=False)
    result = results[0]

    if len(result.boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    boxes = result.boxes.xyxy.cpu().numpy()      # [N, 4]
    scores = result.boxes.conf.cpu().numpy()     # [N,]
    classes = result.boxes.cls.cpu().numpy().astype(int)  # [N,]
    return boxes, scores, classes

# --- Count objects per class ---
def count_objects(model, image):
    results = model.predict(image, verbose=False)
    class_counts = {}
    for cls_id in results[0].boxes.cls.cpu().numpy():
        name = model.names[int(cls_id)]
        class_counts[name] = class_counts.get(name, 0) + 1
    return class_counts
# {'person': 3, 'car': 2, 'dog': 1}

# --- Multi-model pipeline (detect then classify) ---
det_model = YOLO('yolo11n.pt')
cls_model = YOLO('yolo11n-cls.pt')

def detect_and_classify(image):
    detections = det_model.predict(image, conf=0.5, verbose=False)[0]
    results = []
    for box in detections.boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        cls_result = cls_model.predict(crop, verbose=False)[0]
        results.append({
            'box': box,
            'class': cls_result.probs.top1,
            'confidence': cls_result.probs.top1conf.item(),
        })
    return results

# --- Callback-based training monitoring ---
from ultralytics import YOLO

def on_train_epoch_end(trainer):
    """Custom callback — called after each epoch"""
    metrics = trainer.metrics
    print(f"Epoch {trainer.epoch}: mAP50={metrics.get('metrics/mAP50(B)', 0):.4f}")

model = YOLO('yolo11n.pt')
model.add_callback('on_train_epoch_end', on_train_epoch_end)
model.train(data='coco128.yaml', epochs=10)
```

## Prefer / Avoid

| ❌ Avoid | ✅ Prefer | Why |
|----------|-----------|-----|
| `stream=False` for video | `stream=True` for video | Prevents OOM |
| Loading model every frame | Load once, reuse model | Initialization overhead |
| `conf=0.25` for deployment | Tune conf per use case | Balance precision/recall |
| Training from scratch | Fine-tune from pretrained | Converges faster |
| Ignoring `close_mosaic` | `close_mosaic=10` | Better final accuracy |
| Manual NMS post-processing | Built-in NMS via `iou` param | Already optimized |
| `model('image')` for production | `model.predict(image, verbose=False)` | Less overhead |

---

[← Previous: Chapter 9 — MLflow](./09_mlflow.md) · **Next: [Chapter 11 — TensorRT →](./11_tensorrt.md)**

---
*Last updated: May 2026*
