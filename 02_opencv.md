---
title: "Chapter 2 — OpenCV"
---

[← Back to Table of Contents](./README.md)

# OpenCV — Computer Vision Swiss Army Knife

> *"90% of vision preprocessing is OpenCV. Learn it once, use it everywhere."*

## Installation

```bash
pip install opencv-python          # main package
pip install opencv-contrib-python  # with extra modules (SIFT, etc.)
# For headless servers (no GUI):
pip install opencv-python-headless
```

## Import Convention

```python
import cv2
import numpy as np
```

## Image I/O

```python
# Read image (BGR by default!)
img = cv2.imread('image.jpg')                    # [H, W, 3] BGR uint8
img = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)  # preserves alpha channel [H, W, 4]
gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)  # [H, W] single channel

# Write image
cv2.imwrite('output.jpg', img)                   # auto-detects format from extension
cv2.imwrite('output.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])  # quality 0-100
cv2.imwrite('output.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])  # 0-9

# Read from bytes/URL
import urllib.request
resp = urllib.request.urlopen('https://example.com/image.jpg')
arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)       # decode from memory buffer

# Encode to bytes (useful for APIs)
success, buffer = cv2.imencode('.jpg', img)
img_bytes = buffer.tobytes()                     # send over network
```

> ⚠️ **Pitfall**: OpenCV reads images as **BGR**, not RGB! When displaying with matplotlib or passing to PyTorch models, you MUST convert: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

## Color Space Conversions

```python
# BGR ↔ RGB (most common conversion)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# BGR → Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # [H, W]

# BGR → HSV (useful for color-based segmentation)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# H: 0-179, S: 0-255, V: 0-255

# Color-based masking in HSV
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)   # [H, W] binary mask

# BGR → LAB (perceptually uniform)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
```

> 🔥 **Pro Tip**: For color detection, always work in HSV. RGB/BGR makes thresholding nearly impossible for real-world lighting conditions.

## Geometric Transforms

```python
h, w = img.shape[:2]

# Resize
resized = cv2.resize(img, (640, 480))                           # (width, height)!
resized = cv2.resize(img, None, fx=0.5, fy=0.5)               # half size
resized = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)  # bilinear
resized = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)    # best for downscaling

# Crop (just numpy slicing)
crop = img[y1:y2, x1:x2]                                       # [h, w, 3]

# Rotate
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)       # [2, 3]
rotated = cv2.warpAffine(img, M, (w, h))

# Flip
flipped_h = cv2.flip(img, 1)    # horizontal
flipped_v = cv2.flip(img, 0)    # vertical
flipped_both = cv2.flip(img, -1)  # both

# Perspective transform
src_pts = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
dst_pts = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (300, 300))

# Padding / border
padded = cv2.copyMakeBorder(img, 10, 10, 10, 10,
                            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
# Letterbox resize (preserve aspect ratio)
def letterbox(img, new_shape=(640, 640)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img
```

> ⚠️ **Pitfall**: `cv2.resize` takes `(width, height)` but NumPy arrays are `(height, width)`. This is the #1 source of bugs.

## Filtering & Morphology

```python
# Gaussian blur
blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)

# Bilateral filter (edge-preserving)
smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Median blur (great for salt-and-pepper noise)
denoised = cv2.medianBlur(img, 5)

# Canny edge detection
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(mask, kernel, iterations=2)
eroded = cv2.erode(mask, kernel, iterations=1)
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove small noise
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes

# Adaptive threshold
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, blockSize=11, C=2)
```

## Drawing & Annotations

```python
# Draw on a copy to preserve original
canvas = img.copy()

# Rectangle (bounding box)
cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# Circle
cv2.circle(canvas, center=(320, 240), radius=50, color=(0, 0, 255), thickness=-1)  # filled

# Line
cv2.line(canvas, (0, 0), (640, 480), color=(255, 0, 0), thickness=2)

# Text
cv2.putText(canvas, 'person 0.95', (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
            color=(0, 255, 0), thickness=2)

# Polyline / polygon
pts = np.array([[100, 50], [200, 300], [700, 200], [500, 100]], dtype=np.int32)
cv2.polylines(canvas, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
cv2.fillPoly(canvas, [pts], color=(255, 255, 0))  # filled

# Draw detection results
def draw_detections(img, boxes, scores, class_names, threshold=0.5):
    """boxes: [N, 4] as xyxy, scores: [N,]"""
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{score:.2f}'
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img
```

## Video Processing

```python
# Read video
cap = cv2.VideoCapture('input.mp4')
# Or webcam: cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)                 # 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 1920
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 1080
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Write video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Process frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame...
    processed = cv2.GaussianBlur(frame, (5, 5), 0)
    writer.write(processed)

cap.release()
writer.release()

# Seek to specific frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # jump to frame 100
ret, frame = cap.read()
```

> 🔥 **Pro Tip**: For production video pipelines, use `cv2.VideoCapture` with a thread for reading and a queue for processing. The I/O is the bottleneck, not the processing.

## Contours & Shape Analysis

```python
# Find contours
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter by area
large_contours = [c for c in contours if cv2.contourArea(c) > 500]

# Bounding box from contour
for cnt in large_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Rotated bounding box
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # Contour properties
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

# Convex hull
hull = cv2.convexHull(cnt)
```

## DNN Module (Run ONNX models)

```python
# Load ONNX model
net = cv2.dnn.readNetFromONNX('model.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Preprocess
blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0,
                             size=(640, 640),
                             mean=(0, 0, 0),
                             swapRB=True, crop=False)  # [1, 3, 640, 640]

# Inference
net.setInput(blob)
outputs = net.forward()                          # or net.forward(['output_name'])

# For multiple output layers
output_names = net.getUnconnectedOutLayersNames()
outputs = net.forward(output_names)
```

## Common Patterns

```python
# --- Stack images into a grid for visualization ---
def make_grid(images, cols=4):
    """images: list of same-sized BGR images"""
    rows_needed = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    grid = np.zeros((rows_needed * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
    return grid

# --- Apply mask overlay ---
def overlay_mask(img, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay binary mask on image with transparency"""
    overlay = img.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# --- Convert between OpenCV and PIL ---
from PIL import Image

# OpenCV → PIL
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(rgb)

# PIL → OpenCV
cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- Center crop ---
def center_crop(img, crop_size):
    h, w = img.shape[:2]
    ch, cw = crop_size
    y = (h - ch) // 2
    x = (w - cw) // 2
    return img[y:y+ch, x:x+cw]
```

## Prefer / Avoid

| ❌ Avoid | ✅ Prefer | Why |
|----------|-----------|-----|
| `cv2.imread` + assume RGB | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | OpenCV is BGR |
| `cv2.resize(img, (h, w))` | `cv2.resize(img, (w, h))` | resize takes (width, height) |
| Processing full-res for detection | Resize first, scale boxes back | 10x+ faster |
| `INTER_LINEAR` for downscaling | `INTER_AREA` for downscaling | Less aliasing |
| Hardcoded codec strings | `cv2.VideoWriter_fourcc(*'mp4v')` | Cross-platform |
| Drawing on original | `img.copy()` then draw | Preserves source |

---

[← Previous: Chapter 1 — NumPy](./01_numpy.md) · **Next: [Chapter 3 — Matplotlib →](./03_matplotlib.md)**

---
*Last updated: May 2026*
