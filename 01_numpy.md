---
title: "Chapter 1 — NumPy"
---

[← Back to Table of Contents](./README.md)

# NumPy — The Foundation of Everything

> *"If you can express it as a NumPy operation, you probably shouldn't write a loop."*

## Installation

```bash
pip install numpy
```

## Import Convention

```python
import numpy as np
```

## Array Creation

```python
# From Python lists
a = np.array([1, 2, 3], dtype=np.float32)          # [3,]
b = np.array([[1, 2], [3, 4]], dtype=np.int64)      # [2, 2]

# Common initializers
zeros = np.zeros((224, 224, 3), dtype=np.uint8)     # [224, 224, 3]
ones = np.ones((5, 5), dtype=np.float32)            # [5, 5]
empty = np.empty((1000, 1000))                       # uninitialized — fast!
full = np.full((3, 3), fill_value=255)              # all 255s

# Ranges
lin = np.linspace(0, 1, num=100)                    # 100 evenly spaced [0, 1]
rng = np.arange(0, 10, step=0.5)                    # [0, 0.5, 1, ..., 9.5]

# Identity & diagonal
eye = np.eye(4)                                     # [4, 4] identity
diag = np.diag([1, 2, 3])                           # [3, 3] diagonal matrix

# Random
rng = np.random.default_rng(seed=42)               # ← modern API, always use this
samples = rng.normal(0, 1, size=(1000,))            # Gaussian
uniform = rng.uniform(0, 255, size=(64, 64, 3))    # uniform in [0, 255)
indices = rng.choice(1000, size=100, replace=False) # random sampling without replacement
```

> 🔥 **Pro Tip**: Always use `np.random.default_rng()` instead of `np.random.seed()`. The legacy global state API is not thread-safe and deprecated.

## Indexing & Slicing

```python
img = np.zeros((480, 640, 3), dtype=np.uint8)       # [H, W, C]

# Basic slicing (returns views — no copy!)
roi = img[100:200, 200:400]                         # [100, 200, 3]
channel_r = img[:, :, 0]                            # [480, 640] — red channel

# Boolean indexing (returns copy)
mask = img[:, :, 0] > 128
bright_pixels = img[mask]                           # [N, 3] — variable length

# Fancy indexing (returns copy)
rows = np.array([0, 5, 10])
selected = img[rows]                                # [3, 640, 3]

# Ellipsis — shorthand for "all remaining dims"
batch = np.zeros((32, 3, 224, 224))                 # [B, C, H, W]
first_sample = batch[0, ...]                        # [3, 224, 224]
all_channels = batch[..., :112]                     # [32, 3, 224, 112]
```

> ⚠️ **Pitfall**: Slicing returns a *view* (shared memory). Modifying the slice modifies the original! Use `.copy()` if you need independence.

```python
# View vs Copy — know the difference
roi = img[10:20, 10:20]
roi[:] = 255                    # ← MODIFIES img!

safe_roi = img[10:20, 10:20].copy()
safe_roi[:] = 255               # img unchanged
```

## Reshaping & Dimension Manipulation

```python
x = np.random.randn(32, 3, 224, 224)               # [B, C, H, W]

# Reshape (must preserve total elements)
flat = x.reshape(32, -1)                            # [32, 150528]
restore = flat.reshape(32, 3, 224, 224)             # back to original

# Transpose — channel-first to channel-last
x_nhwc = x.transpose(0, 2, 3, 1)                   # [32, 224, 224, 3]
x_nchw = x_nhwc.transpose(0, 3, 1, 2)              # [32, 3, 224, 224]

# Add/remove dimensions
img_2d = np.zeros((224, 224))                       # [224, 224]
img_3d = img_2d[np.newaxis, :, :]                   # [1, 224, 224]
img_4d = np.expand_dims(img_2d, axis=(0, 1))       # [1, 1, 224, 224]
squeezed = np.squeeze(img_4d)                       # [224, 224]

# Stacking and concatenation
imgs = [np.zeros((224, 224, 3)) for _ in range(8)]
batch = np.stack(imgs, axis=0)                      # [8, 224, 224, 3]
concat = np.concatenate(imgs[:2], axis=0)           # [448, 224, 3]
```

> 🔥 **Pro Tip**: `reshape` vs `resize` — `reshape` returns a view when possible (no copy), `resize` changes the array in-place and can truncate/pad. Always prefer `reshape`.

## Broadcasting

```python
# Broadcasting rules: dimensions are compared right-to-left
# A dim matches if: equal, or one of them is 1

image = np.random.randn(224, 224, 3)    # [224, 224, 3]
mean = np.array([0.485, 0.456, 0.406])  # [3,]     — broadcasts across H, W
std = np.array([0.229, 0.224, 0.225])   # [3,]

normalized = (image - mean) / std       # [224, 224, 3] — no loop needed!

# Per-channel operations with explicit shape
channel_weights = np.array([0.2989, 0.5870, 0.1140]).reshape(1, 1, 3)  # [1, 1, 3]
grayscale = (image * channel_weights).sum(axis=-1)  # [224, 224]

# Batch normalization
batch = np.random.randn(32, 3, 224, 224)           # [B, C, H, W]
batch_mean = batch.mean(axis=(0, 2, 3), keepdims=True)  # [1, 3, 1, 1]
batch_std = batch.std(axis=(0, 2, 3), keepdims=True)    # [1, 3, 1, 1]
normalized = (batch - batch_mean) / (batch_std + 1e-5)  # [B, C, H, W]
```

> ⚠️ **Pitfall**: Broadcasting silently expands dimensions. If shapes don't align as expected, you get wrong results without errors. Always verify shapes with `.shape`.

## Memory Layout & Performance

```python
# C-contiguous (row-major) vs F-contiguous (column-major)
c_arr = np.zeros((1000, 1000), order='C')   # default — row-major
f_arr = np.zeros((1000, 1000), order='F')   # column-major

print(c_arr.flags['C_CONTIGUOUS'])          # True
print(f_arr.flags['F_CONTIGUOUS'])          # True

# Why it matters: iterating along contiguous axis is 10-100x faster
# Row iteration on C-order: FAST ✓
row_sums = c_arr.sum(axis=1)

# Make contiguous after transpose
x = np.random.randn(3, 224, 224)
x_transposed = x.transpose(1, 2, 0)        # NOT contiguous
x_contig = np.ascontiguousarray(x_transposed)  # force contiguity

# dtype matters for memory
img_float64 = np.zeros((1080, 1920, 3))                 # 47.5 MB (default float64!)
img_float32 = np.zeros((1080, 1920, 3), dtype=np.float32)  # 23.7 MB
img_uint8 = np.zeros((1080, 1920, 3), dtype=np.uint8)      # 5.9 MB
```

> 🔥 **Pro Tip**: NumPy defaults to `float64`. For ML/vision work, always specify `dtype=np.float32` or `np.uint8`. You'll save 50%+ memory.

## Common Patterns

```python
# --- Normalize image to [0, 1] ---
img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
img_float = img.astype(np.float32) / 255.0      # [0, 1]

# --- ImageNet normalization ---
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
normalized = (img_float - mean) / std

# --- One-hot encoding ---
labels = np.array([0, 3, 1, 2, 0])             # class indices
num_classes = 4
one_hot = np.eye(num_classes)[labels]            # [5, 4]

# --- IoU calculation (vectorized) ---
def iou_vectorized(boxes1, boxes2):
    """boxes: [N, 4] as [x1, y1, x2, y2]"""
    x1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3], boxes2[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)

# --- Non-Maximum Suppression ---
def nms_numpy(boxes, scores, iou_threshold=0.5):
    """boxes: [N, 4], scores: [N,]"""
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        areas = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + areas - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)

# --- Resize with interpolation (nearest neighbor) ---
def resize_nearest(img, new_h, new_w):
    """Simple nearest-neighbor resize without OpenCV"""
    h, w = img.shape[:2]
    row_indices = (np.arange(new_h) * h / new_h).astype(int)
    col_indices = (np.arange(new_w) * w / new_w).astype(int)
    return img[np.ix_(row_indices, col_indices)]

# --- Softmax ---
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

# --- Cosine similarity (batch) ---
def cosine_similarity(a, b):
    """a, b: [N, D]"""
    dot = (a * b).sum(axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + 1e-8)
```

## Linear Algebra

```python
# Matrix multiply
A = np.random.randn(64, 128)    # [64, 128]
B = np.random.randn(128, 256)   # [128, 256]
C = A @ B                       # [64, 256] — prefer @ over np.dot

# Batch matrix multiply
batch_A = np.random.randn(32, 64, 128)   # [B, M, K]
batch_B = np.random.randn(32, 128, 256)  # [B, K, N]
batch_C = np.einsum('bmk,bkn->bmn', batch_A, batch_B)  # [32, 64, 256]
# or: batch_C = batch_A @ batch_B  # also works for batched!

# SVD (used in LoRA, PCA, etc.)
W = np.random.randn(768, 768)
U, S, Vt = np.linalg.svd(W, full_matrices=False)
# Low-rank approximation (rank 64)
rank = 64
W_approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
print(f"Reconstruction error: {np.linalg.norm(W - W_approx):.4f}")

# Eigendecomposition (used in PCA)
cov = np.cov(np.random.randn(100, 50).T)  # [50, 50] covariance
eigenvalues, eigenvectors = np.linalg.eigh(cov)  # use eigh for symmetric
```

## Structured Arrays & File I/O

```python
# Save/load arrays efficiently
features = np.random.randn(10000, 512).astype(np.float32)
labels = np.random.randint(0, 10, 10000)

# Single array
np.save('features.npy', features)
loaded = np.load('features.npy')

# Multiple arrays
np.savez_compressed('dataset.npz', features=features, labels=labels)
data = np.load('dataset.npz')
print(data['features'].shape)   # (10000, 512)
print(data['labels'].shape)     # (10000,)

# Memory-mapped files (huge datasets that don't fit in RAM)
mmap = np.memmap('huge_features.dat', dtype=np.float32,
                  mode='w+', shape=(1_000_000, 2048))
mmap[0:1000] = np.random.randn(1000, 2048).astype(np.float32)
mmap.flush()  # write to disk
```

> 🔥 **Pro Tip**: Use `np.savez_compressed` for datasets. It's 2-5x smaller than uncompressed `.npy` and loads lazily.

## Prefer / Avoid

| ❌ Avoid | ✅ Prefer | Why |
|----------|-----------|-----|
| `np.random.seed(42)` | `rng = np.random.default_rng(42)` | Thread-safe, reproducible |
| `np.dot(A, B)` | `A @ B` | Clearer, same performance |
| `for i in range(len(arr))` | Vectorized ops | 100x+ faster |
| `arr.resize(...)` | `arr.reshape(...)` | reshape doesn't mutate |
| `dtype` default (float64) | Explicit `dtype=np.float32` | Half the memory |
| `np.concatenate` for stacking | `np.stack` for new dim | Semantically clearer |

---

[← Back to Table of Contents](./README.md) · **Next: [Chapter 2 — OpenCV →](./02_opencv.md)**

---
*Last updated: May 2026*
