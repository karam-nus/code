---
title: "Chapter 4 — tqdm"
---

[← Back to Table of Contents](./README.md)

# tqdm — Progress Bars Done Right

> *"Always know how long your training run has left."*

## Installation

```bash
pip install tqdm
```

## Import Convention

```python
from tqdm import tqdm, trange
from tqdm.auto import tqdm as auto_tqdm  # auto-detects notebook vs terminal
```

## Basic Usage

```python
import time

# Simple iterable wrapping
for item in tqdm(range(1000)):
    time.sleep(0.001)
# Output: 100%|████████████████| 1000/1000 [00:01<00:00, 912.34it/s]

# trange shortcut
for i in trange(100):
    time.sleep(0.01)

# With description
for batch in tqdm(dataloader, desc='Training'):
    pass
# Output: Training: 100%|████████| 500/500 [02:30<00:00, 3.33it/s]

# Unknown length (streaming)
for item in tqdm(stream, total=None):
    pass  # shows items/sec without percentage
```

## Training Loop Integration

```python
# --- Standard training loop with tqdm ---
def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    return running_loss / len(dataloader)
# Output: Epoch 5: 100%|████| 391/391 [01:23<00:00, loss=0.3421, acc=87.65%, lr=1.00e-03]

# --- Multi-epoch progress ---
epochs = 50
epoch_pbar = trange(epochs, desc='Training')
for epoch in epoch_pbar:
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
    val_loss = validate(model, val_loader, criterion, device)
    epoch_pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'})
```

## Nested Progress Bars

```python
# Nested bars (epochs + batches)
for epoch in trange(10, desc='Epochs', position=0):
    for batch in tqdm(dataloader, desc=f'Epoch {epoch}', position=1, leave=False):
        pass  # process batch

# leave=False removes inner bar when done — keeps output clean
```

> ⚠️ **Pitfall**: In Jupyter notebooks, nested `tqdm` bars can duplicate. Always use `from tqdm.auto import tqdm` for notebook compatibility, or `tqdm.notebook.tqdm`.

## Manual Control

```python
# Manual update (useful for custom increments)
pbar = tqdm(total=1000, desc='Processing')
for chunk in data_chunks:
    process(chunk)
    pbar.update(len(chunk))     # increment by chunk size
pbar.close()

# Context manager (auto-closes)
with tqdm(total=total_files, desc='Uploading') as pbar:
    for f in files:
        upload(f)
        pbar.update(1)
        pbar.set_postfix(file=f.name)

# Write messages without breaking the bar
with tqdm(range(100)) as pbar:
    for i in pbar:
        if i % 25 == 0:
            tqdm.write(f'Checkpoint saved at step {i}')  # prints above the bar
```

> 🔥 **Pro Tip**: Use `tqdm.write()` instead of `print()` inside tqdm loops. Regular `print()` will break the progress bar formatting.

## File Download Progress

```python
import requests

def download_with_progress(url, filepath):
    """Download file with progress bar showing MB"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filepath, 'wb') as f, \
         tqdm(total=total_size, unit='B', unit_scale=True,
              unit_divisor=1024, desc=filepath) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
# Output: model.pt: 45%|████▌     | 1.23G/2.74G [00:45<00:55, 28.1MB/s]
```

## Pandas Integration

```python
import pandas as pd

# Register tqdm with pandas
tqdm.pandas(desc='Processing')

# .progress_apply replaces .apply with a progress bar
df['processed'] = df['text'].progress_apply(lambda x: expensive_function(x))

# Also works with groupby
results = df.groupby('category').progress_apply(aggregate_fn)
```

## Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_images_parallel(image_paths, num_workers=8):
    """Process images in parallel with progress tracking"""
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single, path): path
                   for path in image_paths}

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc='Processing images'):
            result = future.result()
            results.append(result)

    return results
```

## Custom Formatting

```python
# Custom bar format
for i in tqdm(range(1000),
              bar_format='{l_bar}{bar:30}{r_bar}',  # fixed 30-char bar
              colour='green'):
    pass

# Custom units
for mb in tqdm(range(2048), unit='MB', unit_scale=True):
    pass  # 2.00kMB [00:05, 410MB/s]

# Dynamic miniters (reduce update frequency for fast loops)
for i in tqdm(range(1_000_000), miniters=1000):
    pass  # updates every 1000 iterations — less I/O overhead

# Disable in production
verbose = False
for item in tqdm(data, disable=not verbose):
    pass  # no output when disabled
```

> 🔥 **Pro Tip**: For very fast loops (>10k it/s), set `miniters` to reduce terminal I/O overhead. The progress bar update itself can become the bottleneck!

## Common Patterns

```python
# --- Evaluation with tqdm ---
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in tqdm(dataloader, desc='Evaluating', leave=False):
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    return torch.cat(all_preds), torch.cat(all_labels)

# --- Dataset preprocessing ---
def preprocess_dataset(input_dir, output_dir, size=(224, 224)):
    import cv2
    from pathlib import Path

    paths = list(Path(input_dir).rglob('*.jpg'))
    for path in tqdm(paths, desc='Resizing images'):
        img = cv2.imread(str(path))
        img = cv2.resize(img, size)
        output_path = Path(output_dir) / path.relative_to(input_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)

# --- Multi-GPU progress (only show on rank 0) ---
from tqdm import tqdm

def train_ddp(rank, model, dataloader, ...):
    disable = rank != 0  # only rank 0 shows progress
    for batch in tqdm(dataloader, desc='Training', disable=disable):
        pass
```

## Prefer / Avoid

| ❌ Avoid | ✅ Prefer | Why |
|----------|-----------|-----|
| `from tqdm import tqdm` in notebooks | `from tqdm.auto import tqdm` | Auto notebook/terminal |
| `print()` inside tqdm loop | `tqdm.write()` | Doesn't break bar |
| Updating every iteration (fast loops) | `miniters=N` | Reduces I/O overhead |
| Nested bars without `position` | `position=0`, `position=1` | Prevents flickering |
| No `leave=False` on inner bars | `leave=False` for inner bars | Clean output |
| Wrapping generator without `total` | Always pass `total=len(...)` | Shows percentage |

---

[← Previous: Chapter 3 — Matplotlib](./03_matplotlib.md) · **Next: [Chapter 5 — TorchVision →](./05_torchvision.md)**

---
*Last updated: May 2026*
