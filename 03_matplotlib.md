---
title: "Chapter 3 — Matplotlib"
---

[← Back to Table of Contents](./README.md)

# Matplotlib — Publication-Quality Visualization

> *"A plot is worth a thousand print statements."*

## Installation

```bash
pip install matplotlib
```

## Import Convention

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
```

## Basic Figure Creation

```python
# Single plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('My Plot')
plt.tight_layout()
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
plt.close()

# Multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax in axes.flat:
    ax.set_axis_off()
plt.tight_layout()
plt.savefig('grid.png', dpi=150)
plt.close()
```

> 🔥 **Pro Tip**: Always use `fig, ax = plt.subplots()` instead of `plt.plot()`. The OOP API (`ax.plot`) gives you full control and avoids state issues with multiple figures.

> ⚠️ **Pitfall**: Always call `plt.close()` after `plt.savefig()` in loops or scripts. Otherwise you'll leak memory — each figure stays in memory until explicitly closed.

## Displaying Images

```python
# Show single image
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img)
ax.set_axis_off()
ax.set_title('Sample Image')
plt.savefig('image.png', dpi=150, bbox_inches='tight')
plt.close()

# Show image grid (most common in CV)
def show_image_grid(images, titles=None, cols=4, figsize=(16, 8)):
    """Display a grid of images"""
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flat if rows > 1 else [axes] if cols == 1 else axes

    for idx, ax in enumerate(axes):
        if idx < len(images):
            ax.imshow(images[idx])
            if titles:
                ax.set_title(titles[idx], fontsize=9)
        ax.set_axis_off()
    plt.tight_layout()
    return fig

# Show image with bounding boxes
def show_detections(img, boxes, labels=None, scores=None):
    """img: [H, W, 3] RGB, boxes: [N, 4] as xyxy"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor='lime',
                                  facecolor='none')
        ax.add_patch(rect)
        if labels is not None:
            text = f'{labels[i]}'
            if scores is not None:
                text += f' {scores[i]:.2f}'
            ax.text(x1, y1 - 5, text, color='lime', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='black', alpha=0.7))
    ax.set_axis_off()
    plt.tight_layout()
    return fig

# Show image channels separately
def show_channels(img):
    """img: [H, W, 3]"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img)
    axes[0].set_title('RGB')
    for i, (ax, name) in enumerate(zip(axes[1:], ['Red', 'Green', 'Blue'])):
        channel = np.zeros_like(img)
        channel[:, :, i] = img[:, :, i]
        ax.imshow(channel)
        ax.set_title(name)
    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    return fig
```

## Training Curves

```python
# Plot loss curves (most common ML plot)
def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None):
    """Plot training metrics over epochs"""
    has_acc = train_accs is not None
    fig, axes = plt.subplots(1, 1 + has_acc, figsize=(6 * (1 + has_acc), 5))
    if not has_acc:
        axes = [axes]

    # Loss
    axes[0].plot(train_losses, label='Train Loss', color='#3b82f6')
    axes[0].plot(val_losses, label='Val Loss', color='#f97316')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if has_acc:
        axes[1].plot(train_accs, label='Train Acc', color='#3b82f6')
        axes[1].plot(val_accs, label='Val Acc', color='#f97316')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Plot learning rate schedule
def plot_lr_schedule(lrs, title='Learning Rate Schedule'):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lrs, color='#22c55e', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
```

## Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True):
    """Publication-ready confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]}',
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black',
                fontsize=9)

    plt.tight_layout()
    return fig
```

## Histogram & Distribution Plots

```python
# Pixel intensity histogram
def plot_histogram(img, title='Pixel Intensity Distribution'):
    """img: [H, W, 3] uint8"""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        ax.hist(img[:, :, i].ravel(), bins=256, range=(0, 256),
                color=color, alpha=0.5, label=color.capitalize())
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

# Feature distribution (embeddings)
def plot_embedding_distribution(embeddings, labels=None):
    """embeddings: [N, D]"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Norm distribution
    norms = np.linalg.norm(embeddings, axis=1)
    axes[0].hist(norms, bins=50, color='#3b82f6', alpha=0.7)
    axes[0].set_title('Embedding Norms')
    axes[0].set_xlabel('L2 Norm')

    # Cosine similarity distribution
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    sample_idx = np.random.choice(len(embeddings), min(500, len(embeddings)), replace=False)
    sims = cos_sim(embeddings[sample_idx])
    upper_tri = sims[np.triu_indices_from(sims, k=1)]
    axes[1].hist(upper_tri, bins=50, color='#f97316', alpha=0.7)
    axes[1].set_title('Pairwise Cosine Similarity')
    axes[1].set_xlabel('Similarity')

    plt.tight_layout()
    return fig
```

## Annotations & Styling

```python
# Custom style for ML papers
def set_paper_style():
    """Apply publication-ready styling"""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

# Annotated plot with custom styling
fig, ax = plt.subplots(figsize=(10, 6))
epochs = range(1, 51)
train_loss = np.exp(-np.linspace(0, 3, 50)) + np.random.randn(50) * 0.02
val_loss = np.exp(-np.linspace(0, 2.5, 50)) + np.random.randn(50) * 0.03

ax.plot(epochs, train_loss, 'o-', markersize=3, label='Train', color='#3b82f6')
ax.plot(epochs, val_loss, 's-', markersize=3, label='Val', color='#f97316')

# Add annotation arrow
best_epoch = np.argmin(val_loss)
ax.annotate(f'Best: {val_loss[best_epoch]:.3f}',
            xy=(best_epoch + 1, val_loss[best_epoch]),
            xytext=(best_epoch + 10, val_loss[best_epoch] + 0.1),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

ax.axvline(x=best_epoch + 1, color='red', linestyle='--', alpha=0.5)
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.tight_layout()
plt.close()
```

## Animation (Video from Frames)

```python
from matplotlib.animation import FuncAnimation, FFMpegWriter

def create_training_animation(losses_per_step, save_path='training.mp4'):
    """Animate loss curve building up over time"""
    fig, ax = plt.subplots(figsize=(10, 5))

    def update(frame):
        ax.clear()
        ax.plot(losses_per_step[:frame], color='#3b82f6')
        ax.set_xlim(0, len(losses_per_step))
        ax.set_ylim(0, max(losses_per_step) * 1.1)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training Progress — Step {frame}')
        ax.grid(True, alpha=0.3)

    anim = FuncAnimation(fig, update, frames=len(losses_per_step),
                         interval=50, blit=False)
    writer = FFMpegWriter(fps=30)
    anim.save(save_path, writer=writer)
    plt.close()
```

## Common Patterns

```python
# --- Save figure with transparent background ---
fig.savefig('plot.png', transparent=True, dpi=300, bbox_inches='tight')

# --- Side-by-side image comparison ---
def compare_images(img1, img2, title1='Before', title2='After'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img1)
    ax1.set_title(title1)
    ax1.set_axis_off()
    ax2.imshow(img2)
    ax2.set_title(title2)
    ax2.set_axis_off()
    plt.tight_layout()
    return fig

# --- Bar chart for model comparison ---
def plot_model_comparison(model_names, metrics, metric_name='mAP'):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(model_names, metrics, color='#3b82f6', alpha=0.8)
    ax.set_ylabel(metric_name)
    ax.set_title(f'Model Comparison — {metric_name}')
    # Add value labels on bars
    for bar, val in zip(bars, metrics):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

# --- ROC curve ---
def plot_roc(fpr, tpr, auc_score):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color='#3b82f6', linewidth=2, label=f'AUC = {auc_score:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
```

## Prefer / Avoid

| ❌ Avoid | ✅ Prefer | Why |
|----------|-----------|-----|
| `plt.plot(...)` stateful API | `fig, ax = plt.subplots()` | Explicit, no global state |
| `plt.show()` in scripts | `plt.savefig()` + `plt.close()` | Memory leak prevention |
| Default DPI (72) | `dpi=150` for screen, `dpi=300` for papers | Crisp output |
| Forgetting `plt.close()` | Always close after save | Each figure ~20MB RAM |
| `plt.imshow(cv2_img)` directly | Convert BGR→RGB first | Color channels swapped |
| Manual subplot spacing | `plt.tight_layout()` | Auto-fixes overlaps |

---

[← Previous: Chapter 2 — OpenCV](./02_opencv.md) · **Next: [Chapter 4 — tqdm →](./04_tqdm.md)**

---
*Last updated: May 2026*
