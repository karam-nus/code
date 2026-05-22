---
title: "Chapter 8 — Weights & Biases"
---

[← Back to Table of Contents](./README.md)

# Weights & Biases — Experiment Tracking That Scales

> *"If you didn't log it, it didn't happen."*

## Installation

```bash
pip install wandb
wandb login  # one-time API key setup
```

## Import Convention

```python
import wandb
from wandb import AlertLevel
```

## Quick Start

```python
import wandb

# Initialize a run
run = wandb.init(
    project='my-cv-project',
    name='resnet50-baseline',
    config={
        'model': 'resnet50',
        'lr': 3e-4,
        'batch_size': 64,
        'epochs': 50,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'img_size': 224,
        'augmentation': 'trivial_augment',
    },
    tags=['baseline', 'resnet', 'imagenet'],
)

# Log metrics
for epoch in range(50):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    wandb.log({
        'epoch': epoch,
        'train/loss': train_loss,
        'val/loss': val_loss,
        'val/accuracy': val_acc,
        'lr': optimizer.param_groups[0]['lr'],
    })

# Finish
wandb.finish()
```

> 🔥 **Pro Tip**: Always pass `config` to `wandb.init()`. It makes every run reproducible and enables hyperparameter comparison in the dashboard.

## Logging Images & Predictions

```python
# Log images with predictions
import torch
import numpy as np

def log_predictions(model, dataloader, class_names, num_images=16):
    """Log sample predictions as a wandb Table"""
    model.eval()
    images, preds, targets = [], [], []

    with torch.no_grad():
        for batch_imgs, batch_targets in dataloader:
            outputs = model(batch_imgs.cuda())
            pred_classes = outputs.argmax(dim=1).cpu()

            for img, pred, target in zip(batch_imgs, pred_classes, batch_targets):
                images.append(wandb.Image(
                    img.permute(1, 2, 0).numpy(),
                    caption=f'Pred: {class_names[pred]} | True: {class_names[target]}'
                ))
                preds.append(class_names[pred])
                targets.append(class_names[target])

                if len(images) >= num_images:
                    break
            if len(images) >= num_images:
                break

    wandb.log({'predictions': images})

# Log detection results
def log_detections(images, boxes, scores, labels, class_names):
    """Log detection results with bounding boxes"""
    wandb_images = []
    for img, box, score, label in zip(images, boxes, scores, labels):
        box_data = []
        for b, s, l in zip(box, score, label):
            if s > 0.5:
                box_data.append({
                    "position": {
                        "minX": b[0].item(), "minY": b[1].item(),
                        "maxX": b[2].item(), "maxY": b[3].item(),
                    },
                    "class_id": l.item(),
                    "box_caption": f"{class_names[l]}: {s:.2f}",
                    "scores": {"confidence": s.item()},
                })

        wandb_images.append(wandb.Image(
            img,
            boxes={"predictions": {"box_data": box_data, "class_labels": dict(enumerate(class_names))}}
        ))
    wandb.log({"detections": wandb_images})
```

## Hyperparameter Sweeps

```python
# Define sweep configuration
sweep_config = {
    'method': 'bayes',           # 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'val/accuracy',
        'goal': 'maximize',
    },
    'parameters': {
        'lr': {'min': 1e-5, 'max': 1e-2, 'distribution': 'log_uniform_values'},
        'batch_size': {'values': [16, 32, 64, 128]},
        'optimizer': {'values': ['adam', 'adamw', 'sgd']},
        'weight_decay': {'min': 0.0, 'max': 0.1},
        'dropout': {'min': 0.0, 'max': 0.5},
    },
}

# Create and run sweep
sweep_id = wandb.sweep(sweep_config, project='my-cv-project')

def train_sweep():
    run = wandb.init()
    config = wandb.config  # hyperparameters chosen by sweep agent

    model = build_model(dropout=config.dropout)
    optimizer = build_optimizer(model, config.optimizer, config.lr, config.weight_decay)
    dataloader = build_dataloader(config.batch_size)

    for epoch in range(20):
        train_loss = train_one_epoch(model, dataloader, optimizer)
        val_acc = validate(model, val_loader)
        wandb.log({'train/loss': train_loss, 'val/accuracy': val_acc})

    wandb.finish()

# Launch 50 sweep runs
wandb.agent(sweep_id, function=train_sweep, count=50)
```

> 🔥 **Pro Tip**: Use `method: 'bayes'` for sweeps. It converges to optimal hyperparameters 3-5x faster than random search.

## Artifacts (Model & Data Versioning)

```python
# Save model as artifact
def save_model_artifact(model, model_name, metadata=None):
    artifact = wandb.Artifact(
        name=model_name,
        type='model',
        metadata=metadata or {},
    )
    torch.save(model.state_dict(), 'model.pt')
    artifact.add_file('model.pt')
    wandb.log_artifact(artifact)

# Load model from artifact
def load_model_artifact(model_name, version='latest'):
    artifact = wandb.use_artifact(f'{model_name}:{version}')
    artifact_dir = artifact.download()
    state_dict = torch.load(f'{artifact_dir}/model.pt')
    return state_dict

# Dataset versioning
def log_dataset_artifact(data_dir, name='my-dataset', split='train'):
    artifact = wandb.Artifact(name=name, type='dataset')
    artifact.add_dir(data_dir)
    artifact.metadata = {'split': split, 'num_images': len(os.listdir(data_dir))}
    wandb.log_artifact(artifact)
```

## Tables (Interactive Data Exploration)

```python
# Create evaluation table
def log_eval_table(model, dataloader, class_names):
    """Log predictions as interactive table"""
    table = wandb.Table(columns=['image', 'prediction', 'truth', 'confidence', 'correct'])

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            outputs = torch.softmax(model(images.cuda()), dim=1)
            confs, preds = outputs.max(dim=1)

            for img, pred, target, conf in zip(images, preds.cpu(), targets, confs.cpu()):
                table.add_data(
                    wandb.Image(img.permute(1, 2, 0).numpy()),
                    class_names[pred],
                    class_names[target],
                    conf.item(),
                    pred == target,
                )

    wandb.log({'eval_table': table})
```

## Alerts & Monitoring

```python
# Send alert when metric degrades
val_acc = validate(model, val_loader)
if val_acc < 0.8:
    wandb.alert(
        title='Accuracy Drop',
        text=f'Validation accuracy dropped to {val_acc:.4f}',
        level=AlertLevel.WARN,
    )

# Track system metrics (automatic)
# GPU utilization, memory, CPU — all logged automatically by default
```

## Common Patterns

```python
# --- Resume training from checkpoint ---
run = wandb.init(project='my-project', resume='allow', id='previous_run_id')

# --- Offline mode (no internet) ---
import os
os.environ['WANDB_MODE'] = 'offline'
# Later sync: wandb sync ./wandb/offline-run-*

# --- Watch model gradients ---
wandb.watch(model, log='all', log_freq=100)  # logs gradients + parameters

# --- Custom x-axis ---
wandb.define_metric('val/*', step_metric='epoch')  # plot val metrics vs epoch
wandb.define_metric('train/*', step_metric='step')  # plot train metrics vs step

# --- Group related runs ---
run = wandb.init(project='my-project', group='experiment-v2', job_type='train')

# --- Log confusion matrix ---
wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(
    probs=None, y_true=all_labels, preds=all_preds, class_names=class_names
)})

# --- Log learning rate finder ---
wandb.log({'lr_finder': wandb.Table(
    data=list(zip(lrs, losses)),
    columns=['learning_rate', 'loss']
)})
```

## Prefer / Avoid

| ❌ Avoid | ✅ Prefer | Why |
|----------|-----------|-----|
| No config in `wandb.init()` | Always pass full `config` dict | Reproducibility |
| `print()` metrics | `wandb.log()` everything | Searchable, comparable |
| Manual model checkpointing only | `wandb.Artifact` for models | Versioned, linked to runs |
| Random sweep for >5 params | Bayesian sweep | Much more efficient |
| Forget `wandb.finish()` | Always finish or use context manager | Proper cleanup |
| Log every step for slow metrics | Log expensive metrics every N steps | Performance |

---

[← Previous: Chapter 7 — TorchMetrics](./07_torchmetrics.md) · **Next: [Chapter 9 — MLflow →](./09_mlflow.md)**

---
*Last updated: May 2026*
