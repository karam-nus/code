---
title: "Chapter 9 — MLflow"
---

[← Back to Table of Contents](./README.md)

# MLflow — Open-Source ML Lifecycle Management

> *"From experiment to production in one platform."*

## Installation

```bash
pip install mlflow
# Start local tracking server:
mlflow server --host 0.0.0.0 --port 5000
```

## Import Convention

```python
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
```

## Quick Start

```python
import mlflow

# Set tracking URI (local or remote)
mlflow.set_tracking_uri('http://localhost:5000')  # or 'file:./mlruns' for local
mlflow.set_experiment('image-classification')

# Start a run
with mlflow.start_run(run_name='resnet50-baseline'):
    # Log parameters
    mlflow.log_params({
        'model': 'resnet50',
        'lr': 3e-4,
        'batch_size': 64,
        'epochs': 50,
        'optimizer': 'adamw',
        'img_size': 224,
    })

    # Training loop
    for epoch in range(50):
        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_acc = validate(model, val_loader)

        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
        }, step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, 'model')

    # Log artifacts (plots, configs, etc.)
    mlflow.log_artifact('training_curve.png')
    mlflow.log_artifact('config.yaml')
```

> 🔥 **Pro Tip**: Use `mlflow.start_run()` as a context manager. It auto-ends the run even if training crashes, preventing orphaned runs.

## Autologging

```python
# PyTorch Lightning autolog — logs everything automatically
mlflow.pytorch.autolog()

# Or for general frameworks:
mlflow.autolog()  # auto-detects sklearn, pytorch, tensorflow, etc.

# What it logs automatically:
# - Parameters (lr, batch_size, optimizer settings)
# - Metrics (loss, accuracy per epoch)
# - Model artifacts
# - Training duration
# - System metrics
```

> ⚠️ **Pitfall**: `autolog` can be noisy and log too much. For fine-grained control in production, manually log only what you need.

## Model Registry

```python
from mlflow.tracking import MlflowClient

# Register model
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, 'model', registered_model_name='image-classifier')

# Transition model stages
client = MlflowClient()

# Promote to staging
client.transition_model_version_stage(
    name='image-classifier',
    version=1,
    stage='Staging',
)

# Promote to production
client.transition_model_version_stage(
    name='image-classifier',
    version=1,
    stage='Production',
)

# Load production model
model = mlflow.pytorch.load_model('models:/image-classifier/Production')
```

## Model Serving

```bash
# Serve model as REST API (one command!)
mlflow models serve -m "models:/image-classifier/Production" --port 5001

# Or serve a specific run's model:
mlflow models serve -m "runs:/<run_id>/model" --port 5001
```

```python
# Query the served model
import requests
import numpy as np

data = {"inputs": np.random.randn(1, 3, 224, 224).tolist()}
response = requests.post('http://localhost:5001/invocations',
                        json=data,
                        headers={'Content-Type': 'application/json'})
predictions = response.json()
```

## Custom PyFunc Model

```python
import mlflow.pyfunc

class VisionModelWrapper(mlflow.pyfunc.PythonModel):
    """Custom model wrapper for serving"""

    def load_context(self, context):
        import torch
        from torchvision.models import resnet50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = resnet50(num_classes=10)
        state_dict = torch.load(context.artifacts['model_path'])
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)

    def predict(self, context, model_input):
        import torch
        import numpy as np
        tensor = torch.tensor(model_input.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

# Log custom model
with mlflow.start_run():
    torch.save(model.state_dict(), 'model.pt')
    mlflow.pyfunc.log_model(
        artifact_path='model',
        python_model=VisionModelWrapper(),
        artifacts={'model_path': 'model.pt'},
        conda_env={
            'dependencies': ['torch>=2.0', 'torchvision', 'numpy'],
        },
    )
```

## Experiment Comparison & Search

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs with filters
runs = mlflow.search_runs(
    experiment_names=['image-classification'],
    filter_string="metrics.val_accuracy > 0.9 AND params.model = 'resnet50'",
    order_by=['metrics.val_accuracy DESC'],
    max_results=10,
)
print(runs[['params.model', 'params.lr', 'metrics.val_accuracy']])

# Get best run
best_run = runs.iloc[0]
print(f"Best accuracy: {best_run['metrics.val_accuracy']:.4f}")
print(f"Run ID: {best_run['run_id']}")

# Load best model
best_model = mlflow.pytorch.load_model(f"runs:/{best_run['run_id']}/model")
```

## Logging Artifacts

```python
import mlflow
import matplotlib.pyplot as plt
import json

with mlflow.start_run():
    # Log single file
    mlflow.log_artifact('confusion_matrix.png')

    # Log directory
    mlflow.log_artifacts('plots/', artifact_path='figures')

    # Log text/dict
    mlflow.log_text(json.dumps(config, indent=2), 'config.json')

    # Log figure directly
    fig, ax = plt.subplots()
    ax.plot(losses)
    mlflow.log_figure(fig, 'loss_curve.png')
    plt.close()

    # Log model summary
    mlflow.log_text(str(model), 'model_architecture.txt')

    # Set tags for organization
    mlflow.set_tags({
        'team': 'vision',
        'task': 'classification',
        'priority': 'high',
    })
```

## Common Patterns

```python
# --- Nested runs (parent/child) ---
with mlflow.start_run(run_name='hyperparameter-search') as parent:
    for lr in [1e-3, 1e-4, 1e-5]:
        with mlflow.start_run(run_name=f'lr={lr}', nested=True):
            mlflow.log_param('lr', lr)
            model = train_model(lr=lr)
            val_acc = evaluate(model)
            mlflow.log_metric('val_accuracy', val_acc)

# --- Log training curves as artifact ---
def log_training_curves(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Val')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    mlflow.log_figure(fig, 'training_curves.png')
    plt.close()

# --- Compare two models ---
def compare_models(run_id_1, run_id_2):
    client = MlflowClient()
    run1 = client.get_run(run_id_1)
    run2 = client.get_run(run_id_2)

    print(f"Model 1 accuracy: {run1.data.metrics['val_accuracy']:.4f}")
    print(f"Model 2 accuracy: {run2.data.metrics['val_accuracy']:.4f}")

# --- MLflow + Docker deployment ---
# Build Docker image from model
# mlflow models build-docker -m "models:/image-classifier/Production" -n "classifier-service"
# docker run -p 5001:8080 classifier-service
```

## Prefer / Avoid

| ❌ Avoid | ✅ Prefer | Why |
|----------|-----------|-----|
| File-based tracking in production | Remote tracking server | Team collaboration |
| `autolog()` for complex pipelines | Manual `log_params/metrics` | Fine control |
| Saving models manually | `mlflow.pytorch.log_model()` | Auto dependencies |
| No model registry | Stage transitions (Staging→Prod) | Safe deployment |
| Flat experiment structure | Nested runs for sweeps | Cleaner UI |
| Forgetting to end runs | Context manager `with start_run()` | No orphaned runs |

---

[← Previous: Chapter 8 — Weights & Biases](./08_wandb.md) · **Next: [Chapter 10 — Ultralytics →](./10_ultralytics.md)**

---
*Last updated: May 2026*
