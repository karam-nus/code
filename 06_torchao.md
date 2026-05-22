---
title: "Chapter 6 — TorchAO"
---

[← Back to Table of Contents](./README.md)

# TorchAO — Quantization & Optimization for PyTorch

> *"Make your model 2-4x faster without retraining."*

## Installation

```bash
pip install torchao
```

## Import Convention

```python
import torch
import torchao
from torchao.quantization import (
    quantize_,
    int8_weight_only,
    int8_dynamic_activation_int8_weight,
    int4_weight_only,
    autoquant,
)
```

## Quick Start — One-Line Quantization

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchao.quantization import quantize_, int8_weight_only

# Load model
model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().cuda()

# Quantize in-place — one line!
quantize_(model, int8_weight_only())

# That's it. Model is now quantized.
x = torch.randn(1, 3, 224, 224).cuda()
with torch.no_grad():
    out = model(x)    # runs with int8 weights, ~2x smaller, faster matmuls
```

> 🔥 **Pro Tip**: `int8_weight_only()` is the safest quantization — almost zero accuracy loss on most vision models. Start here, then try more aggressive options.

## Weight-Only Quantization

```python
from torchao.quantization import int8_weight_only, int4_weight_only

# INT8 weight-only (best accuracy/speed tradeoff)
quantize_(model, int8_weight_only())
# Weights: float32 → int8 (4x smaller)
# Activations: still float16/32
# Accuracy: <0.1% degradation typically

# INT4 weight-only (more aggressive, for LLMs and large models)
quantize_(model, int4_weight_only(group_size=128))
# Weights: float32 → int4 (8x smaller)
# group_size=128: quantize in groups for better accuracy
# group_size=32: even better accuracy, slightly slower

# Check model size reduction
def model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    return param_size / 1024 / 1024

print(f"Original: {model_size_mb(original_model):.1f} MB")   # ~97.5 MB
print(f"INT8: {model_size_mb(int8_model):.1f} MB")           # ~24.4 MB
print(f"INT4: {model_size_mb(int4_model):.1f} MB")           # ~12.2 MB
```

## Dynamic Quantization (Activations + Weights)

```python
from torchao.quantization import int8_dynamic_activation_int8_weight

# Both weights AND activations quantized
quantize_(model, int8_dynamic_activation_int8_weight())
# Per-token dynamic quantization of activations
# Best for: models where memory bandwidth is bottleneck
# Requires calibration-free — works out of the box
```

> ⚠️ **Pitfall**: Dynamic activation quantization is more aggressive than weight-only. Always validate accuracy on your specific task before deploying.

## Autoquant — Let TorchAO Choose

```python
from torchao.quantization import autoquant

# Autoquant profiles your model and picks the best quantization per layer
model = autoquant(model)

# Run some representative inputs to trigger profiling
for _ in range(10):
    x = torch.randn(1, 3, 224, 224).cuda()
    model(x)

# After profiling, each layer uses optimal quantization:
# - Some layers: int8 (safe)
# - Some layers: int4 (aggressive but OK for that layer)
# - Some layers: no quantization (too sensitive)
```

> 🔥 **Pro Tip**: `autoquant` is the easiest way to get maximum speedup with minimal accuracy loss. It handles the per-layer decision for you.

## Compile + Quantize (Maximum Speed)

```python
import torch

# The magic combo: quantize THEN compile
model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().cuda()
quantize_(model, int8_weight_only())
model = torch.compile(model, mode='max-autotune')

# Warmup (compilation happens on first few runs)
for _ in range(3):
    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224).cuda())

# Now runs with fused INT8 kernels — maximum throughput
```

> 🔥 **Pro Tip**: Always combine `quantize_` with `torch.compile`. The compiler can fuse dequantize + matmul into a single kernel, giving you the speed of quantized compute with no overhead.

## Sparsity

```python
from torchao.sparsity import sparsify_, semi_structured_sparsify

# 2:4 structured sparsity (50% sparse, hardware-accelerated on A100+)
sparsify_(model, semi_structured_sparsify())

# This zeroes out 2 of every 4 weights in a structured pattern
# NVIDIA Ampere+ GPUs have hardware support for 2:4 sparsity
# Result: ~2x speedup on matmuls with minimal accuracy loss

# Combine with quantization for maximum compression
quantize_(model, int8_weight_only())
sparsify_(model, semi_structured_sparsify())
model = torch.compile(model)
```

## Float8 Training (Transformer Engine style)

```python
from torchao.float8 import convert_to_float8_training

# Convert model to use float8 for matmuls during training
convert_to_float8_training(model)

# Training loop remains the same
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Benefits:
# - ~2x faster matmuls on H100 GPUs
# - Same convergence as BF16/FP16 training
# - Automatic scaling factors per tensor
```

## Benchmarking

```python
import torch
from torch.utils.benchmark import Timer

def benchmark_model(model, input_shape=(1, 3, 224, 224), device='cuda'):
    """Benchmark inference throughput"""
    x = torch.randn(*input_shape, device=device)
    model = model.eval().to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(x)

    # Benchmark
    timer = Timer(
        stmt='model(x)',
        globals={'model': model, 'x': x},
    )
    result = timer.blocked_autorange(min_run_time=5.0)
    print(f"Latency: {result.median * 1000:.2f} ms")
    print(f"Throughput: {1 / result.median:.1f} img/s")
    return result

# Compare original vs quantized
print("=== Original ===")
benchmark_model(original_model)

print("=== INT8 Weight-Only ===")
benchmark_model(int8_model)

print("=== INT8 + Compiled ===")
benchmark_model(compiled_int8_model)
```

## Common Patterns

```python
# --- Quantize specific layers only ---
from torchao.quantization import quantize_

def filter_fn(module, fqn):
    """Only quantize layers with enough parameters"""
    if isinstance(module, torch.nn.Linear):
        return module.in_features >= 256 and module.out_features >= 256
    return False

quantize_(model, int8_weight_only(), filter_fn=filter_fn)

# --- Export quantized model to ONNX ---
quantize_(model, int8_weight_only())
model = torch.compile(model)

# Export via torch.export
with torch.no_grad():
    exported = torch.export.export(model, (torch.randn(1, 3, 224, 224).cuda(),))

# --- Quantization-Aware Training (QAT) ---
from torchao.quantization import int8_weight_only

# 1. Train normally first
# 2. Fine-tune with fake quantization to recover accuracy
quantize_(model, int8_weight_only())  # apply after training
# or use QAT APIs for training-time quantization simulation

# --- Profile memory savings ---
import torch

def profile_memory(model, input_shape=(32, 3, 224, 224)):
    """Profile peak memory usage"""
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(*input_shape, device='cuda')
    with torch.no_grad():
        model(x)
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak memory: {peak_mb:.1f} MB")
    return peak_mb
```

## Prefer / Avoid

| ❌ Avoid | ✅ Prefer | Why |
|----------|-----------|-----|
| Manual quantization code | `quantize_()` one-liner | Less error-prone |
| Quantize without compile | `quantize_()` + `torch.compile()` | Missed kernel fusion |
| INT4 on small models | INT8 for models < 1B params | INT4 needs scale |
| Skip accuracy validation | Always benchmark before/after | Catch regressions |
| Quantize during training | Quantize trained model (PTQ) | Simpler, usually sufficient |
| Same quantization for all layers | `autoquant` per-layer | Better accuracy/speed |

---

[← Previous: Chapter 5 — TorchVision](./05_torchvision.md) · **Next: [Chapter 7 — TorchMetrics →](./07_torchmetrics.md)**

---
*Last updated: May 2026*
