---
title: "Chapter 11 — TensorRT"
---

[← Back to Table of Contents](./README.md)

# TensorRT — Maximum Inference Performance

> *"When milliseconds matter, TensorRT delivers."*

## Installation

```bash
# Via pip (NVIDIA provides wheels)
pip install tensorrt
pip install torch-tensorrt  # PyTorch integration

# Or via NVIDIA container:
# docker pull nvcr.io/nvidia/tensorrt:24.05-py3
```

## Import Convention

```python
import tensorrt as trt
import torch_tensorrt
import numpy as np
```

## Quick Start — PyTorch to TensorRT

```python
import torch
import torch_tensorrt
from torchvision.models import resnet50, ResNet50_Weights

# Load PyTorch model
model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().cuda()

# Compile with Torch-TensorRT (easiest path)
inputs = [torch_tensorrt.Input(
    min_shape=(1, 3, 224, 224),
    opt_shape=(8, 3, 224, 224),    # optimal batch size
    max_shape=(32, 3, 224, 224),   # maximum batch size
    dtype=torch.float16,
)]

trt_model = torch_tensorrt.compile(model,
    inputs=inputs,
    enabled_precisions={torch.float16},   # FP16 inference
    workspace_size=1 << 30,                # 1GB workspace
)

# Inference (same as PyTorch!)
x = torch.randn(8, 3, 224, 224).cuda().half()
with torch.no_grad():
    output = trt_model(x)     # [8, 1000] — accelerated!

# Save compiled model
torch.jit.save(trt_model, 'resnet50_trt.ts')
# Load: trt_model = torch.jit.load('resnet50_trt.ts')
```

> 🔥 **Pro Tip**: `torch_tensorrt.compile()` is the easiest way to get TensorRT speed. It handles the entire ONNX→TRT pipeline internally.

## ONNX → TensorRT Engine (Production Path)

```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=True, max_batch_size=32,
                 max_workspace_size=1 << 30):
    """Build TensorRT engine from ONNX model"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX")

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name,
        min=(1, 3, 224, 224),
        opt=(max_batch_size // 2, 3, 224, 224),
        max=(max_batch_size, 3, 224, 224),
    )
    config.add_optimization_profile(profile)

    # Build engine (this takes minutes!)
    print("Building TensorRT engine... (this may take several minutes)")
    serialized_engine = builder.build_serialized_network(network, config)

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"Engine saved to {engine_path}")
    return serialized_engine

# Build
build_engine('model.onnx', 'model.engine', fp16=True)
```

> ⚠️ **Pitfall**: TensorRT engines are NOT portable. An engine built on one GPU (e.g., A100) will NOT work on another (e.g., T4). Always rebuild for target hardware.

## Inference with TensorRT Engine

```python
import tensorrt as trt
import numpy as np
from cuda import cudart  # pip install cuda-python

class TRTInference:
    """Production TensorRT inference wrapper"""

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)

        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cudart.cudaStreamCreate()[1]

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = np.prod(shape) * np.dtype(dtype).itemsize

            # Allocate device memory
            _, device_mem = cudart.cudaMalloc(size)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'mem': device_mem, 'shape': shape, 'dtype': dtype})
            else:
                self.outputs.append({'name': name, 'mem': device_mem, 'shape': shape, 'dtype': dtype})

    def infer(self, input_data):
        """Run inference on numpy array"""
        # Copy input to device
        cudart.cudaMemcpyAsync(
            self.inputs[0]['mem'],
            input_data.ctypes.data,
            input_data.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )

        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], inp['mem'])
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], out['mem'])

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream)

        # Copy output back
        output = np.empty(self.outputs[0]['shape'], dtype=self.outputs[0]['dtype'])
        cudart.cudaMemcpyAsync(
            output.ctypes.data,
            self.outputs[0]['mem'],
            output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        )
        cudart.cudaStreamSynchronize(self.stream)
        return output

    def __del__(self):
        cudart.cudaStreamDestroy(self.stream)
        for buf in self.inputs + self.outputs:
            cudart.cudaFree(buf['mem'])

# Usage
trt_infer = TRTInference('model.engine')
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = trt_infer.infer(input_data)    # [1, 1000]
```

## INT8 Calibration

```python
import tensorrt as trt
import numpy as np

class CalibrationDataset:
    """Calibration dataset for INT8 quantization"""

    def __init__(self, data_dir, batch_size=32, num_batches=100):
        self.batch_size = batch_size
        self.num_batches = num_batches
        # Load representative data
        self.data = self._load_calibration_data(data_dir)
        self.current_batch = 0

    def _load_calibration_data(self, data_dir):
        # Load ~100 representative batches from your dataset
        import cv2
        from pathlib import Path
        images = []
        for path in sorted(Path(data_dir).glob('*.jpg'))[:self.batch_size * self.num_batches]:
            img = cv2.imread(str(path))
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC → CHW
            images.append(img)
        return np.array(images)

class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator using entropy calibration"""

    def __init__(self, dataset, cache_file='calibration.cache'):
        super().__init__()
        self.dataset = dataset
        self.cache_file = cache_file
        self.current_batch = 0
        self.batch_size = dataset.batch_size

        # Allocate device buffer
        self.device_input = cudart.cudaMalloc(
            self.batch_size * 3 * 224 * 224 * 4  # float32
        )[1]

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_batch >= self.dataset.num_batches:
            return None

        batch = self.dataset.data[
            self.current_batch * self.batch_size:
            (self.current_batch + 1) * self.batch_size
        ].astype(np.float32)

        cudart.cudaMemcpy(
            self.device_input, batch.ctypes.data,
            batch.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        self.current_batch += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        import os
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

# Build INT8 engine
def build_int8_engine(onnx_path, engine_path, calibration_data_dir):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # fallback for unsupported layers

    dataset = CalibrationDataset(calibration_data_dir)
    config.int8_calibrator = INT8Calibrator(dataset)

    engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine)
```

> 🔥 **Pro Tip**: INT8 gives ~2x speedup over FP16 on Turing+ GPUs. Use 100-500 representative calibration images for best accuracy.

## PyTorch Export to ONNX

```python
import torch
import torch.onnx

def export_to_onnx(model, onnx_path, input_shape=(1, 3, 224, 224), dynamic_batch=True):
    """Export PyTorch model to ONNX for TensorRT"""
    model.eval().cuda()
    dummy_input = torch.randn(*input_shape).cuda()

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported to {onnx_path}")

    # Verify
    import onnx
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)

    # Simplify (optional but recommended)
    import onnxsim
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "Simplification failed"
    onnx.save(model_onnx, onnx_path)
    print("ONNX model simplified")
```

## Benchmarking

```python
import time
import numpy as np

def benchmark_trt(engine_path, input_shape=(1, 3, 224, 224),
                  warmup=50, iterations=1000):
    """Benchmark TensorRT engine latency and throughput"""
    trt_infer = TRTInference(engine_path)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        trt_infer.infer(input_data)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        trt_infer.infer(input_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times = np.array(times)
    print(f"Latency — mean: {times.mean():.2f}ms, "
          f"p50: {np.percentile(times, 50):.2f}ms, "
          f"p95: {np.percentile(times, 95):.2f}ms, "
          f"p99: {np.percentile(times, 99):.2f}ms")
    print(f"Throughput: {1000 / times.mean() * input_shape[0]:.1f} img/s")

# Compare PyTorch vs TensorRT
print("=== PyTorch FP32 ===")
benchmark_pytorch(model, input_shape=(1, 3, 224, 224))
print("=== TensorRT FP16 ===")
benchmark_trt('model_fp16.engine', input_shape=(1, 3, 224, 224))
print("=== TensorRT INT8 ===")
benchmark_trt('model_int8.engine', input_shape=(1, 3, 224, 224))
```

## Common Patterns

```python
# --- YOLO + TensorRT (easiest path) ---
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.export(format='engine', half=True, imgsz=640, device=0)

# Inference with TRT engine (same Ultralytics API)
trt_model = YOLO('yolo11n.engine')
results = trt_model.predict('image.jpg')

# --- Dynamic shape handling ---
# When building engine with dynamic shapes, set input shape before inference:
context.set_input_shape('input', (batch_size, 3, 224, 224))

# --- Multi-stream inference (maximize GPU utilization) ---
import threading

def multi_stream_inference(engine_path, batches, num_streams=2):
    """Run multiple inference streams in parallel"""
    inferencers = [TRTInference(engine_path) for _ in range(num_streams)]

    def infer_stream(inferencer, data):
        return inferencer.infer(data)

    threads = []
    results = [None] * len(batches)
    for i, batch in enumerate(batches):
        stream_idx = i % num_streams
        t = threading.Thread(target=lambda idx, b: results.__setitem__(idx, inferencers[stream_idx].infer(b)),
                           args=(i, batch))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    return results

# --- Profile layer-by-layer ---
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
# Then use trtexec for profiling:
# trtexec --onnx=model.onnx --fp16 --verbose --dumpProfile
```

## Prefer / Avoid

| ❌ Avoid | ✅ Prefer | Why |
|----------|-----------|-----|
| FP32 TensorRT | FP16 minimum | 2x faster, minimal accuracy loss |
| Static batch only | Dynamic shapes with profile | Flexibility in production |
| Skip ONNX simplify | Always `onnxsim.simplify()` | Better TRT optimization |
| One engine for all GPUs | Build per target GPU | Not portable across architectures |
| Ignoring warmup | 50+ warmup iterations | First runs include JIT overhead |
| Manual TRT API for YOLO | `ultralytics export format=engine` | Much simpler, same speed |
| CPU preprocessing | GPU preprocessing pipeline | End-to-end optimization |

---

[← Previous: Chapter 10 — Ultralytics](./10_ultralytics.md) · [← Back to Table of Contents](./README.md)

---
*Last updated: May 2026*
