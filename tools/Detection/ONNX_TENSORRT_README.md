# Hand detector: ONNX / CUDA / TensorRT

This directory contains a portable export of the EasyCV YOLOX-PAI hand detector.
Runtime inference no longer imports ModelScope, EasyCV, MMCV, or the legacy PyTorch
1.10 environment.

## Files

- `hand_yolox_pai_640.onnx`: exported fixed-shape ONNX model, input `images`
  is float32 NCHW `1x3x640x640`; output `predictions` is `1x8400x6`.
- `infer_hand_onnx.py`: preprocessing, ONNX Runtime inference, confidence filter,
  NumPy NMS, coordinate restoration, and visualization.
- `export_hand_onnx.py`: reproducible CPU exporter using the legacy
  `hand-detection` environment only during export.
- `build_hand_tensorrt.sh`: optional `trtexec` FP16 engine builder.

ONNX SHA-256:

```text
8eed3890c190367fb9cd75c974c0d8a49e6d54f0e3df6bf2f30f6b5166e80c2e
```

## Verified result

The bundled test image produces the same five detections with EasyCV and ONNX
Runtime. Bounding boxes and confidence scores match to displayed precision.

## Modern CUDA 12.9 environment

The `dreamface-omni` environment has been prepared with:

```text
onnxruntime-gpu==1.23.2
opencv-python-headless==5.0.0.93
```

Keep ONNX Runtime pinned to 1.23.2. Version 1.27 requires CUDA 13 and fails with
`libcudart.so.13` in the current CUDA 12.9 environment.

Run on CUDA device 0:

```bash
cd /mnt/image-edit/datasets/duanyufa
CUDA_VISIBLE_DEVICES=0 /root/.venv/dreamface-omni/bin/python \
  task_shengsheng/Detection/infer_hand_onnx.py \
  --provider cuda \
  --input task_shengsheng/Detection/cv_yolox-pai_hand-detection/resources/1.jpg \
  --output /tmp/hand_cuda.jpg \
  --threshold 0.3
```

Use `--provider auto` to prefer CUDA with CPU fallback, or `--provider cpu` for a
GPU-independent check.

## Re-export ONNX

Only re-export if the checkpoint or EasyCV model configuration changes:

```bash
cd /mnt/image-edit/datasets/duanyufa
CUDA_VISIBLE_DEVICES="" conda run -n hand-detection python \
  task_shengsheng/Detection/export_hand_onnx.py
```

## Optional TensorRT

TensorRT and `trtexec` are not currently installed. After installing a TensorRT
release compatible with the host CUDA/GPU, build an FP16 engine with:

```bash
bash task_shengsheng/Detection/build_hand_tensorrt.sh
```

The generated engine is hardware- and TensorRT-version-specific. Alternatively,
when the TensorRT shared libraries are installed, use ONNX Runtime's TensorRT EP:

```bash
CUDA_VISIBLE_DEVICES=0 /root/.venv/dreamface-omni/bin/python \
  task_shengsheng/Detection/infer_hand_onnx.py \
  --provider tensorrt \
  --input task_shengsheng/Detection/cv_yolox-pai_hand-detection/resources/1.jpg \
  --output /tmp/hand_tensorrt.jpg
```

## CPU validation

```bash
CUDA_VISIBLE_DEVICES="" /root/.venv/dreamface-omni/bin/python \
  task_shengsheng/Detection/infer_hand_onnx.py \
  --provider cpu \
  --input task_shengsheng/Detection/cv_yolox-pai_hand-detection/resources/1.jpg \
  --output /tmp/hand_cpu.jpg
```
