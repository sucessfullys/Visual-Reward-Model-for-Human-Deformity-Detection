#!/usr/bin/env python3
"""Hand detection with ONNX Runtime; no ModelScope or EasyCV dependency."""

import argparse
from pathlib import Path

import cv2
import numpy as np
# Importing modern PyTorch first makes its CUDA 12/cuDNN 9 shared libraries
# visible to ONNX Runtime in environments where both packages are installed.
try:
    import torch  # noqa: F401
except ImportError:
    pass
import onnxruntime as ort


INPUT_SIZE = 640
MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess(image):
    height, width = image.shape[:2]
    ratio = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = int(width * ratio + 0.5)
    resized_height = int(height * ratio + 0.5)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114.0, dtype=np.float32)
    canvas[:resized_height, :resized_width] = resized
    canvas = (canvas - MEAN) / STD
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])
    return tensor, ratio


def nms(boxes, scores, threshold):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-7)
        order = rest[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def postprocess(predictions, ratio, image_shape, confidence, nms_threshold):
    predictions = predictions[0]
    scores = predictions[:, 4] * predictions[:, 5]
    selected = scores >= confidence
    predictions = predictions[selected]
    scores = scores[selected]
    if len(predictions) == 0:
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32)

    centers = predictions[:, :2]
    sizes = predictions[:, 2:4]
    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
    keep = nms(boxes, scores, nms_threshold)
    boxes = boxes[keep] / ratio
    scores = scores[keep]
    height, width = image_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes, scores


def choose_providers(requested):
    available = ort.get_available_providers()
    mapping = {
        "tensorrt": "TensorrtExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    if requested != "auto":
        provider = mapping[requested]
        if provider not in available:
            raise RuntimeError(f"{provider} unavailable; available providers: {available}")
        return [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [provider]
    # CUDA is the safe automatic choice. TensorRT is opt-in because an ORT
    # wheel may advertise its EP even when external TensorRT libraries are absent.
    return [
        provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]


def main():
    parser = argparse.ArgumentParser()
    detector_dir = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, default=detector_dir / "hand_yolox_pai_640.onnx")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--nms", type=float, default=0.65)
    parser.add_argument("--provider", choices=("auto", "tensorrt", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")
    tensor, ratio = preprocess(image)
    providers = choose_providers(args.provider)
    print(f"ONNX Runtime providers: {providers}")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.model), sess_options=session_options, providers=providers)
    predictions = session.run(["predictions"], {"images": tensor})[0]
    boxes, scores = postprocess(predictions, ratio, image.shape, args.threshold, args.nms)

    print(f"Detected {len(boxes)} hand(s):")
    for index, (box, score) in enumerate(zip(boxes, scores), 1):
        coords = [int(value) for value in box]
        print(f"  [{index}] bbox={coords}, score={score:.4f}")
        if args.output:
            x1, y1, x2, y2 = coords
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"hand {score:.2f}", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), image):
            raise RuntimeError(f"Failed to write: {args.output}")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
