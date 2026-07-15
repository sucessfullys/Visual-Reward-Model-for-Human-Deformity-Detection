#!/usr/bin/env python3
"""手部检测 —— 只保留置信度最高的一个框，鲜艳红色加粗。"""

import argparse, random
from pathlib import Path
import cv2, numpy as np
try: import torch  # noqa: F401
except ImportError: pass
import onnxruntime as ort

INPUT_SIZE = 640
MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess(image):
    h, w = image.shape[:2]
    r = min(INPUT_SIZE / w, INPUT_SIZE / h)
    rw, rh = int(w * r + 0.5), int(h * r + 0.5)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114.0, dtype=np.float32)
    canvas[:rh, :rw] = resized
    canvas = (canvas - MEAN) / STD
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), r


def nms(boxes, scores, thre):
    if len(boxes) == 0: return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        cur = order[0]; keep.append(cur)
        if order.size == 1: break
        rest = order[1:]
        xx1, yy1 = np.maximum(x1[cur], x1[rest]), np.maximum(y1[cur], y1[rest])
        xx2, yy2 = np.minimum(x2[cur], x2[rest]), np.minimum(y2[cur], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[cur] + areas[rest] - inter, 1e-7)
        order = rest[iou <= thre]
    return np.asarray(keep, dtype=np.int64)


def postprocess(preds, ratio, img_shape, conf, nms_thre):
    preds = preds[0]
    scores = preds[:, 4] * preds[:, 5]
    sel = scores >= conf; preds, scores = preds[sel], scores[sel]
    if len(preds) == 0: return np.empty((0, 4)), np.empty((0,))
    centers, sizes = preds[:, :2], preds[:, 2:4]
    boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
    keep = nms(boxes, scores, nms_thre)
    boxes, scores = boxes[keep] / ratio, scores[keep]
    h, w = img_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes, scores


def pick_best(boxes, scores):
    """只保留最高置信度框，并列最高则随机选一个。"""
    if len(boxes) == 0: return boxes, scores
    max_score = scores.max()
    best_idx = [i for i, s in enumerate(scores) if s == max_score]
    idx = random.choice(best_idx)
    return boxes[idx:idx + 1], scores[idx:idx + 1]


def main():
    p = argparse.ArgumentParser()
    d = Path(__file__).resolve().parent
    p.add_argument("--model", type=Path, default=d / "hand_yolox_pai_640.onnx")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--nms", type=float, default=0.65)
    p.add_argument("--provider", choices=("auto", "tensorrt", "cuda", "cpu"), default="auto")
    args = p.parse_args()

    img = cv2.imread(str(args.input))
    if img is None: raise FileNotFoundError(str(args.input))
    tensor, ratio = preprocess(img)

    prov_map = {"tensorrt": "TensorrtExecutionProvider", "cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}
    available = ort.get_available_providers()
    if args.provider == "auto":
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    else:
        ep = prov_map[args.provider]
        if ep not in available: raise RuntimeError(f"{ep} not in {available}")
        providers = [ep, "CPUExecutionProvider"] if ep != "CPUExecutionProvider" else [ep]

    session = ort.InferenceSession(str(args.model), providers=providers)
    preds = session.run(["predictions"], {"images": tensor})[0]
    boxes, scores = postprocess(preds, ratio, img.shape, args.threshold, args.nms)

    # --- 只保留最好的 ---
    boxes, scores = pick_best(boxes, scores)

    print(f"Detected {len(boxes)} hand(s) (best only):")
    for i, (box, score) in enumerate(zip(boxes, scores), 1):
        x1, y1, x2, y2 = map(int, box)
        print(f"  [{i}] x1={x1} y1={y1} x2={x2} y2={y2} score={score:.4f}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 6)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), img)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
