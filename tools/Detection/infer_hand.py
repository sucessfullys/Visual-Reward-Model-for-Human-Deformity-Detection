#!/usr/bin/env python3
"""手部检测推理 —— EasyCV DetectionPredictor，直接用。

用法:
    /root/miniconda3/envs/hand-detection/bin/python infer_hand.py --input /path/to/image.jpg
"""

import argparse
import json
import os
import sys
import tempfile

import cv2
import torch

MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/cv_yolox-pai_hand-detection"


def _decode_modelscope_value(value):
    """Convert ModelScope's JSON wrappers into plain EasyCV config values."""
    if isinstance(value, dict):
        if "__class__" in value and "__value__" in value:
            return _decode_modelscope_value(value["__value__"])
        return {
            key: _decode_modelscope_value(item)
            for key, item in value.items()
            if key != "__easycv_arch__"
        }
    if isinstance(value, list):
        return [_decode_modelscope_value(item) for item in value]
    return value


def make_easycv_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = _decode_modelscope_value(json.load(f))
    # ModelScope stores these under dataset.val; EasyCV PredictorV2 expects
    # them at the root of the config.
    val_config = config.get("dataset", {}).get("val", {})
    if "test_pipeline" not in config and "pipeline" in val_config:
        config["test_pipeline"] = val_config["pipeline"]
    data_source = val_config.get("data_source", {})
    if "CLASSES" not in config and "classes" in data_source:
        config["CLASSES"] = data_source["classes"]
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="hand_easycv_", delete=False,
        encoding="utf-8")
    json.dump(config, tmp, ensure_ascii=False)
    tmp.close()
    return tmp.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="输入图片路径")
    p.add_argument("--output", default=None, help="输出图片路径")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    # 直接用 EasyCV 的 DetectionPredictor
    from easycv.predictors import DetectionPredictor

    print(f"Loading model from {MODEL_DIR}")
    config_path = make_easycv_config(f"{MODEL_DIR}/configuration.json")
    try:
        predictor = DetectionPredictor(
            model_path=f"{MODEL_DIR}/pytorch_model.pt",
            config_file=config_path,
            device=args.device,
            score_threshold=args.threshold,
        )
    finally:
        os.unlink(config_path)

    print(f"Running inference on {args.input}")
    result = predictor(args.input)

    # Parse result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        result = result[0]
    if isinstance(result, dict):
        boxes = result.get("detection_boxes", [])
        scores = result.get("detection_scores", [])
        labels = result.get("detection_classes", result.get("detection_labels", []))
    else:
        raise TypeError(f"Unexpected EasyCV result type: {type(result).__name__}")

    print(f"\nDetected {len(boxes)} hand(s):")
    for i, box in enumerate(boxes):
        s = scores[i] if i < len(scores) else "?"
        x1, y1, x2, y2 = map(int, box) if len(box) == 4 else (0, 0, 0, 0)
        print(f"  [{i+1}] x1={x1} y1={y1} x2={x2} y2={y2}  score={s}")

    if args.output and len(boxes) > 0:
        img = cv2.imread(args.input)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            s = scores[i] if i < len(scores) else 0
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"hand {s:.2f}", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(args.output, img)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
