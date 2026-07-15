#!/usr/bin/env python3
"""Export the EasyCV YOLOX-PAI hand detector to a portable ONNX model."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from task_shengsheng.models.Detection.infer_hand import make_easycv_config


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        # EasyCV's compression path returns decoded YOLOX predictions without
        # Python-side NMS: [batch, 8400, 6] for one class at 640x640.
        return self.model(images, mode="compression")


def main():
    parser = argparse.ArgumentParser()
    detector_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--model-dir", type=Path,
        default=detector_dir / "cv_yolox-pai_hand-detection")
    parser.add_argument(
        "--output", type=Path,
        default=detector_dir / "hand_yolox_pai_640.onnx")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    from easycv.predictors import DetectionPredictor

    config_path = make_easycv_config(str(args.model_dir / "configuration.json"))
    try:
        predictor = DetectionPredictor(
            model_path=str(args.model_dir / "pytorch_model.pt"),
            config_file=config_path,
            device="cpu",
            score_threshold=0.01,
            input_processor_threads=1,
        )
    finally:
        os.unlink(config_path)

    model = ExportWrapper(predictor.model.eval()).cpu()
    dummy = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        reference = model(dummy)
        print(f"PyTorch output shape: {tuple(reference.shape)}")
        torch.onnx.export(
            model,
            dummy,
            str(args.output),
            input_names=["images"],
            output_names=["predictions"],
            opset_version=args.opset,
            do_constant_folding=True,
        )

    try:
        import onnx
        exported = onnx.load(str(args.output))
        onnx.checker.check_model(exported)
        print("ONNX checker: passed")
    except ImportError:
        print("ONNX checker skipped: install the 'onnx' package to enable it")

    print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()
