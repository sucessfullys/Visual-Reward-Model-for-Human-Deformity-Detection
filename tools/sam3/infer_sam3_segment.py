#!/usr/bin/env python3
"""SAM3 文字驱动分割 —— 输入图片 + 文字描述（手臂/手/头/脸...），输出分割结果。

用法:
    /root/miniconda3/envs/sam3/bin/python infer_sam3_segment.py \
      --input /path/to/img.jpg --prompt "hand" --output result.png

    # By default this script uses the local checkpoint downloaded under hf_sam3.1:
    /root/miniconda3/envs/sam3/bin/python infer_sam3_segment.py \
      --input /path/to/img.jpg --prompt "hand" --output result.png
"""

import argparse
import os
import numpy as np
from PIL import Image
import torch

# SAM3 benefits from TF32 on supported CUDA GPUs. Mixed precision is entered
# only around inference below rather than leaking a global autocast context.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import download_ckpt_from_hf
from huggingface_hub.errors import GatedRepoError

HERE = os.path.dirname(os.path.abspath(__file__))
BPE_PATH = os.path.join(os.path.dirname(sam3.__file__), "assets/bpe_simple_vocab_16e6.txt.gz")
DEFAULT_CHECKPOINT_PATH = os.path.join(HERE, "hf_sam3.1", "sam3.1_multiplex.pt")

MODEL = None
MODEL_VERSION = None
MODEL_CHECKPOINT_PATH = None


def get_model(device, version, checkpoint_path=None):
    global MODEL, MODEL_VERSION, MODEL_CHECKPOINT_PATH
    checkpoint_key = os.path.abspath(checkpoint_path) if checkpoint_path else None
    if MODEL is None or MODEL_VERSION != version or MODEL_CHECKPOINT_PATH != checkpoint_key:
        print(f"Loading SAM3 model (version={version})...")
        if checkpoint_path:
            ckpt_path = checkpoint_key
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        else:
            print("No --checkpoint-path provided; downloading checkpoint from Hugging Face...")
            try:
                ckpt_path = download_ckpt_from_hf(version=version)
            except GatedRepoError as exc:
                repo_id = "facebook/sam3.1" if version == "sam3.1" else "facebook/sam3"
                raise RuntimeError(
                    "\n".join(
                        [
                            f"Cannot download {repo_id}: the Hugging Face repo is gated.",
                            "Fix options:",
                            f"  1. Request/accept access at https://huggingface.co/{repo_id}",
                            "  2. Login in this environment: hf auth login",
                            "  3. Or pass a local checkpoint with --checkpoint-path /path/to/checkpoint.pt",
                        ]
                    )
                ) from exc
        print(f"Checkpoint: {ckpt_path}")
        MODEL = build_sam3_image_model(
            bpe_path=BPE_PATH,
            checkpoint_path=ckpt_path,
            load_from_HF=False,
            # The upstream builder only recognizes the exact string "cuda";
            # build on CPU first, then honor cuda:N explicitly below.
            device="cpu",
        )
        MODEL = MODEL.to(torch.device(device))
        MODEL.eval()
        MODEL_VERSION = version
        MODEL_CHECKPOINT_PATH = checkpoint_key
        print(f"Model ready (v={version}).")
    return MODEL


def segment(image_path, prompt, device, version, threshold=0.5, instance_mode="all", checkpoint_path=None):
    model = get_model(device, version, checkpoint_path)
    img = Image.open(image_path).convert("RGB")

    processor = Sam3Processor(
        model, device=device, confidence_threshold=threshold)
    use_cuda_amp = str(device).startswith("cuda")
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_amp
    ):
        state = processor.set_image(img)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)

    # Official Sam3Processor output is a dict containing tensors:
    # masks [N, 1, H, W], boxes [N, 4], scores [N].
    masks = state.get("masks")
    scores = state.get("scores")
    boxes = state.get("boxes")
    if masks is None or scores is None or masks.shape[0] == 0:
        print(f"No objects found for prompt: '{prompt}'")
        return img  # 返回原图

    if instance_mode == "best":
        selected = [int(torch.argmax(scores).item())]
    else:
        selected = list(range(masks.shape[0]))

    selected_masks = masks[selected].squeeze(1).detach().cpu().numpy().astype(bool)
    mask = np.any(selected_masks, axis=0)
    score_values = scores.detach().float().cpu().numpy()
    print(f"Found {masks.shape[0]} object(s); rendering {len(selected)} ({instance_mode}).")
    for rank, index in enumerate(selected, 1):
        message = f"  [{rank}] score={score_values[index]:.3f}"
        if boxes is not None:
            box = boxes[index].detach().float().cpu().tolist()
            message += f", box={[round(value, 1) for value in box]}"
        print(message)

    # 生成分割结果：原图 + mask 叠加（半透明红色）
    img_np = np.array(img).astype(np.float32)
    if mask.shape[:2] != img_np.shape[:2]:
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (img.width, img.height), Image.NEAREST)) > 128

    overlay = img_np.copy()
    overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5  # 红色半透明
    result = Image.fromarray(overlay.astype(np.uint8))

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", required=True, help="文字提示，如 hand/arm/head/face/person")
    p.add_argument("--output", default="segmented.png")
    p.add_argument("--threshold", type=float, default=0.5, help="置信度阈值")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--version", default="sam3.1", choices=("sam3", "sam3.1"),
                   help="模型版本，sam3.1 效果更好（默认）")
    p.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH,
                   help=f"本地 SAM3 checkpoint 路径；默认：{DEFAULT_CHECKPOINT_PATH}")
    p.add_argument("--instance-mode", default="all", choices=("all", "best"),
                   help="all 合并所有匹配实例；best 仅保留最高分实例")
    args = p.parse_args()

    result = segment(
        args.input, args.prompt, args.device, args.version,
        args.threshold, args.instance_mode, args.checkpoint_path)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    result.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
