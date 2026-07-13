# Qwen3-VL Body Deformity SFT Smoke Test

This directory contains a minimal multimodal SFT setup for body deformity detection.

The training objective is:

- input: one image plus the user question
- output: visible evidence inside `<think>...</think>`
- final label: `<conclusion>normal</conclusion>` or `<conclusion>abnormal</conclusion>`

## Dataset Format

MS-SWIFT's stable multimodal format keeps the image placeholder in `messages[*].content`
and stores the actual image path in the top-level `images` field:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是人体畸形检测助手。判断图中人体是否存在结构畸形，并给出可见证据。最后用 <conclusion>normal</conclusion> 或 <conclusion>abnormal</conclusion> 输出结论。"
    },
    {
      "role": "user",
      "content": "<image>这张图片是否存在人体畸形？请说明理由。"
    },
    {
      "role": "assistant",
      "content": "<think>图中手部比例异常，手指数量或形态不自然，局部结构与正常人体解剖不一致。</think>\n<conclusion>abnormal</conclusion>"
    }
  ],
  "images": ["/absolute/path/to/image.jpg"]
}
```

`sample_body_deformity.jsonl` is only for smoke testing. Replace it with your real
JSONL when the HumanRefiner annotations are ready.

## Smoke Training

Activate your `ms-swift` environment, then run:

```bash
cd /mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift
bash /mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/smoke_sft.sh
```

The LoRA script defaults to:

- model: `/mnt/image-edit/datasets/duanyufa/models/Qwen3-VL-8B-Instruct`
- dataset: `examples/train/body_deformity_qwen3_vl/sample_body_deformity.jsonl`
- strategy: SFT + LoRA on the LLM part
- frozen modules: ViT and aligner/projector
- epochs: `NUM_TRAIN_EPOCHS=1`
- max steps: `MAX_STEPS=-1`, meaning epoch-based training

You can override the defaults:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MAX_STEPS=5 \
DATASET_PATH=/path/to/your_real_dataset.jsonl \
bash /mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/smoke_sft.sh
```

For a one-step smoke test, set `MAX_STEPS=1` explicitly:

```bash
MAX_STEPS=1 \
bash /mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/smoke_sft.sh
```

Enable DeepSpeed when using multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DEEPSPEED=zero2 \
bash /mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/smoke_sft.sh
```

## Full SFT

If you want SFT without LoRA adapters, use:

```bash
cd /mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift
bash /mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/smoke_sft_full.sh
```

This uses `--tuner_type full`, freezes the ViT and aligner, and trains the LLM
weights directly. It needs much more GPU memory than the LoRA smoke test.

For 8x H100 full SFT, start with ZeRO-2:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DEEPSPEED=zero2 \
IMAGE_MAX_TOKEN_NUM=1024 \
MAX_LENGTH=2048 \
NUM_TRAIN_EPOCHS=1 \
MAX_STEPS=-1 \
GRADIENT_ACCUMULATION_STEPS=2 \
LEARNING_RATE=1e-5 \
LR_SCHEDULER_TYPE=cosine \
WARMUP_RATIO=0.05 \
WEIGHT_DECAY=0.1 \
SAVE_STEPS=100 \
EVAL_STRATEGY=no \
bash /mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/smoke_sft_full.sh
```

If memory is still tight, switch to `DEEPSPEED=zero3`. ZeRO-3 saves more memory
by sharding parameters too, but it usually has more communication overhead.

## Inference After Training

After a checkpoint is saved, run:

```bash
ADAPTER_PATH=/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/output/body_deformity_qwen3_vl_smoke/vx-xxx/checkpoint-xxx \
VAL_DATASET_PATH=/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/sample_body_deformity.jsonl \
bash /mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/infer_adapter.sh
```
