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
xxx (暂未公开)
```

`sample_body_deformity.jsonl` is only for smoke testing. Replace it with your real
JSONL when the HumanRefiner annotations are ready.

## Smoke Training

Activate your `ms-swift` environment, then run:

```bash
cd xxx
bash xxx/smoke_sft.sh
```

The LoRA script defaults to:

- model: `xxx/Qwen3-VL-8B-Instruct`
- dataset: `xxx/sample_body_deformity.jsonl`
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
bash xxx/smoke_sft.sh
```

For a one-step smoke test, set `MAX_STEPS=1` explicitly:

```bash
MAX_STEPS=1 \
bash xxx/smoke_sft.sh
```

Enable DeepSpeed when using multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DEEPSPEED=zero2 \
bash xxx/smoke_sft.sh
```

## Full SFT

If you want SFT without LoRA adapters, use:

```bash
cd xxx/ms-swift
bash xxx/smoke_sft_full.sh
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
bash xxx/smoke_sft_full.sh
```

If memory is still tight, switch to `DEEPSPEED=zero3`. ZeRO-3 saves more memory
by sharding parameters too, but it usually has more communication overhead.

## Inference After Training

After a checkpoint is saved, run:

```bash
ADAPTER_PATH=xxx/vx-xxx/checkpoint-xxx \
VAL_DATASET_PATH=xxx/sample_body_deformity.jsonl \
bash xxx/infer_adapter.sh
```
