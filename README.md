# Visual Reward Model for Human Deformity Detection

High-precision Reward Model (RM) for evaluating and detecting human body deformities. This project is built on top of ms-swift and is optimized for RLHF and visual alignment tasks.

## 📰 News

- **[2026.7.13]**: 🤗  Release SFT training code
- **[xxxx.x.xx]**: 🔥  We are releasing the human body deformities dataset

## 📋 Table of Contents

1. [🛠️ Installation](#%EF%B8%8F-installation)

## 🛠️ Installation

To install using pip:

```shell
git clone https://github.com/sucessfullys/Visual-Reward-Model-for-Human-Deformity-Detection.git
cd Visual-Reward-Model-for-Human-Deformity-Detection

# The main branch is for swift 4.x. To install swift 3.x, please run the following command:
# git checkout release/3.12
pip install -e .

# Using uv
uv pip install -e . --torch-backend=auto
```

Running Environment:

|              | Range        | Recommended    | Notes                                     |
|--------------|--------------|----------------|-------------------------------------------|
| python       | >=3.10       | 3.12           |                                           |
| cuda         |              | cuda12.8/13.0  | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0        | 2.8.0/2.11.0   |                                           |
| transformers | >=4.33       | 4.57.6/5.12.1  |                                           |
| modelscope   | >=1.23       |                |                                           |
| datasets     | >=3.0,<4.8.5 | 3.6.0/4.8.4    |                                           |
| peft         | >=0.11,<0.20 |                |                                           |
| flash_attn   |              | 2.8.3/4.0.0b15 |                                           |
| trl          | >=0.15,<1.0  | 0.29.1         | RLHF                                      |
| deepspeed    | >=0.14       | 0.18.9         | Training                                  |
| vllm         | >=0.5.1      | 0.11.0/0.23.0  | Inference/Deployment                      |
| sglang       | >=0.4.6      |                | Inference/Deployment                      |
| evalscope    | >=1.0        |                | Evaluation                                |
| gradio       |              | 5.32.1         | Web-UI/App                                |
