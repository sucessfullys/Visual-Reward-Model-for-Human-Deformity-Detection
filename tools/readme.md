### 🧬 Project Introduction

This project is an advanced generative AI workflow designed to synthesize human imagery featuring specific anatomical anomalies or deformities. By integrating computer vision with Large Language Models, the system achieves precise control over generation targets and semantic optimization.

#### Core Workflow

1. **Precise Localization**: Utilizing state-of-the-art **Detection and Segmentation models**, the system analyzes input data to accurately locate and segment specific body parts.
2. **Intelligent Prompt Engineering**: We leverage the **Gemma** LLM as an intelligent prompt engineer. Based on the localized body parts, Gemma automatically generates, refines, and optimizes descriptive prompts to ensure semantic accuracy and descriptive depth.
3. **High-Fidelity Generation**: The optimized prompts are fed into the **Flux2** model. Flux2 then executes either **inpainting** or **full generation** to synthesize high-quality images of deformed or anatomically variant human figures based on the specific instructions.

We plan to release the detailed data construction pipeline and technical specifications.



