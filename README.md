### Kalkulio AI Architect Generator
An AI-powered pipeline designed to generate valid, topologically correct JSON floor plans for the Kalkulio AI Challenge.

This project leverages Apple Silicon's native MLX framework to fine-tune an LLM (Qwen2.5-Coder) using LoRA, transforming raw conversational prompts into highly structured, geometrically strict JSON architectural representations.

This project leverages the **HuggingFace & PyTorch ecosystem (CUDA)** to fine-tune an LLM (Qwen2.5-Coder) using LoRA, transforming raw conversational prompts into highly structured, geometrically strict JSON architectural representations. It is perfectly optimized for an **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition**.

The pipeline consists of three main phases:

Data Augmentation: A Python script ingests raw JSON samples and applies geometric scaling and mirroring to synthetically expand the dataset.

The pipeline consists of three main phases:
1. **Data Augmentation:** A Python script ingests 70 initial parsed JSON samples from Kalkulio and applies geometric scaling (and other augmentations) to synthetically expand the dataset.
2. **LoRA Fine-Tuning:** The augmented dataset is formatted into conversational `.jsonl` pairs and used to fine-tune `Qwen2.5-Coder-3B-Instruct` on NVIDIA GPUs via `transformers`, `peft`, and `trl`.
3. **Geometric Post-Processing (Coming Soon):** A deterministic Python algorithm to "snap" AI-hallucinated coordinates to a grid, ensuring closed polygons and perfectly connected walls.

## 🚀 Quick Start (NVIDIA GPUs / CUDA)

This project requires Python 3.10+ and a CUDA-capable NVIDIA GPU (e.g., RTX 6000 series).

### Bash
git clone https://github.com/naitik0009/ai-architect-generator.git
cd ai-architect-generator
python3 -m venv kalkulio-env
source kalkulio-env/bin/activate

# Install dependencies (PyTorch, Transformers, TRL, etc.)
pip install -r requirements.txt
```

### 2. Prepare the Data
Run the data preparation script to augment and format the Kalkulio JSON data into conversational `.jsonl` formats:
```bash
python prepare_data.py
```
This will create `data/train.jsonl` and `data/valid.jsonl`.

### 3. Start Training
Launch the training script. The script is configured to use `bfloat16` and Flash Attention 2 (if available) for maximum efficiency on Blackwell GPUs:
```bash
python train.py
```
The final model adapter will be saved in `./qwen-kalkulio-lora/final`.
