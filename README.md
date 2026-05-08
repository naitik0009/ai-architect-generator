This is a great move. Updating your README.md now ensures that when you open this repo on your friend's M3 Max on Monday, you have a perfect "instruction manual" to follow.

I have updated the Quick Start section to include the missing datasets library, the split_data.py step, and the optimized training command for Apple Silicon.

Kalkulio AI Architect Generator
An AI-powered pipeline designed to generate valid, topologically correct JSON floor plans for the Kalkulio AI Challenge.

This project leverages Apple Silicon's native MLX framework to fine-tune an LLM (Qwen2.5-Coder) using LoRA, transforming raw conversational prompts into highly structured, geometrically strict JSON architectural representations.

🏗️ Architecture & Strategy
Because standard image generation models cannot output valid vector graphics or specific JSON structures with mathematically sound geometries, this project treats floor plan generation as a Sequence-to-Sequence (Structured Output) problem.

The pipeline consists of three main phases:

Data Augmentation: A Python script ingests raw JSON samples and applies geometric scaling and mirroring to synthetically expand the dataset.

LoRA Fine-Tuning: The augmented dataset is formatted into conversational .jsonl pairs and used to fine-tune Qwen2.5-Coder-3B-Instruct natively on M-series chips via mlx_lm.

Geometric Post-Processing: A deterministic algorithm (planned) to "snap" AI-hallucinated coordinates to a grid.

🚀 Quick Start (Apple Silicon)
1. Environment Setup
Clone the repository and create a local virtual environment.
Note: The virtual environment folder is not portable; recreate it if moving between different Macs.

Bash
git clone https://github.com/naitik0009/ai-architect-generator.git
cd ai-architect-generator
python3 -m venv kalkulio-env
source kalkulio-env/bin/activate
pip install mlx-lm datasets pandas huggingface_hub
2. Data Preparation
Place your master kalkulio_all.json file in the root directory. Run the split and preparation scripts to generate the training data.

Bash
# 1. Split master JSON into individual house files
python split_data.py

# 2. Augment and format for LLM training
python prepare_data.py
3. Fine-Tuning (MLX)
Use the following command to start training. For 16GB RAM (M4), use a lower max-seq-length. For 24GB+ RAM (M3 Max), use the higher limit shown below to include complex houses.

Bash
python -m mlx_lm.lora \
    --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \
    --data data \
    --train \
    --batch-size 1 \
    --max-seq-length 8192 \
    --iters 500 \
    --adapter-path kalkulio-adapters
4. Testing the Model
Once training is complete, test the generation using the generated adapters:

Bash
python -m mlx_lm.generate \
    --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \
    --adapter-path ./kalkulio-adapters \
    --max-tokens 4000 \
    --prompt "Generate a floor plan for a house with an approximate area of 120m2."
📁 Repository Structure
kalkulio_all.json: Master dataset (keep local, gitignored).

prepare_data.py: Script for scaling, mirroring, and token-saving (rounding/whitespace stripping).

split_data.py: Utility to split bulk JSON arrays into the raw_data folder.

kalkulio-adapters/: Local folder containing the trained LoRA weights.
