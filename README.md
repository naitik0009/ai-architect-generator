### Kalkulio AI Architect Generator
An AI-powered pipeline designed to generate valid, topologically correct JSON floor plans for the Kalkulio AI Challenge.

This project leverages Apple Silicon's native MLX framework to fine-tune an LLM (Qwen2.5-Coder) using LoRA, transforming raw conversational prompts into highly structured, geometrically strict JSON architectural representations.

### 🏗️ Architecture & Strategy
Because standard image generation models cannot output valid vector graphics or specific JSON structures with mathematically sound geometries, this project treats floor plan generation as a Sequence-to-Sequence (Structured Output) problem.

The pipeline consists of three main phases:

Data Augmentation: A Python script ingests raw JSON samples and applies geometric scaling and mirroring to synthetically expand the dataset.

LoRA Fine-Tuning: The augmented dataset is formatted into conversational .jsonl pairs and used to fine-tune Qwen2.5-Coder-7B-Instruct-4bit natively on M-series Max chips via mlx_lm.

Geometric Post-Processing: A deterministic algorithm (planned) to "snap" AI-hallucinated coordinates to a grid.

### 🚀 Quick Start (Apple Silicon - 32GB M3 Max Optimized)
1. Environment Setup
Clone the repository and create a local virtual environment.
Note: The virtual environment folder is not portable; recreate it fresh on the new Mac.

### Bash
git clone https://github.com/naitik0009/ai-architect-generator.git
cd ai-architect-generator
python3 -m venv kalkulio-env
source kalkulio-env/bin/activate
pip install mlx-lm datasets pandas huggingface_hub
2. Data Preparation
AirDrop or copy your master kalkulio_all.json file into the root directory of the project. Run the split and preparation scripts to generate the training data.

### Bash
### 1. Split master JSON into individual house files
python split_data.py

### 2. Augment and format for LLM training
python prepare_data.py

3. Fine-Tuning (MLX)
Use the command below to launch the training process. This utilizes the highly intelligent 7B model and sets the sequence buffer to 9200 to capture your most complex outlier floor plans completely unclipped.

### Bash
python -m mlx_lm lora \
    --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
    --data data \
    --train \
    --batch-size 1 \
    --max-seq-length 9200 \
    --iters 500 \
    --adapter-path kalkulio-adapters
### 4. Testing the Model
Once training hits 500/500 iterations, test your newly minted architectural brain using the following generation command:

### Bash
python -m mlx_lm generate \
    --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
    --adapter-path ./kalkulio-adapters \
    --max-tokens 4000 \
    --prompt "Generate a floor plan for a house with an approximate area of 120m2."
### 📁 Repository Structure
kalkulio_all.json: Master dataset (keep local, gitignored).

prepare_data.py: Script for scaling, mirroring, and token-saving (rounding/whitespace stripping).

split_data.py: Utility to split bulk JSON arrays into the raw_data folder.

kalkulio-adapters/: Local folder containing the trained LoRA weights.
