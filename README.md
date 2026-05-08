# Kalkulio AI Architect Generator

An AI-powered pipeline designed to generate valid, topologically correct JSON floor plans for the [Kalkulio AI Challenge](https://kalkulio.cz/). 

This project leverages Apple Silicon's native **MLX** framework to fine-tune an LLM (Qwen2.5-Coder) using LoRA, transforming raw conversational prompts into highly structured, geometrically strict JSON architectural representations.

## 🏗️ Architecture & Strategy

Because standard image generation models cannot output valid vector graphics or specific JSON structures with mathematically sound geometries, this project treats floor plan generation as a **Sequence-to-Sequence (Structured Output)** problem.

The pipeline consists of three main phases:
1. **Data Augmentation:** A Python script ingests 70 initial parsed JSON samples from Kalkulio and applies geometric scaling (and other augmentations) to synthetically expand the dataset.
2. **LoRA Fine-Tuning:** The augmented dataset is formatted into conversational `.jsonl` pairs and used to fine-tune `Qwen2.5-Coder-3B-Instruct` natively on an M4 Mac via `mlx_lm`.
3. **Geometric Post-Processing (Coming Soon):** A deterministic Python algorithm to "snap" AI-hallucinated coordinates to a grid, ensuring closed polygons and perfectly connected walls.

## 🚀 Quick Start (macOS / Apple Silicon)

This project requires Python 3.10+ and uses `mlx` for local training on Apple Silicon.

### 1. Setup the Environment
```bash
python3 -m venv kalkulio-env
source kalkulio-env/bin/activate
pip install mlx-lm pandas huggingface_hub
