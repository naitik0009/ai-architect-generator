Kalkulio AI Architect Generator
An AI-powered pipeline designed to generate valid, topologically correct JSON floor plans for [the Kalkulio AI Challenge](https://ai.kalkulio.cz/challenge).

An AI system that generates valid, geometrically-correct JSON floor plans for
single-family houses from a simple text prompt (target area in m²), built for
the **Kalkulio AI Challenge**.

The system pairs a **fine-tuned LLM** (Qwen2.5-Coder + LoRA) with a
**deterministic geometric post-processor** that guarantees every output is
watertight, correctly scaled, and architecturally sensible — then serves it
through an interactive **Gradio UI** with floor-plan visualization.

## Example outputs

Generated live through the Gradio UI and cleaned by the post-processor —
watertight, exactly area-matched, orthogonal walls, Czech room labels.

| 60 m² | 120 m² | 180 m² |
|:---:|:---:|:---:|
| ![60 m² floor plan](docs/images/plan_60m2.png) | ![120 m² floor plan](docs/images/plan_120m2.png) | ![180 m² floor plan](docs/images/plan_180m2.png) |

---

## Why this architecture

A language model alone will produce *plausible-looking* floor plans, but with
geometric flaws: floating walls, non-orthogonal lines, overlapping rooms,
mislabeled spaces, and gaps that break watertightness. Rather than hoping a
bigger model fixes this, we **split responsibilities**:

| Layer | Responsibility |
|-------|----------------|
| **LLM** | Propose rooms — sensible layout, types, and approximate sizes |
| **Post-processor** | Enforce hard geometric guarantees deterministically |
| **Best-of-N sampler** | Generate several candidates, keep the best |
| **Procedural fallback** | Guarantee the UI always returns a valid plan |

This makes the output **robust by construction** rather than probabilistically.

---

## Pipeline

```
prompt ("90 m² house")
      │
      ▼
┌─────────────────┐   raw JSON    ┌──────────────────────┐  clean JSON  ┌──────────────┐
│ Qwen2.5-Coder   │ ────────────▶ │  post_process.py      │ ───────────▶ │  Gradio UI   │
│ 14B + LoRA      │  (best-of-N)  │  (geometric cleanup)  │              │  + renderer  │
└─────────────────┘               └──────────────────────┘              └──────────────┘
```

### 1. Data preparation (`prepare_data.py`)
- Ingests **60 real Kalkulio houses** + **synthetic plans** (see Data Generation)
- Geometric augmentation: 4 rotations × 3 mirrors × 3 scales = **36 variants/house**
- 8 prompt variants (English + Czech, formal + casual)
- Area-balanced oversampling (large houses weighted up to fight regression-to-mean)
- Holds out whole houses for validation (no augmented-twin leakage)

### 2. Fine-tuning (`train.py`)
- Base model: **Qwen2.5-Coder-14B-Instruct**
- **LoRA** (r=128, α=256) on all attention + MLP projections
- bf16, gradient checkpointing, TF32, fused AdamW
- **Multi-GPU data-parallel (DDP)** via `torchrun` — auto-detects `WORLD_SIZE`
  and auto-scales gradient accumulation to keep a stable effective batch size
- **Auto-resume** from the last checkpoint after any interruption

### 3. Geometric post-processing (`post_process.py`)
The core differentiator. A deterministic pipeline:

1. **Clean rooms** — drop outdoor/invalid types, tiny rooms, overlaps
2. **Snap rooms together** — pull near-touching rooms to shared vertices
3. **Drop disconnected islands** — keep only the connected house cluster
4. **Rescale** to the exact target area
5. **Rebuild walls from room polygons** — walls are *derived* from room edges,
   which **guarantees watertight geometry** and discards the model's messy walls
6. **Fix labels** — enforce realistic room types (one LivingRoom/Kitchen/etc.),
   guarantee every room has a valid type + Czech name

### 4. Serving (`app.py`)
- **Gradio UI** with a generation tab (area slider → floor plan)
- **Best-of-N sampling** (staggered temperatures) scored on watertightness,
  room count, and area accuracy
- **Procedural fallback** so the UI never shows a broken result
- Matplotlib renderer: walls over rooms, door arcs, windows, Czech labels

---

## Data generation

To enrich the original 60-house dataset, two complementary generators were built:

### `generate_synthetic_data.py` — LLM-generated plans
- Calls **Gemini** (3.5-flash → 3.1-pro → … fallback chain across models so a
  single model's daily quota doesn't block the run)
- Asks only for **rooms**; walls are derived by `post_process`
- Validates every plan (watertight + area match) before saving

### `generate_handcrafted_data.py` — template-based plans
- **70 hand-designed plans** across 5 size buckets (60–200 m²) and many styles
  (narrow, L-shape, U-shape, ranch, open-plan, etc.)
- Mathematically perfect tiling — watertight by construction
- Zero API dependency, fully deterministic

---

## Quick start

Requires Python 3.10+ and a CUDA-capable NVIDIA GPU.

```bash
git clone https://github.com/naitik0009/ai-architect-generator.git
cd ai-architect-generator
python3 -m venv kalkulio-env
source kalkulio-env/bin/activate
pip install -r requirements.txt
```

### Prepare data
```bash
python prepare_data.py          # builds data/train.jsonl + data/valid.jsonl
```

### Train

Single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

Multi-GPU (data-parallel, e.g. 3 GPUs):
```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 train.py
```

The adapter is saved to `./qwen-kalkulio-lora-14b-v4/final`. Training
checkpoints every 100 steps and auto-resumes if restarted.

### Generate / clean a single plan
```bash
python quick_test.py                                  # generate sample plans
python post_process.py raw.json -o clean.json --target-area 90 --pretty
python evaluate.py                                    # before/after quality metrics
```

### Run the app
```bash
python app.py        # launches the Gradio UI
```

---

## Repository layout

| File | Purpose |
|------|---------|
| `prepare_data.py` | Dataset augmentation + conversational formatting |
| `train.py` | LoRA fine-tuning (single- or multi-GPU, auto-resume) |
| `post_process.py` | Deterministic geometric cleanup + CLI |
| `app.py` | Gradio UI: generation, best-of-N, rendering |
| `quick_test.py` | Inference sanity check (raw + cleaned output) |
| `evaluate.py` | Quantitative quality metrics |
| `generate_synthetic_data.py` | Gemini-based synthetic plan generator |
| `generate_handcrafted_data.py` | Template-based plan generator |
| `preview_synthetic.py` | Quick visual preview of a plan |
| `kalkulio_all.json` | 60 source Kalkulio houses |
| `raw_data/` | Source + synthetic training plans |

---

## Model & results

- **Base:** Qwen2.5-Coder-14B-Instruct
- **Method:** LoRA (r=128, α=256), 2 epochs
- **Hardware:** 3× NVIDIA RTX PRO 6000 Blackwell (data-parallel DDP, ~1.5 h)
- **Training set:** 60 real Kalkulio houses + Gemini-synthetic + 70 handcrafted
  plans, ×8 geometric augmentation (rotation × mirror)

Measured on **12 generated plans spanning 70–190 m²** (`evaluate.py`):

| Metric | Raw model | After post-process |
|--------|-----------|--------------------|
| Valid JSON | 100% | 100% |
| **Watertight rate** | 83% | **100%** |
| **Mean area error** | 27.7% | **0.0%** |
| Orphan rooms | 1.2% | **0%** |
| Polygon closure | 91% | **100%** |
| Wall connectivity | 99% | **100%** |
| Avg rooms / plan | 10.6 | 9.9 |

The fine-tuned model already produces valid, mostly-watertight JSON; the
deterministic post-processor closes the remaining gaps — **every** output ends
up watertight, exactly area-matched, and free of orphan rooms. This split
(LLM for layout, deterministic geometry for guarantees) is the core of the
design.

---

## Running & deployment

The model is Qwen2.5-Coder-14B (~28 GB in bf16), so the recommended way to run
the project is **locally on a CUDA GPU**:

```bash
python app.py        # launches the Gradio UI on http://localhost:7860
```

A short **demo video** shows generation + visualization without requiring judges
to provision a 14B-capable GPU.

### Hosting options
| Option | Notes |
|--------|-------|
| Local GPU (recommended) | Run `app.py` directly; needs ~30 GB VRAM |
| HF Spaces (paid GPU, A10G/A100) | Push the adapter to the Hub; `app.py` is the Space entrypoint |
| 4-bit quantized | Load the base model with `bitsandbytes` 4-bit to fit a ~16 GB GPU |

Back up / share the trained adapter:
```bash
huggingface-cli upload <user>/qwen-kalkulio-lora-14b-v4 ./qwen-kalkulio-lora-14b-v4/final .
```

---

## License

Built for the Kalkulio AI Challenge.
