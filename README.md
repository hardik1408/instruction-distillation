# Instruction Distillation: Text Instructions as Visual Examples

A research codebase exploring whether **text-based reasoning instructions** can replace **visual examples** in few-shot image classification with Vision-Language Models (VLMs).

## Overview

Standard few-shot classification provides a VLM with example images at inference time. This project investigates an alternative: use a VLM to generate natural-language reasoning rules (instructions) for training images, then retrieve and pass those text instructions to classify unseen test images — without sending any example images.

**Pipeline:**

1. **Instruction Generation** (`instruction_manual.py`) — For each training image, prompt a VLM to produce 3 actionable classification rules grounded in visible perceptual properties.
2. **CLIP Embedding** (`clip.py`) — Encode all train and test images with CLIP to enable retrieval.
3. **Instruction Distillation Inference** (`instr_dist.py`) — At test time, retrieve the K most similar training images via CLIP cosine similarity, fetch their stored text instructions, and pass those instructions (no images) as in-context examples to classify the test image.

## Approaches

| Script | Method | In-context examples |
|--------|--------|---------------------|
| `zero_shot.py` | Zero-shot | None |
| `few_shot.py` | Visual few-shot | K example images |
| `instr_dist.py` | Instruction distillation | K text instructions |
| `hybrid.py` | Hybrid (images-first) | K images + K' instructions |
| `hybrid_new.py` | Hybrid (instructions-first) | K' instructions + K images |

## Datasets

Each dataset lives in its own directory with identical scripts and a `config.yaml`:

| Directory | Dataset |
|-----------|---------|
| `dtd/` | Describable Textures Dataset (DTD) |
| `euro/` | EuroSAT |
| `fgvc/` | FGVC Aircraft |
| `flower/` | Oxford 102 Flowers |
| `food/` | Food-101 |
| `pets/` | Oxford-IIIT Pets |
| `sun/` | SUN397 |

## Models

Models are served locally via an OpenAI-compatible API (e.g., vLLM). Supported models are configured in each `config.yaml`:

| Key | Model |
|-----|-------|
| `qwen` | Qwen/Qwen2.5-VL-7B-Instruct |
| `gemma` | google/gemma-3-4b-it |

## Usage

All scripts are run from within the dataset directory (e.g., `cd dtd`).

**Step 1 — Generate CLIP embeddings:**
```bash
python clip.py
```

**Step 2 — Generate text instructions for training images:**
```bash
python instruction_manual.py --model qwen
```

**Step 3 — Run inference:**
```bash
# Zero-shot
python zero_shot.py --model qwen

# Visual few-shot
python few_shot.py --model qwen --k 3

# Instruction distillation (text-only in-context examples)
python instr_dist.py --model qwen --k 3

# Hybrid
python hybrid.py --model qwen --k 3
python hybrid_new.py --model qwen --k 3
```

**Run all experiments for a dataset:**
```bash
bash run.sh
```

The `--k` argument controls how many retrieved examples are used as in-context examples.

## Configuration

Each dataset has a `config.yaml` with paths, model endpoints, inference hyperparameters, and prompt templates. Update `api_base` to point to your locally running model server.

```yaml
models:
  qwen:
    full_name: "Qwen/Qwen2.5-VL-7B-Instruct"
    api_base: "http://localhost:8003/v1"
    api_key: "skip"

inference:
  temperature: 0.2
  max_tokens_inference: 150

experiment:
  top_k_retrieval: [1, 2, 3, 5]
  max_workers: 20
  clip_model: "ViT-B-32"
```

## Requirements

```
openai
numpy
scikit-learn
open_clip_torch (or clip)
pyyaml
tqdm
```

## Output Structure

```
<dataset>/
  data/
    clip_embeddings_train.npy
    clip_embeddings_test.npy
  instruction/<model>/
    per_image_instructions.json      # generated training instructions
    predictions_few_shot_<k>.json    # inference results
  image/<model>/
    predictions_zero_shot.json
  results/
```
