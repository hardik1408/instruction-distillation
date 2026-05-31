# Instruction Distillation: Text Instructions as Visual Examples

## Abstract

Visual in-context learning (ICL) with multimodal large language models (MLLMs) is effective for fine-grained visual classification, but each retrieved image example consumes several hundred context tokens, making large-K settings prohibitively expensive at inference scale. We propose instruction distillation: an offline procedure in which the MLLM itself
generates, for each individual training image, a structured identification instruction encoding general appearance cues, features that differen-
tiate the class from visually similar ones, and a common confusion point. Unlike prior work that produces a single description per class, our instructions are generated per training image, preserving the intra-class visual diversity that per-class descriptions collapse. At inference time, we study five configurations sharing a single CLIP retrieval index: zero-shot, image ICL, instruction-only ICL, and two hybrid vari-
ants in which retrieved neighbors are split between images and instructions. Across seven fine-grained benchmarks and two MLLM backbones, instruction based pipelines matches, or
exceeds image ICL at K=1 and reduces perquery tokens by 2.9× at K=5. Hybrid configurations further show that visual and textual ICL
signals are complementary, images gives visual
patterns to learn and see, while instructions
give explicit rule and logic. When both of these
are provided, the quality of context improves,
which is noticeable in the performance.

## Approaches

| Script | Method | In-context examples |
|--------|--------|---------------------|
| `zero_shot.py` | Zero-shot(P0) | None |
| `few_shot.py` | Visual few-shot(P1) | K example images |
| `instr_dist.py` | Instruction distillation(P2) | K text instructions |
| `hybrid.py` | Hybrid (images-first)(P3) | K images + K' instructions |
| `hybrid_new.py` | Hybrid (instructions-first)(P4) | K' instructions + K images |

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

## Requirements

```
openai
numpy
scikit-learn
open_clip_torch (or clip)
pyyaml
tqdm
```

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

## Evaluation

Each dataset directory contains an `eval.py` script that scores a predictions JSON file produced by any of the inference scripts.

```bash
python eval.py --p <path/to/predictions.json> --o <path/to/output_results.json>
```

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| **Top-1 Accuracy** | Fraction of test images where the top prediction exactly matches the ground-truth label. |
| **Mean Per-Class Accuracy** | Macro-average of per-class accuracies — the primary reported metric. Treats every class equally regardless of how many test images it has, making it robust to class imbalance. |

Label comparison is case-insensitive and normalizes punctuation (underscores, hyphens, and curly apostrophes are collapsed to spaces) so that `american_bulldog`, `American Bulldog`, and `american bulldog` all match.

