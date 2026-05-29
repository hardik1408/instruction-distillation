#!/usr/bin/env python3

import os
import json
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_dataset(config):
    """Load Stanford Cars annotation JSON."""
    annotations_path = config["dataset"]["fgvc_test_annotations_path"]

    print(f"Loading dataset from {annotations_path}...")

    with open(annotations_path, "r") as f:
        return json.load(f)


def precompute_clip_embeddings(config):
    """
    Precompute CLIP embeddings for all images in the Stanford Cars dataset.
    """

    print("=" * 70)
    print("PRECOMPUTING CLIP EMBEDDINGS")
    print("=" * 70)

    dataset = load_dataset(config)
    images = dataset

    print(f"\nTotal images found: {len(images)}")

    # Configuration
    clip_model_name = config["experiment"]["clip_model"]
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading CLIP model: {clip_model_name}")
    print(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained="openai"
    )

    model = model.to(device)
    model.eval()

    image_paths = [img["image_path"] for img in images]
    image_names = [img["image_name"] for img in images]

    print(f"\nComputing embeddings in batches of {batch_size}...")

    all_embeddings = []
    valid_image_names = []
    failed_images = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):

            batch_paths = image_paths[i:i + batch_size]
            batch_names = image_names[i:i + batch_size]

            batch_images = []
            valid_names = []

            for img_path, img_name in zip(batch_paths, batch_names):

                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(preprocess(img))
                    valid_names.append(img_name)

                except Exception as e:
                    print(f"\nError processing image {img_name} ({img_path}): {e}")
                    failed_images.append(img_name)

            if not batch_images:
                continue

            batch_images = torch.stack(batch_images).to(device)

            embeddings = model.encode_image(batch_images)

            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())
            valid_image_names.extend(valid_names)

    all_embeddings = np.vstack(all_embeddings)

    print(f"\nSuccessfully computed embeddings for {len(valid_image_names)} images")

    if failed_images:
        print(f"Failed to process {len(failed_images)} images")

    print(f"Embeddings shape: {all_embeddings.shape}")

    output_path = config["output"]["test_embeddings_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nSaving embeddings to {output_path}...")

    np.save(
        output_path,
        {
            "embeddings": all_embeddings,
            "image_names": valid_image_names,
            "clip_model": clip_model_name,
            "embedding_dim": all_embeddings.shape[1],
        },
    )

    print("\n" + "=" * 70)
    print("CLIP EMBEDDINGS PRECOMPUTATION COMPLETE")
    print("=" * 70)

    print(f"Output file: {output_path}")
    print(f"Number of embeddings: {len(valid_image_names)}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

    print("\nVerifying saved embeddings...")

    loaded_data = np.load(output_path, allow_pickle=True).item()

    print(f"✓ Successfully loaded {loaded_data['embeddings'].shape[0]} embeddings")
    print(f"✓ Embedding dimension: {loaded_data['embedding_dim']}")
    print(f"✓ CLIP model used: {loaded_data['clip_model']}")

    return all_embeddings, valid_image_names


def main():
    config = load_config()
    precompute_clip_embeddings(config)


if __name__ == "__main__":
    main()