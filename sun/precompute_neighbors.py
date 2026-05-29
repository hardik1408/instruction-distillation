#!/usr/bin/env python3
"""
Precompute the 50 nearest train neighbors for each test image using CLIP embeddings.
Saves results to data/nearest_neighbors.json.

Run once before approachone.py, few_shot.py, hybrid.py, hybrid_new.py.
"""

import json
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_clip_embeddings(config):
    data = np.load(config["output"]["embeddings_path"], allow_pickle=True).item()
    return data["embeddings"], data["image_names"]


def load_clip_test_embeddings(config):
    data = np.load(config["output"]["test_embeddings_path"], allow_pickle=True).item()
    return data["embeddings"], data["image_names"]


def load_train_dataset(config):
    with open(config["dataset"]["sun_train_annotations_path"], "r") as f:
        return json.load(f)


def load_test_dataset(config):
    with open(config["dataset"]["sun_test_annotations_path"], "r") as f:
        return json.load(f)


def main():
    config = load_config()
    n_neighbors = 50

    print("Loading CLIP embeddings...")
    train_embeddings, train_image_names = load_clip_embeddings(config)
    test_embeddings, test_image_names = load_clip_test_embeddings(config)

    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Test  embeddings: {test_embeddings.shape}")

    print("Loading datasets...")
    train_dataset = load_train_dataset(config)
    test_dataset = load_test_dataset(config)

    # Build lookup: image_name -> train entry
    train_by_name = {str(entry["image_name"]): entry for entry in train_dataset}

    # Map test image_name -> index in test_embeddings
    test_id_to_idx = {str(name): idx for idx, name in enumerate(test_image_names)}

    output = {}

    print(f"Computing {n_neighbors} nearest neighbors for {len(test_dataset)} test images...")

    for entry in tqdm(test_dataset):
        image_name = str(entry["image_name"])

        if image_name not in test_id_to_idx:
            print(f"  [WARN] No embedding for test image {image_name}, skipping.")
            continue

        query_embedding = test_embeddings[test_id_to_idx[image_name]]
        similarities = cosine_similarity([query_embedding], train_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        neighbors = []
        for idx in sorted_indices:
            neighbor_name = str(train_image_names[idx])
            if neighbor_name in train_by_name:
                train_entry = train_by_name[neighbor_name]
                neighbors.append({
                    "image_name": train_entry["image_name"],
                    "image_path": train_entry["image_path"],
                    "class_name": train_entry["class_name"],
                    "similarity": float(similarities[idx])
                })
            if len(neighbors) == n_neighbors:
                break

        output[image_name] = neighbors

    output_path = "data/nearest_neighbors.json"
    print(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"Done. Saved {len(output)} entries.")


if __name__ == "__main__":
    main()
