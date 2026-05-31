#!/usr/bin/env python3

import os
import json
import yaml
import base64
import argparse
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_dataset(config):
    annotations_path = config["dataset"]["sun_test_annotations_path"]
    with open(annotations_path, "r") as f:
        return json.load(f)


def load_neighbors():
    with open("data/nearest_neighbors.json", "r") as f:
        return json.load(f)


def load_instructions(model_name, config):

    instructions_path = os.path.join(
        config["output"]["instruction_base_dir"],
        model_name,
        "per_image_instructions.json"
    )

    with open(instructions_path, "r") as f:
        return json.load(f)


def image_to_base64(path):

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def retrieve_similar_images(image_name, k, neighbors, instructions_dict):
    """
    Retrieve K most similar train images that have instructions,
    using precomputed nearest neighbors.
    """
    retrieved = []

    for neighbor in neighbors[str(image_name)]:
        neighbor_id = str(neighbor["image_name"])

        if neighbor_id in instructions_dict:
            instr = instructions_dict[neighbor_id]
            retrieved.append({
                "image_id": neighbor_id,
                "label": instr["class_name"],
                "instruction": instr["instruction"],
                "similarity": neighbor["similarity"]
            })

        if len(retrieved) == k:
            break

    return retrieved


def create_prompt(retrieved_instructions, prompts_config):

    prompt_parts = [prompts_config["approach1_system"]]

    for idx, instr in enumerate(retrieved_instructions, 1):

        block = f"""
Reasoning Strategy {idx}

Flower Name: {instr['label']}

{instr['instruction']}
"""
        prompt_parts.append(block)

    prompt_parts.append(prompts_config["approach1_target"])

    return "\n\n".join(prompt_parts)


def process_single_image(
    entry,
    retrieved,
    model_config,
    inference_config,
    prompts_config
):

    image_id = str(entry["image_name"])
    image_path = entry["image_path"]
    label = entry["class_name"]

    image_b64 = image_to_base64(image_path)

    prompt = create_prompt(retrieved, prompts_config)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        }
    ]

    client = OpenAI(
        base_url=model_config["api_base"],
        api_key=model_config["api_key"]
    )

    try:

        response = client.chat.completions.create(
            model=model_config["full_name"],
            messages=messages,
            max_tokens=inference_config["max_tokens_inference"],
            temperature=inference_config["temperature"]
        )

        raw = response.choices[0].message.content.strip()

        prediction = raw.lower().strip().rstrip(".")
        print(prediction)

        return {
            "image_id": image_id,
            "image_path": image_path,
            "ground_truth": label,
            "predicted_label": prediction,
            "retrieved_image_ids": [r["image_id"] for r in retrieved],
            "raw_response": raw
        }

    except Exception as e:

        print(f"Error processing image {image_id}: {e}")

        return {
            "image_id": image_id,
            "ground_truth": label,
            "predicted_label": "ERROR",
            "error": str(e)
        }


def run_inference(model_name, k, config):

    print("=" * 70)
    print(f"INSTRUCTION INFERENCE (K={k}) - MODEL: {model_name}")
    print("=" * 70)

    test_dataset = load_dataset(config)
    test_images  = test_dataset

    print(f"Test images : {len(test_images)}")

    print("Loading precomputed neighbors...")
    neighbors = load_neighbors()

    print("Loading instructions...")
    instructions_dict = load_instructions(model_name, config)

    model_config     = config["models"][model_name]
    inference_config = config["inference"]
    prompts_config   = config["prompts"]

    output_dir = os.path.join(
        config["output"]["instruction_base_dir"],
        model_name
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"predictions_few_shot_{k}.json"
    )

    tasks = []

    print(f"Retrieving top {k} similar train images for each test image...")

    for entry in tqdm(test_images):

        image_id = str(entry["image_name"])

        if image_id not in neighbors:
            print(f"  [WARN] No neighbors found for test image {image_id}, skipping.")
            continue

        retrieved = retrieve_similar_images(
            image_id,
            k,
            neighbors,
            instructions_dict
        )

        tasks.append((entry, retrieved))

    results = []

    max_workers = config["experiment"]["max_workers"]

    print(f"Running inference with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(
                process_single_image,
                entry,
                retrieved,
                model_config,
                inference_config,
                prompts_config
            )
            for entry, retrieved in tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), mininterval=10):
            results.append(future.result())

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved predictions:", output_path)

    correct = sum(
        1 for r in results
        if r["predicted_label"] == r["ground_truth"].lower()
    )

    accuracy = correct / len(results)

    print("Top-1 Accuracy:", round(accuracy * 100, 2), "%")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True
    )

    parser.add_argument(
        "--k",
        type=int,
        default=None
    )

    args = parser.parse_args()

    config = load_config()

    if args.k is not None:
        k_values = [args.k]
    else:
        k_values = config["experiment"]["top_k_retrieval"]

    for k in k_values:
        run_inference(args.model, k, config)


if __name__ == "__main__":
    main()
