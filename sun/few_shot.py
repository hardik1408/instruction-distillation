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


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def retrieve_similar_examples(image_name, k, neighbors):
    return neighbors[str(image_name)][:k]


def process_single_image(entry, examples, model_config, inference_config, prompts_config):

    image_name = entry["image_name"]
    image_path = entry["image_path"]
    ground_truth = entry["class_name"]

    target_image_b64 = image_to_base64(image_path)

    example_images_b64 = [image_to_base64(ex["image_path"]) for ex in examples]

    content = []

    content.append({
        "type": "text",
        "text": prompts_config["few_shot_system"]
    })

    for idx, (example, img_b64) in enumerate(zip(examples, example_images_b64), 1):

        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

        content.append({
            "type": "text",
            "text": prompts_config["few_shot_example"].format(
                idx=idx,
                label=example["class_name"]
            )
        })

    content.append({
        "type": "text",
        "text": prompts_config["few_shot_target"]
    })

    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{target_image_b64}"}
    })

    messages = [{"role": "user", "content": content}]

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

        raw_response = response.choices[0].message.content.strip()

        predicted_class = raw_response.split("\n")[0].strip().lower().rstrip(".")

        return {
            "image_name": image_name,
            "predicted_class": predicted_class,
            "ground_truth": ground_truth,
            "example_images": [ex["image_name"] for ex in examples],
            "raw_response": raw_response
        }

    except Exception as e:

        print(f"\nError processing image {image_name}: {e}")

        return {
            "image_name": image_name,
            "predicted_class": "ERROR",
            "ground_truth": ground_truth,
            "error": str(e)
        }


def run_few_shot_inference(model_name, k, config):

    print("=" * 70)
    print(f"FEW-SHOT INFERENCE (K={k}) - MODEL: {model_name.upper()}")
    print("=" * 70)

    # Load test dataset (queries)
    test_dataset  = load_dataset(config)
    test_images   = test_dataset

    print(f"\nTest images  : {len(test_images)}")

    print("\nLoading precomputed neighbors...")
    neighbors = load_neighbors()

    model_config     = config["models"][model_name]
    inference_config = config["inference"]
    prompts_config   = config["prompts"]

    output_dir = os.path.join(config["output"]["image_base_dir"], model_name)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"predictions_few_shot_{k}.json")

    print(f"\nRetrieving {k} similar train examples for each test image...")

    images_with_examples = []

    for entry in tqdm(test_images):

        image_name = str(entry["image_name"])

        if image_name not in neighbors:
            print(f"  [WARN] No neighbors found for test image {image_name}, skipping.")
            continue

        examples = retrieve_similar_examples(image_name, k, neighbors)

        images_with_examples.append((entry, examples))

    results = []

    max_workers = config["experiment"]["max_workers"]

    print(f"\nProcessing images with {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = []

        for entry, examples in images_with_examples:

            future = executor.submit(
                process_single_image,
                entry,
                examples,
                model_config,
                inference_config,
                prompts_config
            )

            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), mininterval=10):

            try:

                results.append(future.result())

            except Exception as e:

                print(f"Error: {e}")

    print(f"\nSaving predictions to {output_path}")

    with open(output_path, "w") as f:

        json.dump(results, f, indent=2)

    successful = sum(1 for r in results if r["predicted_class"] != "ERROR")
    errors     = len(results) - successful

    print("\n" + "=" * 70)
    print(f"FEW-SHOT INFERENCE (K={k}) COMPLETE")
    print("=" * 70)

    print(f"Total images : {len(results)}")
    print(f"Successful   : {successful}")
    print(f"Errors       : {errors}")

    print("\nSample predictions")

    for r in results[:3]:

        print("\nImage     :", r["image_name"])
        print("Predicted :", r["predicted_class"])
        print("Ground truth:", r["ground_truth"])


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gemma", "qwen", "phi", "llama"]
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

        run_few_shot_inference(args.model, k, config)

        print("\n")


if __name__ == "__main__":
    main()