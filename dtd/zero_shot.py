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
    """Load Stanford Cars dataset."""
    annotations_path = config["dataset"]["dtd_test_annotations_path"]

    with open(annotations_path, "r") as f:
        return json.load(f)


def image_to_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_zero_shot_prompt(config):
    """Create zero-shot prompt from template."""
    prompt_template = config["prompts"]["zero_shot"]

    # classes = ", ".join(label_list)

    prompt = prompt_template + "Answer:"
    return prompt


def process_single_image(entry, model_config, inference_config, prompts_config):
    """
    Process a single image with zero-shot classification.
    """

    image_name = entry["image_name"]
    image_path = entry["image_path"]
    ground_truth = entry["class_name"]
    # print(image_path)
    image_b64 = image_to_base64(image_path)
    # print(image_b64)
    prompt = create_zero_shot_prompt({"prompts": prompts_config})
    # print(prompt)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    client = OpenAI(
        base_url=model_config["api_base"],
        api_key=model_config["api_key"]
    )
    # print(client)
    try:
        response = client.chat.completions.create(
            model=model_config["full_name"],
            messages=messages,
            max_tokens=inference_config["max_tokens_inference"],
            temperature=inference_config["temperature"]
        )

        raw_response = response.choices[0].message.content.strip()

        predicted_class = raw_response.split("\n")[0].strip().lower().rstrip(".")
        print(predicted_class)
        return {
            "image_name": image_name,
            "predicted_class": predicted_class,
            "ground_truth": ground_truth,
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


def run_zero_shot_inference(model_name, config):
    """
    Run zero-shot classification on Stanford Cars dataset.
    """

    print("=" * 70)
    print(f"ZERO-SHOT INFERENCE - MODEL: {model_name.upper()}")
    print("=" * 70)

    dataset = load_dataset(config)

    images = dataset["images"]
    # print(images)
    # label_list = dataset["classes"]

    print(f"\nTotal images: {len(images)}")
    # print(f"Total classes: {len(label_list)}")

    model_config = config["models"][model_name]
    inference_config = config["inference"]
    prompts_config = config["prompts"]

    print(f"\nModel: {model_config['full_name']}")
    print(f"API Base: {model_config['api_base']}")
    print(f"Temperature: {inference_config['temperature']}")
    print(f"Max tokens: {inference_config['max_tokens_inference']}")

    output_dir = os.path.join(config["output"]["image_base_dir"], model_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "predictions_zero_shot.json")

    results = []
    max_workers = config["experiment"]["max_workers"]

    print(f"\nProcessing images with {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = []

        for entry in images:
            future = executor.submit(
                process_single_image,
                entry,
                model_config,
                inference_config,
                prompts_config
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), mininterval=10):

            try:
                result = future.result()
                results.append(result)

            except Exception as e:
                print(f"Error: {e}")

    print(f"\nSaving predictions to {output_path}...")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    successful = sum(1 for r in results if r["predicted_class"] != "ERROR")
    errors = len(results) - successful

    print("\n" + "=" * 70)
    print("ZERO-SHOT INFERENCE COMPLETE")
    print("=" * 70)

    print(f"Total images: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")

    print("\nSample predictions:")
    print("-" * 70)

    for i, result in enumerate(results[:5]):

        print(f"\nImage {i+1}: {result['image_name']}")
        print(f"Predicted: {result['predicted_class']}")
        print(f"Ground truth: {result['ground_truth']}")

    print(images[0])

def main():
    parser = argparse.ArgumentParser(description="Zero-shot image classification")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gemma", "qwen", "phi", "llama"],
        help="Model to use for inference"
    )

    args = parser.parse_args()

    config = load_config()

    run_zero_shot_inference(args.model, config)


if __name__ == "__main__":
    main()