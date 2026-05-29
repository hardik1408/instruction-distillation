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
    annotations_path = config["dataset"]["dtd_train_annotations_path"]

    with open(annotations_path, "r") as f:
        return json.load(f)


def image_to_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_instruction_prompt(label, config):
    """Create instruction generation prompt."""
    prompt_template = config["prompts"]["instruction_generation"]

    return prompt_template.format(label=label)


def generate_instruction_for_image(entry, model_config, inference_config, prompts_config):
    """
    Generate reasoning instruction for a single image.
    """

    image_name = entry["image_name"]
    image_path = entry["image_path"]
    label = entry["class_name"]

    image_b64 = image_to_base64(image_path)

    prompt = create_instruction_prompt(label, {"prompts": prompts_config})

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

    try:

        response = client.chat.completions.create(
            model=model_config["full_name"],
            messages=messages,
            max_tokens=inference_config["max_tokens_instruction"],
            temperature=inference_config["temperature"]
        )

        instruction = response.choices[0].message.content.strip()
        # print(instruction)
        return {
            "image_name": image_name,
            "class_name": label,
            "instruction": instruction,
            "status": "success"
        }

    except Exception as e:

        print(f"\nError generating instruction for image {image_name}: {e}")

        return {
            "image_name": image_name,
            "class_name": label,
            "instruction": None,
            "status": "error",
            "error": str(e)
        }


def generate_per_image_instructions(model_name, config):

    print("=" * 70)
    print(f"GENERATING IMAGE INSTRUCTIONS - MODEL: {model_name.upper()}")
    print("=" * 70)

    dataset = load_dataset(config)
    images = dataset

    print(f"\nTotal images: {len(images)}")

    model_config = config["models"][model_name]
    inference_config = config["inference"]
    prompts_config = config["prompts"]

    print(f"\nModel: {model_config['full_name']}")
    print(f"API Base: {model_config['api_base']}")
    print(f"Temperature: {inference_config['temperature']}")
    print(f"Max tokens: {inference_config['max_tokens_instruction']}")

    output_dir = os.path.join(config["output"]["instruction_base_dir"], model_name)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "per_image_instructions.json")

    if os.path.exists(output_path):

        print(f"\nWarning: Output file exists: {output_path}")
        response = input("Overwrite? (yes/no): ")

        if response.lower() not in ["yes", "y"]:
            print("Aborting.")
            return

    results = []
    instructions_dict = {}

    max_workers = config["experiment"]["max_workers"]

    print(f"\nGenerating instructions with {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = []

        for entry in images:

            future = executor.submit(
                generate_instruction_for_image,
                entry,
                model_config,
                inference_config,
                prompts_config
            )

            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):

            try:

                result = future.result()

                results.append(result)

                if result["status"] == "success":

                    instructions_dict[result["image_name"]] = {
                        "class_name": result["class_name"],
                        "instruction": result["instruction"]
                    }

            except Exception as e:

                print("Error:", e)

    print(f"\nSaving instructions to {output_path}")

    with open(output_path, "w") as f:

        json.dump(instructions_dict, f, indent=2)

    successful = sum(1 for r in results if r["status"] == "success")

    errors = len(results) - successful

    print("\n" + "=" * 70)
    print("INSTRUCTION GENERATION COMPLETE")
    print("=" * 70)

    print(f"Total images: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")

    print("\nSample instructions")
    print("-" * 70)

    count = 0

    for r in results:

        if r["status"] == "success" and count < 3:

            print("\nImage:", r["image_name"])
            print("Class:", r["class_name"])
            print("Instruction:", r["instruction"][:200], "...")

            count += 1

    log_path = os.path.join(output_dir, "instruction_generation_log.json")

    with open(log_path, "w") as f:

        json.dump(results, f, indent=2)

    print(f"\nDetailed log saved to {log_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Generate reasoning instructions for each image"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gemma", "qwen", "phi", "llama"]
    )

    args = parser.parse_args()

    config = load_config()

    generate_per_image_instructions(args.model, config)


if __name__ == "__main__":
    main()