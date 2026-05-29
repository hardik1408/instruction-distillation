import os
import json
import yaml
import base64
import argparse
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_dataset(config):
    annotations_path = config["dataset"]["pets_test_annotations_path"]
    with open(annotations_path, "r") as f:
        return json.load(f)


def load_clip_embeddings(config):
    embeddings_path = config["output"]["embeddings_path"]
    data = np.load(embeddings_path, allow_pickle=True).item()
    return data["embeddings"], data["image_names"]


def load_clip_test_embeddings(config):
    embeddings_path = config["output"]["test_embeddings_path"]
    data = np.load(embeddings_path, allow_pickle=True).item()
    return data["embeddings"], data["image_names"]


def load_train_dataset(config):
    annotations_path = config["dataset"]["pets_train_annotations_path"]
    with open(annotations_path, "r") as f:
        return json.load(f)


def load_instructions(model_name, config):
    instructions_path = os.path.join(
        config['output']['instruction_base_dir'],
        model_name,
        'per_image_instructions.json'
    )
    with open(instructions_path, 'r') as f:
        return json.load(f)


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def retrieve_hybrid_examples(query_embedding, k, k_prime, train_embeddings, train_images):
    """
    Retrieve k visual and k_prime instruction examples from the TRAIN set using CLIP embeddings.
    No self-exclusion needed since test and train are separate splits.
    """
    similarities = cosine_similarity([query_embedding], train_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    candidates = []
    for idx in sorted_indices:
        candidates.append(train_images[idx])
        if len(candidates) == k + k_prime:
            break

    image_examples = candidates[:k]
    instruction_examples = candidates[k:k + k_prime]
    return image_examples, instruction_examples


def process_hybrid_image(entry, img_exs, instr_exs, instructions_dict, model_config, inference_config, prompts_config):
    image_name = entry["image_name"]
    image_path = entry["image_path"]
    ground_truth = entry["class_name"]

    target_image_b64 = image_to_base64(image_path)
    content = []

    content.append({
        "type": "text",
        "text": prompts_config["hybrid_system"]
    })

    # Visual examples (k)
    for idx, ex in enumerate(img_exs, 1):
        content.append({
            "type": "text",
            "text": prompts_config["hybrid_image_example"].format(
                idx=idx,
                label=ex["class_name"]
            )
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(ex['image_path'])}"}
        })


    # Instruction examples (k')
    for idx, ex in enumerate(instr_exs, 1):
        # Using image_name as key to lookup instruction
        instr_data = instructions_dict.get(ex["image_name"], {})
        content.append({
            "type": "text",
            "text": prompts_config["hybrid_instruction_example"].format(
                idx=idx,
                label=ex["class_name"],
                instruction=instr_data.get("instruction", "")
            )
        })

    content.append({
        "type": "text",
        "text": prompts_config["hybrid_target"]
    })

    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{target_image_b64}"}
    })

    client = OpenAI(
        base_url=model_config["api_base"],
        api_key=model_config["api_key"]
    )

    try:
        response = client.chat.completions.create(
            model=model_config["full_name"],
            messages=[{"role": "user", "content": content}],
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
            "visual_examples": [ex["image_name"] for ex in img_exs],
            "instruction_examples": [ex["image_name"] for ex in instr_exs],
            "raw_response": raw_response
        }
    except Exception as e:
        return {
            "image_name": image_name,
            "predicted_class": "ERROR",
            "ground_truth": ground_truth,
            "error": str(e)
        }


def run_hybrid_inference(model_name, k, config):
    print("=" * 70)
    print(f"HYBRID INFERENCE (K={k}) - MODEL: {model_name.upper()}")
    print("=" * 70)

    # Load test dataset (queries)
    dataset = load_dataset(config)
    images = dataset

    # Load train dataset (example pool)
    train_dataset = load_train_dataset(config)
    train_images = train_dataset

    instructions_dict = load_instructions(model_name, config)

    # Train embeddings — search pool for retrieval
    train_embeddings, _ = load_clip_embeddings(config)

    # Test embeddings — query embeddings
    test_embeddings, test_image_ids = load_clip_test_embeddings(config)

    test_image_id_to_idx = {
        str(i): idx for idx, i in enumerate(test_image_ids)
    }

    model_config = config["models"][model_name]
    inference_config = config["inference"]
    prompts_config = config["prompts"]

    output_dir = os.path.join(config["output"]["hybrid_dir"], model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Vary k' from 1 to 3
    for k_prime in [1, 2, 3]:
        print(f"\nRunning Experiment: k={k}, k'={k_prime}")
        output_path = os.path.join(output_dir, f"predictions_hybrid_k{k}_kp{k_prime}.json")

        images_with_neighbors = []
        for entry in tqdm(images, desc="Retrieving Neighbors", mininterval=10):

            image_id = str(entry["image_name"])

            if image_id not in test_image_id_to_idx:
                print(f"  [WARN] No embedding found for test image {image_id}, skipping.")
                continue

            query_embedding = test_embeddings[test_image_id_to_idx[image_id]]

            img_exs, instr_exs = retrieve_hybrid_examples(
                query_embedding, k, k_prime, train_embeddings, train_images
            )
            images_with_neighbors.append((entry, img_exs, instr_exs))

        results = []
        max_workers = config["experiment"]["max_workers"]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_hybrid_image, entry, img_exs, instr_exs,
                    instructions_dict, model_config, inference_config, prompts_config
                )
                for entry, img_exs, instr_exs in images_with_neighbors
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Inference", mininterval=10):
                results.append(future.result())

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    k_values = [args.k] if args.k is not None else config["experiment"]["top_k_retrieval"]

    for k in k_values:
        run_hybrid_inference(args.model, k, config)


if __name__ == "__main__":
    main()