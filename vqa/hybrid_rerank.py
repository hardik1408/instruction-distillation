#!/usr/bin/env python3

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
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_subset(config):
    """Load the VQA subset data."""
    subset_path = config['output']['subset_path']
    with open(subset_path, 'r') as f:
        return json.load(f)

def load_question_embeddings(config):
    """Load precomputed question embeddings."""
    embeddings_path = config['output']['question_embeddings_path']
    data = np.load(embeddings_path, allow_pickle=True).item()
    return data['embeddings'], data['question_ids'], data['questions']

def load_clip_embeddings(config):
    """Load precomputed CLIP image embeddings."""
    embeddings_path = config['output']['embeddings_path']
    data = np.load(embeddings_path, allow_pickle=True).item()
    image_id_to_embedding = {
        str(iid): emb
        for iid, emb in zip(data['image_ids'], data['embeddings'])
    }
    return image_id_to_embedding

def load_instructions(model_name, config):
    """Load pre-generated instructions."""
    instructions_path = os.path.join(
        config['output']['instruction_base_dir'],
        model_name,
        'per_image_instructions.json'
    )
    with open(instructions_path, 'r') as f:
        return json.load(f)

def image_to_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def retrieve_hybrid_examples(query_question_id, query_image_id, query_q_embedding,
                              k, all_q_embeddings, all_question_ids,
                              clip_embeddings, instructions_dict, subset_data_dict):
    """
    Retrieve k image examples and 5k instruction examples using the two-stage approach.

    A single pool of k_total = k + 5k = 6k candidates is built:
      1. Fetch 3 * k_total candidates by question text similarity.
         Candidates must have an instruction and must not share the query image.
      2. Re-rank those candidates by CLIP image similarity.
      3. Top k  → image examples  (shown with their image + Q&A).
      4. Next 5k → instruction examples (shown as text-only instruction + Q&A).

    Returns (image_examples, instruction_examples).
    """
    k_instr = k
    k_total = k + k_instr
    k_prime = 2*k_total

    # Stage 1: question text similarity — collect up to k_prime candidates
    similarities = cosine_similarity([query_q_embedding], all_q_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    candidates = []
    for idx in sorted_indices:
        neighbor_question_id = all_question_ids[idx]

        if neighbor_question_id == query_question_id:
            continue

        neighbor_entry = subset_data_dict.get(neighbor_question_id)
        if neighbor_entry and str(neighbor_entry['image_id']) == query_image_id:
            continue

        if neighbor_question_id not in instructions_dict:
            continue

        instr_data = instructions_dict[neighbor_question_id]
        neighbor_image_id = subset_data_dict.get(neighbor_question_id, {}).get('image_id')
        candidates.append({
            'question_id': neighbor_question_id,
            'image_id': str(neighbor_image_id) if neighbor_image_id is not None else None,
            'image_path': neighbor_entry['image_path'],
            'question': instr_data['question'],
            'answer': instr_data['answer'],
            'instruction': instr_data['instruction'],
        })

        if len(candidates) == k_prime:
            break

    # Stage 2: re-rank by CLIP image similarity
    query_clip = clip_embeddings.get(str(query_image_id))

    if query_clip is not None and len(candidates) > 0:
        candidate_clips = np.array([
            clip_embeddings.get(c['image_id'], np.zeros_like(query_clip))
            if c['image_id'] is not None else np.zeros_like(query_clip)
            for c in candidates
        ])
        image_sims = cosine_similarity([query_clip], candidate_clips)[0]
        reranked_indices = np.argsort(image_sims)[::-1]
        candidates = [candidates[i] for i in reranked_indices]

    pool = candidates[:k_total]
    image_examples = pool[:k]
    instruction_examples = pool[k:k + k_instr]

    return image_examples, instruction_examples

def process_single_question(entry, image_examples, instruction_examples,
                             model_config, inference_config, prompts_config):
    """
    Build and send a hybrid prompt: K image examples + 5K instruction examples.
    """
    question_id = entry['question_id']
    question    = entry['question']
    image_path  = entry['image_path']
    ground_truth = entry['answer']

    target_image_b64 = image_to_base64(image_path)

    content = []

    # System message
    content.append({"type": "text", "text": prompts_config['hybrid_system']})

    # K visual examples: image then Q&A text
    for idx, ex in enumerate(image_examples, 1):
        ex_img_b64 = image_to_base64(ex['image_path'])
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{ex_img_b64}"}
        })
        content.append({
            "type": "text",
            "text": prompts_config['hybrid_image_example'].format(
                idx=idx,
                question=ex['question'],
                answer=ex['answer']
            )
        })

    # 5K instruction examples: pure text
    for idx, instr in enumerate(instruction_examples, 1):
        content.append({
            "type": "text",
            "text": prompts_config['hybrid_instruction_example'].format(
                idx=idx,
                question=instr['question'],
                answer=instr['answer'],
                instruction=instr['instruction']
            )
        })

    # Target: question text, then target image
    content.append({
        "type": "text",
        "text": prompts_config['hybrid_target'].format(question=question)
    })
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{target_image_b64}"}
    })

    messages = [{"role": "user", "content": content}]

    client = OpenAI(
        base_url=model_config['api_base'],
        api_key=model_config['api_key']
    )

    try:
        response = client.chat.completions.create(
            model=model_config['full_name'],
            messages=messages,
            max_tokens=inference_config['max_tokens_inference'],
            temperature=inference_config['temperature']
        )

        raw_answer = response.choices[0].message.content.strip()

        if "Answer:" in raw_answer:
            predicted_answer = raw_answer.split("Answer:")[-1].strip()
        else:
            predicted_answer = raw_answer

        predicted_answer = predicted_answer.lower().strip().rstrip('.')
        print(predicted_answer)
        return {
            'question_id': question_id,
            'image_id': entry['image_id'],
            'question': question,
            'question_type': entry['question_type'],
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'ground_truth_answers': entry['answers'],
            'image_example_ids': [ex['question_id'] for ex in image_examples],
            'instruction_example_ids': [instr['question_id'] for instr in instruction_examples],
            'raw_response': raw_answer
        }

    except Exception as e:
        print(f"\nError processing question {question_id}: {e}")
        return {
            'question_id': question_id,
            'image_id': entry['image_id'],
            'question': question,
            'question_type': entry['question_type'],
            'predicted_answer': 'ERROR',
            'ground_truth': ground_truth,
            'ground_truth_answers': entry['answers'],
            'error': str(e)
        }

def run_hybrid_inference(model_name, k, config):
    """
    Run hybrid inference: k image examples + 5k instruction examples per query,
    both retrieved via two-stage rerank (question similarity → CLIP re-rank).
    """
    k_instr =  k
    print("="*70)
    print(f"HYBRID RERANK INFERENCE (K_img={k}, K_instr={k_instr}) - MODEL: {model_name.upper()}")
    print("="*70)

    subset_data = load_subset(config)
    questions = subset_data['data']

    print(f"\nTotal questions: {len(questions)}")

    print("Loading question embeddings...")
    all_embeddings, all_question_ids, all_questions = load_question_embeddings(config)
    print(f"Loaded {len(all_question_ids)} question embeddings")

    print("Loading CLIP image embeddings...")
    clip_embeddings = load_clip_embeddings(config)
    print(f"Loaded {len(clip_embeddings)} CLIP image embeddings")

    question_id_to_idx = {qid: idx for idx, qid in enumerate(all_question_ids)}

    subset_data_dict = {}
    for entry in questions:
        subset_data_dict[str(entry['question_id'])] = entry

    print(f"Loading pre-generated instructions for {model_name}...")
    instructions_dict = load_instructions(model_name, config)
    print(f"Loaded {len(instructions_dict)} instructions")

    model_config    = config['models'][model_name]
    inference_config = config['inference']
    prompts_config  = config['prompts']

    print(f"\nModel: {model_config['full_name']}")
    print(f"API Base: {model_config['api_base']}")
    print(f"Temperature: {inference_config['temperature']}")
    print(f"Max tokens: {inference_config['max_tokens_inference']}")

    output_dir = os.path.join(config['output']['hybrid_base_dir'], model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'predictions_hybrid_k{k}.json')

    print(f"\nRetrieving {k} image examples + {k_instr} instruction examples per question...")
    questions_with_examples = []

    for entry in tqdm(questions, mininterval=10):
        query_question_id = str(entry['question_id'])
        query_image_id    = str(entry['image_id'])

        if query_question_id not in question_id_to_idx:
            print(f"Warning: Question {query_question_id} not found in embeddings, skipping")
            continue

        query_q_embedding = all_embeddings[question_id_to_idx[query_question_id]]

        image_examples, instruction_examples = retrieve_hybrid_examples(
            query_question_id,
            query_image_id,
            query_q_embedding,
            k,
            all_embeddings,
            all_question_ids,
            clip_embeddings,
            instructions_dict,
            subset_data_dict
        )

        if len(image_examples) < k:
            print(f"Warning: Only found {len(image_examples)} image examples for question {entry['question_id']}")
        if len(instruction_examples) < k_instr:
            print(f"Warning: Only found {len(instruction_examples)} instruction examples for question {entry['question_id']}")

        questions_with_examples.append((entry, image_examples, instruction_examples))

    results = []
    max_workers = config['experiment']['max_workers']

    print(f"\nProcessing questions with {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for entry, image_examples, instruction_examples in questions_with_examples:
            future = executor.submit(
                process_single_question,
                entry,
                image_examples,
                instruction_examples,
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
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    successful = sum(1 for r in results if r['predicted_answer'] != 'ERROR')
    errors = len(results) - successful

    print("\n" + "="*70)
    print(f"HYBRID RERANK INFERENCE (K_img={k}, K_instr={k_instr}) COMPLETE")
    print("="*70)
    print(f"Total questions: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")

    print("\nSample predictions:")
    print("-"*70)
    for i, result in enumerate(results[:3]):
        print(f"\nQuestion {i+1}: {result['question']}")
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Ground truth: {result['ground_truth']}")

def main():
    parser = argparse.ArgumentParser(
        description='Hybrid VQA inference: K image examples + 5K instruction examples, '
                    'both retrieved via two-stage rerank (question similarity → CLIP).'
    )
    parser.add_argument('--model', type=str, required=True,
                        choices=['gemma', 'qwen', 'phi', 'llama', 'llava'],
                        help='Model to use for inference')
    parser.add_argument('--k', type=int, default=None,
                        help='Number of image examples K (instructions = 5K). '
                             'If not specified, runs all K values from config.')
    args = parser.parse_args()

    config = load_config()

    k_values = [args.k] if args.k is not None else config['experiment']['top_k_retrieval']

    for k in k_values:
        run_hybrid_inference(args.model, k, config)
        print("\n")

if __name__ == "__main__":
    main()
