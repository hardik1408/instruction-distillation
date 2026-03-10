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
    # Return a dict: image_id (str) -> embedding (np.array)
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

def retrieve_similar_by_question_then_clip(query_question_id, query_image_id, query_q_embedding,
                                            k, all_q_embeddings, all_question_ids,
                                            clip_embeddings, instructions_dict, subset_data_dict):
    """
    Retrieve K most similar instructions using a two-stage approach:
      1. Fetch 3K candidates by question text similarity (that have instructions).
      2. Re-rank those candidates by CLIP image similarity.
      3. Return top K.
    """
    k_prime = 3 * k

    # Stage 1: question text similarity
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

        if neighbor_question_id in instructions_dict:
            instr_data = instructions_dict[neighbor_question_id]
            neighbor_image_id = subset_data_dict.get(neighbor_question_id, {}).get('image_id')
            candidates.append({
                'question_id': neighbor_question_id,
                'image_id': str(neighbor_image_id) if neighbor_image_id is not None else None,
                'question': instr_data['question'],
                'answer': instr_data['answer'],
                'instruction': instr_data['instruction'],
                'q_similarity': float(similarities[idx])
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

    return candidates[:k]

def create_approach1_prompt(question, retrieved_instructions, config):
    """Create instruction-based prompt with question and answer for each example."""
    prompts_config = config['prompts']

    prompt_parts = [prompts_config['approach1_system']]

    for idx, instr_data in enumerate(retrieved_instructions, 1):
        instruction_text = f"""## Instruction {idx}:
Question: {instr_data['question']}
Answer: {instr_data['answer']}

Reasoning Strategy:
{instr_data['instruction']}"""
        prompt_parts.append(instruction_text)

    target_text = prompts_config['approach1_target'].format(question=question)
    prompt_parts.append(target_text)

    return "\n\n".join(prompt_parts)

def process_single_question(entry, retrieved_instructions, model_config, inference_config, prompts_config):
    """
    Process a single VQA question with instruction-based inference.
    """
    question_id = entry['question_id']
    question = entry['question']
    image_path = entry['image_path']
    ground_truth = entry['answer']

    target_image_b64 = image_to_base64(image_path)

    prompt = create_approach1_prompt(question, retrieved_instructions, {'prompts': prompts_config})

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{target_image_b64}"
                    }
                }
            ]
        }
    ]

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
            'retrieved_instruction_ids': [instr['question_id'] for instr in retrieved_instructions],
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

def run_approach1_rerank_inference(model_name, k, config):
    """
    Run instruction-based inference with hybrid retrieval (question similarity + CLIP re-rank).
    """
    print("="*70)
    print(f"APPROACH 1 RERANK INFERENCE (K={k}) - MODEL: {model_name.upper()}")
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

    # Build lookup dict for image_id per question (needed for candidate re-ranking)
    subset_data_dict = {}
    for entry in questions:
        subset_data_dict[str(entry['question_id'])] = entry

    print(f"Loading pre-generated instructions for {model_name}...")
    instructions_dict = load_instructions(model_name, config)
    print(f"Loaded {len(instructions_dict)} instructions")

    model_config = config['models'][model_name]
    inference_config = config['inference']
    prompts_config = config['prompts']

    print(f"\nModel: {model_config['full_name']}")
    print(f"API Base: {model_config['api_base']}")
    print(f"Temperature: {inference_config['temperature']}")
    print(f"Max tokens: {inference_config['max_tokens_inference']}")

    output_dir = os.path.join(config['output']['instruction_base_dir'], model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'rerank_predictions_approach1_k{k}.json')

    print(f"\nRetrieving {k} instructions per question (3K by question sim, re-ranked by CLIP)...")
    questions_with_instructions = []

    for entry in tqdm(questions, mininterval=10):
        query_question_id = str(entry['question_id'])
        query_image_id = str(entry['image_id'])

        if query_question_id not in question_id_to_idx:
            print(f"Warning: Question {query_question_id} not found in embeddings, skipping")
            continue

        query_q_embedding = all_embeddings[question_id_to_idx[query_question_id]]

        retrieved_instructions = retrieve_similar_by_question_then_clip(
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

        if len(retrieved_instructions) < k:
            print(f"Warning: Only found {len(retrieved_instructions)} instructions for question {entry['question_id']}")

        questions_with_instructions.append((entry, retrieved_instructions))

    results = []
    max_workers = config['experiment']['max_workers']

    print(f"\nProcessing questions with {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for entry, retrieved_instructions in questions_with_instructions:
            future = executor.submit(
                process_single_question,
                entry,
                retrieved_instructions,
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
    print(f"APPROACH 1 RERANK INFERENCE (K={k}) COMPLETE")
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
    parser = argparse.ArgumentParser(description='Approach 1 VQA inference with question+CLIP hybrid retrieval')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gemma', 'qwen', 'phi', 'llama', 'llava'],
                       help='Model to use for inference')
    parser.add_argument('--k', type=int, default=None,
                       help='Number of instructions to retrieve (if not specified, runs all K values from config)')
    args = parser.parse_args()

    config = load_config()

    if args.k is not None:
        k_values = [args.k]
    else:
        k_values = config['experiment']['top_k_retrieval']

    for k in k_values:
        run_approach1_rerank_inference(args.model, k, config)
        print("\n")

if __name__ == "__main__":
    main()
