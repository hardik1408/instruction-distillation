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
from sklearn.neighbors import NearestNeighbors

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_subset(config):
    """Load the VQA subset data."""
    subset_path = config['output']['subset_path']
    with open(subset_path, 'r') as f:
        return json.load(f)

def load_clip_embeddings(config):
    """Load precomputed CLIP embeddings."""
    embeddings_path = config['output']['embeddings_path']
    data = np.load(embeddings_path, allow_pickle=True).item()
    return data['embeddings'], data['image_ids']

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

def build_knn_index(embeddings, k):
    """Build KNN index for retrieval."""
    nn = NearestNeighbors(
        n_neighbors=k + 1,  # +1 because we'll exclude the query image itself
        metric='cosine',
        algorithm='auto'
    )
    nn.fit(embeddings)
    return nn

def retrieve_similar_instructions(query_image_id, query_idx, k, nn_index, image_ids, instructions_dict):
    """
    Retrieve K most similar instructions using CLIP embeddings.
    """
    # Get neighbors
    distances, indices = nn_index.kneighbors(
        query_idx.reshape(1, -1),
        return_distance=True
    )
    
    # Filter out the query image itself and get valid instructions
    retrieved_instructions = []
    for i, idx in enumerate(indices[0]):  # Changed: use enumerate to get position in distances array
        neighbor_image_id = image_ids[idx]
        
        # Skip if it's the same image
        if neighbor_image_id == query_image_id:
            continue
        
        # Find instruction for this image (take any question from this image)
        found = False
        for question_id, instr_data in instructions_dict.items():
            if str(instr_data['image_id']) == neighbor_image_id:
                retrieved_instructions.append({
                    'question_id': question_id,
                    'image_id': neighbor_image_id,
                    'instruction': instr_data['instruction'],
                    'distance': distances[0][i]  # Changed: use i instead of idx
                })
                found = True
                break
        
        if len(retrieved_instructions) == k:
            break
    
    return retrieved_instructions

def create_approach1_prompt(question, retrieved_instructions, config):
    """Create instruction-based prompt."""
    prompts_config = config['prompts']
    
    # System message
    prompt_parts = [prompts_config['approach1_system']]
    
    # Add retrieved instructions
    for idx, instr_data in enumerate(retrieved_instructions, 1):
        instruction_text = prompts_config['approach1_instruction'].format(
            idx=idx,
            instruction=instr_data['instruction']
        )
        prompt_parts.append(instruction_text)
    
    # Add target question
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
    
    # Convert target image to base64
    target_image_b64 = image_to_base64(image_path)
    
    # Create prompt
    prompt = create_approach1_prompt(question, retrieved_instructions, {'prompts': prompts_config})
    # print(prompt)
    # Build messages (only target image, no example images)
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
    
    # Call VLM API
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
        
        # Extract answer
        if "Answer:" in raw_answer:
            predicted_answer = raw_answer.split("Answer:")[-1].strip()
        else:
            predicted_answer = raw_answer
        
        # Clean up answer
        predicted_answer = predicted_answer.lower().strip().rstrip('.')
        # print(predicted_answer)
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

def run_approach1_inference(model_name, k, config):
    """
    Run instruction-based (Approach 1) inference on VQA subset.
    """
    print("="*70)
    print(f"APPROACH 1 INFERENCE (K={k}) - MODEL: {model_name.upper()}")
    print("="*70)
    
    # Load data
    subset_data = load_subset(config)
    questions = subset_data['data']
    
    print(f"\nTotal questions: {len(questions)}")
    
    # Load CLIP embeddings
    print("Loading CLIP embeddings...")
    embeddings, image_ids = load_clip_embeddings(config)
    
    # Load pre-generated instructions
    print(f"Loading pre-generated instructions for {model_name}...")
    instructions_dict = load_instructions(model_name, config)
    print(f"Loaded {len(instructions_dict)} instructions")
    
    # Create lookup dictionaries
    image_id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}
    
    # Build KNN index
    print(f"Building KNN index for K={k}...")
    nn_index = build_knn_index(embeddings, k)
    
    # Get model config
    model_config = config['models'][model_name]
    inference_config = config['inference']
    prompts_config = config['prompts']
    
    print(f"\nModel: {model_config['full_name']}")
    print(f"API Base: {model_config['api_base']}")
    print(f"Temperature: {inference_config['temperature']}")
    print(f"Max tokens: {inference_config['max_tokens_inference']}")
    
    # Create output directory
    output_dir = os.path.join(config['output']['instruction_base_dir'], model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'predictions_approach1_k{k}.json')
    
    # Prepare questions with retrieved instructions
    print(f"\nRetrieving {k} similar instructions for each question...")
    questions_with_instructions = []
    
    for entry in tqdm(questions):
        query_image_id = str(entry['image_id'])
        
        # Get embedding index for this image
        if query_image_id not in image_id_to_idx:
            print(f"Warning: Image {query_image_id} not found in embeddings, skipping")
            continue
        
        query_idx = embeddings[image_id_to_idx[query_image_id]]
        
        # Retrieve similar instructions
        retrieved_instructions = retrieve_similar_instructions(
            query_image_id,
            query_idx,
            k,
            nn_index,
            image_ids,
            instructions_dict
        )
        
        if len(retrieved_instructions) < k:
            print(f"Warning: Only found {len(retrieved_instructions)} instructions for question {entry['question_id']}")
        
        questions_with_instructions.append((entry, retrieved_instructions))
    
    # Process questions in parallel
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
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
    
    # Save predictions
    print(f"\nSaving predictions to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Quick statistics
    successful = sum(1 for r in results if r['predicted_answer'] != 'ERROR')
    errors = len(results) - successful
    
    print("\n" + "="*70)
    print(f"APPROACH 1 INFERENCE (K={k}) COMPLETE")
    print("="*70)
    print(f"Total questions: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")
    
    # Show some sample predictions
    print("\nSample predictions:")
    print("-"*70)
    for i, result in enumerate(results[:3]):
        print(f"\nQuestion {i+1}: {result['question']}")
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Ground truth: {result['ground_truth']}")

def main():
    parser = argparse.ArgumentParser(description='Approach 1 (Instruction-based) VQA inference')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gemma', 'qwen', 'phi', 'llama', 'llava'],
                       help='Model to use for inference')
    parser.add_argument('--k', type=int, default=None,
                       help='Number of instructions to retrieve (if not specified, runs all K values from config)')
    args = parser.parse_args()
    
    config = load_config()
    
    # Determine which K values to run
    if args.k is not None:
        k_values = [args.k]
    else:
        k_values = config['experiment']['top_k_retrieval']
    
    # Run for each K value
    for k in k_values:
        run_approach1_inference(args.model, k, config)
        print("\n")

if __name__ == "__main__":
    main()
