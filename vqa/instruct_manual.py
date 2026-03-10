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
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_subset(config):
    """Load the VQA subset data."""
    subset_path = config['output']['subset_path']
    with open(subset_path, 'r') as f:
        return json.load(f)

def image_to_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def create_instruction_generation_prompt(question, answer, question_type, config):
    """Create instruction generation prompt from template."""
    prompt_template = config['prompts']['instruction_generation']
    return prompt_template.format(
        question=question,
        answer=answer,
        question_type=question_type
    )

def generate_instruction_for_entry(entry, model_config, inference_config, prompts_config):
    """
    Generate reasoning instruction for a single image-question-answer triplet.
    """
    question_id = entry['question_id']
    image_id = entry['image_id']
    question = entry['question']
    answer = entry['answer']
    question_type = entry['question_type']
    image_path = entry['image_path']
    
    # Convert image to base64
    image_b64 = image_to_base64(image_path)
    
    # Create prompt
    prompt = create_instruction_generation_prompt(
        question, 
        answer, 
        question_type, 
        {'prompts': prompts_config}
    )
    
    # Build messages for API
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
    
    # Call VLM API
    client = OpenAI(
        base_url=model_config['api_base'],
        api_key=model_config['api_key']
    )
    
    try:
        response = client.chat.completions.create(
            model=model_config['full_name'],
            messages=messages,
            max_tokens=inference_config['max_tokens_instruction'],
            temperature=inference_config['temperature']
        )
        
        instruction = response.choices[0].message.content.strip()
        # print(instruction)
        return {
            'question_id': question_id,
            'image_id': image_id,
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'instruction': instruction,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"\nError generating instruction for question {question_id}: {e}")
        return {
            'question_id': question_id,
            'image_id': image_id,
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'instruction': None,
            'status': 'error',
            'error': str(e)
        }

def generate_per_image_instructions(model_name, config):
    """
    Generate reasoning instructions for all entries in the subset.
    """
    print("="*70)
    print(f"GENERATING PER-IMAGE INSTRUCTIONS - MODEL: {model_name.upper()}")
    print("="*70)
    
    # Load subset
    subset_data = load_subset(config)
    questions = subset_data['data']
    
    print(f"\nTotal entries to process: {len(questions)}")
    
    # Get model config
    model_config = config['models'][model_name]
    inference_config = config['inference']
    prompts_config = config['prompts']
    
    print(f"\nModel: {model_config['full_name']}")
    print(f"API Base: {model_config['api_base']}")
    print(f"Temperature: {inference_config['temperature']}")
    print(f"Max tokens: {inference_config['max_tokens_instruction']}")
    
    # Create output directory
    output_dir = os.path.join(config['output']['instruction_base_dir'], model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'per_image_instructions.json')
    
    # Check if already exists and ask to continue
    if os.path.exists(output_path):
        print(f"\nWarning: Output file already exists: {output_path}")
        response = input("Do you want to overwrite it? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborting.")
            return
    
    # Process entries in parallel
    results = []
    instructions_dict = {}
    max_workers = config['experiment']['max_workers']
    
    print(f"\nGenerating instructions with {max_workers} workers...")
    print("This may take a while...\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for entry in questions:
            future = executor.submit(
                generate_instruction_for_entry,
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
                
                # Store in dictionary keyed by question_id
                if result['status'] == 'success':
                    instructions_dict[str(result['question_id'])] = {
                        'image_id': result['image_id'],
                        'question': result['question'],
                        'answer': result['answer'],
                        'question_type': result['question_type'],
                        'instruction': result['instruction']
                    }
            except Exception as e:
                print(f"Error: {e}")
    
    # Save instructions
    print(f"\nSaving instructions to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(instructions_dict, f, indent=2)
    
    # Statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    errors = len(results) - successful
    
    # Statistics by question type
    type_stats = {}
    for result in results:
        qtype = result['question_type']
        if qtype not in type_stats:
            type_stats[qtype] = {'total': 0, 'success': 0, 'error': 0}
        type_stats[qtype]['total'] += 1
        if result['status'] == 'success':
            type_stats[qtype]['success'] += 1
        else:
            type_stats[qtype]['error'] += 1
    
    print("\n" + "="*70)
    print("INSTRUCTION GENERATION COMPLETE")
    print("="*70)
    print(f"Total entries: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")
    
    print("\nStatistics by question type:")
    print("-"*70)
    print(f"{'Type':<15} {'Total':<10} {'Success':<10} {'Errors':<10}")
    print("-"*70)
    for qtype in sorted(type_stats.keys()):
        stats = type_stats[qtype]
        print(f"{qtype:<15} {stats['total']:<10} {stats['success']:<10} {stats['error']:<10}")
    
    # Show sample instructions
    print("\nSample generated instructions:")
    print("-"*70)
    sample_count = 0
    for result in results:
        if result['status'] == 'success' and sample_count < 3:
            print(f"\nQuestion Type: {result['question_type']}")
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"Instruction: {result['instruction'][:200]}...")
            sample_count += 1
    
    # Save detailed results with errors
    error_log_path = os.path.join(output_dir, 'instruction_generation_log.json')
    print(f"\nSaving detailed log to {error_log_path}...")
    with open(error_log_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Generate per-image reasoning instructions')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gemma', 'qwen', 'phi', 'llama', 'llava'],
                       help='Model to use for instruction generation')
    args = parser.parse_args()
    
    config = load_config()
    generate_per_image_instructions(args.model, config)

if __name__ == "__main__":
    main()