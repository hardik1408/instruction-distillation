#!/usr/bin/env python3

import os
import json
import yaml
import base64
import argparse
import re
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

def create_zero_shot_prompt(question, config):
    """Create zero-shot prompt from template."""
    prompt_template = config['prompts']['zero_shot']
    return prompt_template.format(question=question)

def process_single_question(entry, model_config, inference_config, prompts_config):
    """
    Process a single VQA question with zero-shot inference.
    """
    question_id = entry['question_id']
    question = entry['question']
    image_path = entry['image_path']
    ground_truth = entry['answer']
    
    # Convert image to base64
    image_b64 = image_to_base64(image_path)
    
    # Create prompt
    prompt = create_zero_shot_prompt(question, {'prompts': prompts_config})
    
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
            max_tokens=inference_config['max_tokens_inference'],
            temperature=inference_config['temperature']
        )
        
        raw_answer = response.choices[0].message.content.strip()
        
        # Extract answer (remove any extra text)
        # Try to find the actual answer after "Answer:" or just take the whole response
        if "Answer:" in raw_answer:
            predicted_answer = raw_answer.split("Answer:")[-1].strip()
        else:
            predicted_answer = raw_answer
        
        # Clean up answer (lowercase, remove punctuation for consistency)
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

def run_zero_shot_inference(model_name, config):
    """
    Run zero-shot inference on VQA subset.
    """
    print("="*70)
    print(f"ZERO-SHOT INFERENCE - MODEL: {model_name.upper()}")
    print("="*70)
    
    # Load subset
    subset_data = load_subset(config)
    questions = subset_data['data']
    
    print(f"\nTotal questions: {len(questions)}")
    
    # Get model config
    model_config = config['models'][model_name]
    inference_config = config['inference']
    prompts_config = config['prompts']
    
    print(f"Model: {model_config['full_name']}")
    print(f"API Base: {model_config['api_base']}")
    print(f"Temperature: {inference_config['temperature']}")
    print(f"Max tokens: {inference_config['max_tokens_inference']}")
    
    # Create output directory
    output_dir = os.path.join(config['output']['image_base_dir'], model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'predictions_zero_shot.json')
    
    # Process questions in parallel
    results = []
    max_workers = config['experiment']['max_workers']
    
    print(f"\nProcessing questions with {max_workers} workers...\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for entry in questions:
            future = executor.submit(
                process_single_question,
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
    
    # Save predictions
    print(f"\nSaving predictions to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Quick statistics
    successful = sum(1 for r in results if r['predicted_answer'] != 'ERROR')
    errors = len(results) - successful
    
    print("\n" + "="*70)
    print("ZERO-SHOT INFERENCE COMPLETE")
    print("="*70)
    print(f"Total questions: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")
    
    # Show some sample predictions
    print("\nSample predictions:")
    print("-"*70)
    for i, result in enumerate(results[:5]):
        print(f"\nQuestion {i+1}: {result['question']}")
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Ground truth: {result['ground_truth']}")

def main():
    parser = argparse.ArgumentParser(description='Zero-shot VQA inference')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gemma', 'qwen', 'phi', 'llama', 'llava'],
                       help='Model to use for inference')
    args = parser.parse_args()
    
    config = load_config()
    run_zero_shot_inference(args.model, config)

if __name__ == "__main__":
    main()