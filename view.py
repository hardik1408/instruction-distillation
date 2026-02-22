#!/usr/bin/env python3

import os
import json
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_evaluation_results(eval_path):
    """Load evaluation results JSON."""
    if not os.path.exists(eval_path):
        return None
    
    with open(eval_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's a valid evaluation result (dict with expected fields)
    if not isinstance(data, dict):
        return None
    
    if 'total_accuracy' not in data:
        return None
    
    return data

def parse_method_from_predictions_file(predictions_file):
    """
    Extract method name from predictions file path.
    E.g., "image/gemma/predictions_zero_shot.json" -> "zero_shot"
    E.g., "image/gemma/predictions_few_shot_k3.json" -> "few_shot_k3"
    """
    basename = os.path.basename(predictions_file)
    
    # Remove "predictions_" prefix and ".json" suffix
    if basename.startswith('predictions_'):
        method_name = basename.replace('predictions_', '').replace('.json', '')
        return method_name
    
    return None

def estimate_token_cost(method_name, num_questions):
    """
    Estimate token cost for different methods.
    Method name should be like: zero_shot, few_shot_k3, approach1_k5
    """
    # Average tokens per component
    avg_image_tokens = 765
    avg_question_tokens = 10
    avg_instruction_tokens = 120
    avg_answer_tokens = 3
    
    # Prompt overhead tokens (more realistic estimates)
    zero_shot_prompt_tokens = 28  # "Answer the following question about the image. Provide a concise answer..."
    few_shot_system_prompt_tokens = 27 # "You are answering visual questions. Study these examples..."
    few_shot_example_format_tokens = 18  # "Example X: Question: ... Answer: ..."
    approach1_system_prompt_tokens = 57  # "You are an expert in visual question answering. Below are reasoning instructions..."
    approach1_instruction_format_tokens = 10  # "## Instruction X: ..."
    
    # Determine method type and K value
    k = 0
    method_type = None
    
    if method_name == 'zero_shot':
        method_type = 'zero_shot'
        k = 0
    elif method_name.startswith('few_shot_k'):
        method_type = 'few_shot'
        try:
            k = int(method_name.replace('few_shot_k', ''))
        except ValueError:
            print(f"Warning: Could not parse K from '{method_name}'")
            return None
    elif method_name.startswith('approach1_k'):
        method_type = 'approach1'
        try:
            k = int(method_name.replace('approach1_k', ''))
        except ValueError:
            print(f"Warning: Could not parse K from '{method_name}'")
            return None
    else:
        print(f"Warning: Unknown method name format '{method_name}'")
        return None
    
    # print(k)
    # Calculate tokens based on method type
    if method_type == 'zero_shot':
        input_tokens = (avg_image_tokens + avg_question_tokens + 
                       zero_shot_prompt_tokens)
        output_tokens = avg_answer_tokens
        
    elif method_type == 'few_shot':
        # K examples (each with image + Q + A + formatting) + system prompt + target image + question
        input_tokens = (few_shot_system_prompt_tokens +
                       k * (avg_image_tokens + avg_question_tokens + avg_answer_tokens + few_shot_example_format_tokens) +
                       avg_image_tokens + avg_question_tokens)
        output_tokens = avg_answer_tokens
        
    elif method_type == 'approach1':
        # K instructions (text only + formatting) + system prompt + target image + question
        input_tokens = (approach1_system_prompt_tokens +
                       k * (avg_instruction_tokens + approach1_instruction_format_tokens) +
                       avg_image_tokens + avg_question_tokens)
        output_tokens = avg_answer_tokens
    else:
        return None
    
    total_tokens_single = input_tokens + output_tokens
    
    # Calculate total for all questions
    total_tokens_all = total_tokens_single * num_questions
    
    # Cost estimate (GPT-4 Vision pricing: $0.15/1M input, $0.60/1M output)
    input_cost_single = (input_tokens * 0.15) / 1_000_000
    output_cost_single = (output_tokens * 0.60) / 1_000_000
    total_cost_single = input_cost_single + output_cost_single
    
    total_cost_all = total_cost_single * num_questions
    
    return {
        'method_type': method_type,
        'k': k,
        'avg_input_tokens': input_tokens,
        'avg_output_tokens': output_tokens,
        'total_tokens_single': total_tokens_single,  # Single inference
        'total_tokens': total_tokens_all,  # All questions (for backward compatibility)
        'total_input_tokens': input_tokens * num_questions,
        'total_output_tokens': output_tokens * num_questions,
        'estimated_cost_single_usd': round(total_cost_single, 6),  # Single inference cost
        'estimated_cost_usd': round(total_cost_all, 2)  # Total cost
    }
    # Cost estimate (GPT-4 Vision pricing: $0.15/1M input, $0.60/1M output)

def collect_results(model_name, config):
    """
    Collect all evaluation results for a model.
    Looks for metrics_*.json files and extracts method name from the predictions_file field.
    """
    print(f"\nCollecting results for {model_name}...")
    
    results = {}
    
    # Check image directory (zero-shot, few-shot)
    image_dir = os.path.join(config['output']['image_base_dir'], model_name)
    print(f"Checking image directory: {image_dir}")
    
    if os.path.exists(image_dir):
        for filename in os.listdir(image_dir):
            if filename.startswith('metrics_') and filename.endswith('.json'):
                eval_path = os.path.join(image_dir, filename)
                
                print(f"  Loading: {filename}")
                
                eval_data = load_evaluation_results(eval_path)
                if eval_data is None:
                    print(f"    Skipped (invalid format)")
                    continue
                
                # Extract method name from predictions_file field
                predictions_file = eval_data.get('predictions_file', '')
                method_name = parse_method_from_predictions_file(predictions_file)
                
                if method_name is None:
                    print(f"    Skipped (could not parse method name from: {predictions_file})")
                    continue
                
                print(f"    Method: {method_name}")
                
                token_data = estimate_token_cost(method_name, eval_data['successful_predictions'])
                if token_data is None:
                    print(f"    Skipped (could not estimate tokens)")
                    continue
                
                results[method_name] = {
                    'evaluation': eval_data,
                    'token_cost': token_data
                }
                print(f"    Added successfully")
    
    # Check instruction directory (approach1)
    instruction_dir = os.path.join(config['output']['instruction_base_dir'], model_name)
    print(f"Checking instruction directory: {instruction_dir}")
    
    if os.path.exists(instruction_dir):
        for filename in os.listdir(instruction_dir):
            if filename.startswith('metrics_') and filename.endswith('.json'):
                eval_path = os.path.join(instruction_dir, filename)
                
                print(f"  Loading: {filename}")
                
                eval_data = load_evaluation_results(eval_path)
                if eval_data is None:
                    print(f"    Skipped (invalid format)")
                    continue
                
                # Extract method name from predictions_file field
                predictions_file = eval_data.get('predictions_file', '')
                method_name = parse_method_from_predictions_file(predictions_file)
                
                if method_name is None:
                    print(f"    Skipped (could not parse method name from: {predictions_file})")
                    continue
                
                print(f"    Method: {method_name}")
                
                token_data = estimate_token_cost(method_name, eval_data['successful_predictions'])
                if token_data is None:
                    print(f"    Skipped (could not estimate tokens)")
                    continue
                
                results[method_name] = {
                    'evaluation': eval_data,
                    'token_cost': token_data
                }
                print(f"    Added successfully")
    
    print(f"\nFound {len(results)} valid results for {model_name}")
    return results

def create_visualizations(results, model_name, output_dir):
    """
    Create analysis visualizations.
    """
    if not results:
        print("No results to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    methods = sorted(results.keys())
    
    # Separate methods by type for line plots
    zero_shot_data = []
    few_shot_data = []
    approach1_data = []
    
    for method in methods:
        acc = results[method]['evaluation']['total_accuracy']
        tokens = results[method]['token_cost']['total_tokens_single']  # Changed to single inference
        cost = results[method]['token_cost']['estimated_cost_single_usd']  # Changed to single inference
        k = results[method]['token_cost']['k']
        
        if 'zero_shot' in method:
            zero_shot_data.append({'k': k, 'accuracy': acc, 'tokens': tokens, 'cost': cost, 'method': method})
        elif 'few_shot' in method:
            few_shot_data.append({'k': k, 'accuracy': acc, 'tokens': tokens, 'cost': cost, 'method': method})
        elif 'approach1' in method:
            approach1_data.append({'k': k, 'accuracy': acc, 'tokens': tokens, 'cost': cost, 'method': method})
    
    # Sort by K value
    few_shot_data = sorted(few_shot_data, key=lambda x: x['k'])
    approach1_data = sorted(approach1_data, key=lambda x: x['k'])
    
    # 1. Accuracy vs Token Cost (Line Plot)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot zero-shot as a single point
    if zero_shot_data:
        ax.scatter([zero_shot_data[0]['tokens']], [zero_shot_data[0]['accuracy']], 
                  s=200, c='red', marker='o', label='Zero-Shot', zorder=5, edgecolors='black', linewidths=1.5)
    
    # Plot few-shot as a line
    if few_shot_data:
        fs_tokens = [d['tokens'] for d in few_shot_data]
        fs_acc = [d['accuracy'] for d in few_shot_data]
        ax.plot(fs_tokens, fs_acc, marker='s', markersize=10, linewidth=2.5, 
               label='Few-Shot ICL', color='blue', alpha=0.7)
        # Add K labels
        for d in few_shot_data:
            ax.annotate(f"K={d['k']}", (d['tokens'], d['accuracy']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot approach1 as a line
    if approach1_data:
        a1_tokens = [d['tokens'] for d in approach1_data]
        a1_acc = [d['accuracy'] for d in approach1_data]
        ax.plot(a1_tokens, a1_acc, marker='^', markersize=10, linewidth=2.5, 
               label='Approach 1 (Instructions)', color='green', alpha=0.7)
        # Add K labels
        for d in approach1_data:
            ax.annotate(f"K={d['k']}", (d['tokens'], d['accuracy']), 
                       xytext=(5, -15), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Tokens per Inference', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} - Accuracy vs Token Consumption', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_accuracy_vs_tokens.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {model_name}_accuracy_vs_tokens.png")
    
    # 2. Cost-Performance Tradeoff (Line Plot)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot zero-shot as a single point
    if zero_shot_data:
        ax.scatter([zero_shot_data[0]['cost']], [zero_shot_data[0]['accuracy']], 
                  s=200, c='red', marker='o', label='Zero-Shot', zorder=5, edgecolors='black', linewidths=1.5)
    
    # Plot few-shot as a line
    if few_shot_data:
        fs_costs = [d['cost'] for d in few_shot_data]
        fs_acc = [d['accuracy'] for d in few_shot_data]
        ax.plot(fs_costs, fs_acc, marker='s', markersize=10, linewidth=2.5, 
               label='Few-Shot ICL', color='blue', alpha=0.7)
        # Add K labels
        for d in few_shot_data:
            ax.annotate(f"K={d['k']}", (d['cost'], d['accuracy']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot approach1 as a line
    if approach1_data:
        a1_costs = [d['cost'] for d in approach1_data]
        a1_acc = [d['accuracy'] for d in approach1_data]
        ax.plot(a1_costs, a1_acc, marker='^', markersize=10, linewidth=2.5, 
               label='Approach 1 (Instructions)', color='green', alpha=0.7)
        # Add K labels
        for d in approach1_data:
            ax.annotate(f"K={d['k']}", (d['cost'], d['accuracy']), 
                       xytext=(5, -15), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Cost per Inference (USD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} - Cost-Performance Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_cost_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {model_name}_cost_performance.png")
    
    # 3. K-value Impact (including K=0 from zero-shot)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add zero-shot as K=0 for both lines
    if zero_shot_data:
        zero_acc = zero_shot_data[0]['accuracy']
        
        # For few-shot line: add K=0 point
        if few_shot_data:
            fs_k = [0] + [d['k'] for d in few_shot_data]  # Add K=0
            fs_acc = [zero_acc] + [d['accuracy'] for d in few_shot_data]  # Add zero-shot accuracy
            ax.plot(fs_k, fs_acc, marker='o', markersize=10, linewidth=2.5, 
                   label='Few-Shot ICL', color='blue')
        
        # For approach1 line: add K=0 point
        if approach1_data:
            a1_k = [0] + [d['k'] for d in approach1_data]  # Add K=0
            a1_acc = [zero_acc] + [d['accuracy'] for d in approach1_data]  # Add zero-shot accuracy
            ax.plot(a1_k, a1_acc, marker='s', markersize=10, linewidth=2.5, 
                   label='Approach 1 (Instructions)', color='green')
    else:
        # If no zero-shot, plot without K=0
        if few_shot_data:
            fs_k = [d['k'] for d in few_shot_data]
            fs_acc = [d['accuracy'] for d in few_shot_data]
            ax.plot(fs_k, fs_acc, marker='o', markersize=10, linewidth=2.5, 
                   label='Few-Shot ICL', color='blue')
        
        if approach1_data:
            a1_k = [d['k'] for d in approach1_data]
            a1_acc = [d['accuracy'] for d in approach1_data]
            ax.plot(a1_k, a1_acc, marker='s', markersize=10, linewidth=2.5, 
                   label='Approach 1 (Instructions)', color='green')
    
    ax.set_xlabel('K (Number of Examples/Instructions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} - Impact of K on Accuracy', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_k_value_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {model_name}_k_value_impact.png")

def create_summary_statistics(results, model_name, output_dir):
    """
    Create summary statistics JSON and print tables.
    """
    summary = {
        'model': model_name,
        'methods': {}
    }
    
    methods = sorted(results.keys())
    
    for method in methods:
        summary['methods'][method] = {
            'accuracy': results[method]['evaluation']['total_accuracy'],
            'total_questions': results[method]['evaluation']['total_questions'],
            'successful_predictions': results[method]['evaluation']['successful_predictions'],
            'error_count': results[method]['evaluation']['error_count'],
            'token_cost': results[method]['token_cost'],
            'by_question_type': results[method]['evaluation']['by_question_type']
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, f'{model_name}_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved summary statistics to {summary_path}")
    
    # Print comparison table
    print("\n" + "="*90)
    print(f"{model_name.upper()} - PERFORMANCE & COST COMPARISON")
    print("="*90)
    print(f"{'Method':<20} {'Accuracy':<12} {'Total Tokens':<15} {'Cost (USD)':<12} {'Tokens/Accuracy':<15}")
    print("-"*90)
    
    for method in methods:
        acc = results[method]['evaluation']['total_accuracy']
        tokens = results[method]['token_cost']['total_tokens']
        cost = results[method]['token_cost']['estimated_cost_usd']
        efficiency = tokens / acc if acc > 0 else float('inf')
        
        print(f"{method:<20} {acc:<12.4f} {tokens:<15,} ${cost:<11.2f} {efficiency:<15,.0f}")
    
    # Calculate savings if applicable
    if 'few_shot_k3' in results and 'approach1_k3' in results:
        few_shot_tokens = results['few_shot_k3']['token_cost']['total_tokens']
        approach1_tokens = results['approach1_k3']['token_cost']['total_tokens']
        savings_pct = ((few_shot_tokens - approach1_tokens) / few_shot_tokens) * 100
        
        print("\n" + "="*90)
        print("TOKEN SAVINGS (Approach 1 vs Few-Shot, K=3)")
        print("="*90)
        print(f"Few-Shot K=3 tokens: {few_shot_tokens:,}")
        print(f"Approach 1 K=3 tokens: {approach1_tokens:,}")
        print(f"Token savings: {few_shot_tokens - approach1_tokens:,} ({savings_pct:.1f}%)")

def analyze_model(model_name, config):
    """
    Main analysis function for a model.
    """
    print("="*70)
    print(f"ANALYZING RESULTS - MODEL: {model_name.upper()}")
    print("="*70)
    
    # Collect all results
    results = collect_results(model_name, config)
    
    if not results:
        print(f"\nNo evaluation results found for {model_name}")
        print("Make sure you have run evaluate.py first to generate metrics_*.json files")
        return
    
    # Create output directory
    output_dir = os.path.join(config['output']['figures_dir'], model_name)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results, model_name, output_dir)
    
    # Create summary statistics
    print("\nGenerating summary statistics...")
    create_summary_statistics(results, model_name, output_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Analyze VQA experiment results')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gemma', 'qwen', 'phi', 'llama', 'llava'],
                       help='Model to analyze')
    args = parser.parse_args()
    
    config = load_config()
    analyze_model(args.model, config)

if __name__ == "__main__":
    main()
