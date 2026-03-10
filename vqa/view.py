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
    E.g., "image/gemma/predictions_zero_shot.json"               -> "zero_shot"
    E.g., "image/gemma/predictions_few_shot_k3.json"             -> "few_shot_k3"
    E.g., "image/gemma/rerank_predictions_few_shot_k3.json"      -> "few_shot_rerank_k3"
    E.g., "instruction/gemma/rerank_predictions_approach1_k3.json" -> "approach1_rerank_k3"
    E.g., "hybrid/gemma/predictions_hybrid_k3.json"              -> "hybrid_k3"
    """
    basename = os.path.basename(predictions_file)

    if basename.startswith('predictions_'):
        method_name = basename.replace('predictions_', '').replace('.json', '')
        return method_name

    if basename.startswith('rerank_predictions_'):
        # e.g. rerank_predictions_few_shot_k3.json -> few_shot_rerank_k3
        # e.g. rerank_predictions_approach1_k3.json -> approach1_rerank_k3
        inner = basename.replace('rerank_predictions_', '').replace('.json', '')
        if '_k' in inner:
            parts = inner.rsplit('_k', 1)
            method_base = parts[0]   # "few_shot" or "approach1"
            k_val = parts[1]         # "3"
            return f"{method_base}_rerank_k{k_val}"

    return None

def estimate_token_cost(method_name, num_questions):
    """
    Estimate token cost for different methods.
    Method name should be like: zero_shot, few_shot_k3, approach1_k5,
                                 few_shot_rerank_k3, approach1_rerank_k5
    """
    # Average tokens per component
    avg_image_tokens = 765
    avg_question_tokens = 10
    avg_instruction_tokens = 120
    avg_answer_tokens = 3

    # Prompt overhead tokens (more realistic estimates)
    zero_shot_prompt_tokens = 28
    few_shot_system_prompt_tokens = 27
    few_shot_example_format_tokens = 18
    approach1_system_prompt_tokens = 57
    approach1_instruction_format_tokens = 10

    k = 0
    method_type = None

    if method_name == 'zero_shot':
        method_type = 'zero_shot'
        k = 0
    elif method_name.startswith('few_shot_rerank_k'):
        method_type = 'few_shot_rerank'
        try:
            k = int(method_name.replace('few_shot_rerank_k', ''))
        except ValueError:
            print(f"Warning: Could not parse K from '{method_name}'")
            return None
    elif method_name.startswith('few_shot_k'):
        method_type = 'few_shot'
        try:
            k = int(method_name.replace('few_shot_k', ''))
        except ValueError:
            print(f"Warning: Could not parse K from '{method_name}'")
            return None
    elif method_name.startswith('approach1_rerank_k'):
        method_type = 'approach1_rerank'
        try:
            k = int(method_name.replace('approach1_rerank_k', ''))
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
    elif method_name.startswith('hybrid_k'):
        method_type = 'hybrid'
        try:
            k = int(method_name.replace('hybrid_k', ''))
        except ValueError:
            print(f"Warning: Could not parse K from '{method_name}'")
            return None
    else:
        print(f"Warning: Unknown method name format '{method_name}'")
        return None

    # Calculate tokens — rerank uses the same prompt structure as its base method
    if method_type == 'zero_shot':
        input_tokens = (avg_image_tokens + avg_question_tokens +
                       zero_shot_prompt_tokens)
        output_tokens = avg_answer_tokens

    elif method_type in ('few_shot', 'few_shot_rerank'):
        input_tokens = (few_shot_system_prompt_tokens +
                       k * (avg_image_tokens + avg_question_tokens + avg_answer_tokens + few_shot_example_format_tokens) +
                       avg_image_tokens + avg_question_tokens)
        output_tokens = avg_answer_tokens

    elif method_type in ('approach1', 'approach1_rerank'):
        input_tokens = (approach1_system_prompt_tokens +
                       k * (avg_instruction_tokens + approach1_instruction_format_tokens) +
                       avg_image_tokens + avg_question_tokens)
        output_tokens = avg_answer_tokens

    elif method_type == 'hybrid':
        # K image examples + 5K instruction examples
        hybrid_system_prompt_tokens = 45
        hybrid_image_format_tokens = 25     # "## Visual Example {idx}: Question: Answer:"
        hybrid_instruction_format_tokens = 25  # "## Reasoning Instruction {idx}: Q/A/Strategy:"
        k_instr = 5 * k
        input_tokens = (hybrid_system_prompt_tokens +
                        k * (avg_image_tokens + avg_question_tokens + avg_answer_tokens + hybrid_image_format_tokens) +
                        k_instr * (avg_instruction_tokens + hybrid_instruction_format_tokens) +
                        avg_image_tokens + avg_question_tokens)
        output_tokens = avg_answer_tokens
    else:
        return None

    total_tokens_single = input_tokens + output_tokens
    total_tokens_all = total_tokens_single * num_questions

    input_cost_single = (input_tokens * 0.15) / 1_000_000
    output_cost_single = (output_tokens * 0.60) / 1_000_000
    total_cost_single = input_cost_single + output_cost_single
    total_cost_all = total_cost_single * num_questions

    return {
        'method_type': method_type,
        'k': k,
        'avg_input_tokens': input_tokens,
        'avg_output_tokens': output_tokens,
        'total_tokens_single': total_tokens_single,
        'total_tokens': total_tokens_all,
        'total_input_tokens': input_tokens * num_questions,
        'total_output_tokens': output_tokens * num_questions,
        'estimated_cost_single_usd': round(total_cost_single, 6),
        'estimated_cost_usd': round(total_cost_all, 2)
    }

BASELINE_TYPES = {'zero_shot', 'few_shot', 'approach1'}
RERANK_TYPES   = {'zero_shot', 'few_shot_rerank', 'approach1_rerank'}
HYBRID_TYPES   = {'zero_shot', 'few_shot_rerank', 'approach1_rerank', 'hybrid'}

def collect_results(model_name, config, rerank=False, hybrid=False):
    """
    Collect evaluation results for a model.

    rerank=False, hybrid=False (default):
        Loads metrics_*.json from image/ and instruction/ dirs.
        Returns zero_shot / few_shot / approach1 entries.
    rerank=True:
        Loads metrics_*.json + rerank_metrics_*.json.
        Returns zero_shot + few_shot_rerank + approach1_rerank entries.
    hybrid=True:
        Loads rerank_metrics_*.json from image/ and instruction/ dirs,
        plus hybrid_metrics_*.json from hybrid/ dir.
        Returns zero_shot + few_shot_rerank + approach1_rerank + hybrid entries.
    """
    mode = 'hybrid' if hybrid else ('rerank' if rerank else 'baseline')
    print(f"\nCollecting results for {model_name} (mode={mode})...")

    allowed_types = HYBRID_TYPES if hybrid else (RERANK_TYPES if rerank else BASELINE_TYPES)
    all_results = {}

    # Each entry: (directory_path, accepted filename prefixes)
    scan_targets = [
        (config['output']['image_base_dir'],       ['metrics_', 'rerank_metrics_']),
        (config['output']['instruction_base_dir'], ['metrics_', 'rerank_metrics_']),
    ]
    if hybrid:
        scan_targets.append((config['output']['hybrid_base_dir'], ['hybrid_metrics_']))

    for base_dir, accepted_prefixes in scan_targets:
        scan_dir = os.path.join(base_dir, model_name)
        print(f"Checking directory: {scan_dir}")

        if not os.path.exists(scan_dir):
            continue

        for filename in os.listdir(scan_dir):
            if not filename.endswith('.json'):
                continue
            if not any(filename.startswith(p) for p in accepted_prefixes):
                continue

            eval_path = os.path.join(scan_dir, filename)
            print(f"  Loading: {filename}")

            eval_data = load_evaluation_results(eval_path)
            if eval_data is None:
                print(f"    Skipped (invalid format)")
                continue

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

            if token_data['method_type'] not in allowed_types:
                print(f"    Skipped (method type '{token_data['method_type']}' not in mode)")
                continue

            all_results[method_name] = {
                'evaluation': eval_data,
                'token_cost': token_data
            }
            print(f"    Added successfully")

    print(f"\nFound {len(all_results)} valid results for {model_name}")
    return all_results

def _bucket_results(results):
    """Split result entries into per-method-type lists, sorted by K."""
    buckets = defaultdict(list)
    for method, data in results.items():
        acc    = data['evaluation']['total_accuracy']
        tokens = data['token_cost']['total_tokens_single']
        cost   = data['token_cost']['estimated_cost_single_usd']
        k      = data['token_cost']['k']
        mtype  = data['token_cost']['method_type']
        buckets[mtype].append({'k': k, 'accuracy': acc, 'tokens': tokens,
                               'cost': cost, 'method': method})
    for key in buckets:
        buckets[key].sort(key=lambda x: x['k'])
    return buckets


def _annotate_line(ax, data_list, x_key, offset_y=5):
    for d in data_list:
        ax.annotate(f"K={d['k']}", (d[x_key], d['accuracy']),
                    xytext=(5, offset_y), textcoords='offset points', fontsize=8)


def create_visualizations(results, model_name, output_dir):
    """
    Baseline graphs: zero-shot / few-shot ICL / approach1.
    Output filenames have no prefix.
    """
    if not results:
        print("No results to visualize")
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    b = _bucket_results(results)
    zs  = b.get('zero_shot', [])
    fs  = b.get('few_shot', [])
    a1  = b.get('approach1', [])

    def _save(fig, name):
        path = os.path.join(output_dir, f'{model_name}_{name}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(path)}")

    # ── 1. Accuracy vs Tokens ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    if zs:
        ax.scatter([zs[0]['tokens']], [zs[0]['accuracy']], s=200, c='red',
                   marker='o', label='Zero-Shot', zorder=5, edgecolors='black', linewidths=1.5)
    if fs:
        ax.plot([d['tokens'] for d in fs], [d['accuracy'] for d in fs],
                marker='s', markersize=10, linewidth=2.5, label='Few-Shot ICL', color='blue', alpha=0.8)
        _annotate_line(ax, fs, 'tokens', offset_y=5)
    if a1:
        ax.plot([d['tokens'] for d in a1], [d['accuracy'] for d in a1],
                marker='^', markersize=10, linewidth=2.5, label='Approach 1 (Instructions)', color='green', alpha=0.8)
        _annotate_line(ax, a1, 'tokens', offset_y=-15)
    ax.set_xlabel('Tokens per Inference', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} - Accuracy vs Token Consumption', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, 'accuracy_vs_tokens')

    # ── 2. Cost-Performance ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    if zs:
        ax.scatter([zs[0]['cost']], [zs[0]['accuracy']], s=200, c='red',
                   marker='o', label='Zero-Shot', zorder=5, edgecolors='black', linewidths=1.5)
    if fs:
        ax.plot([d['cost'] for d in fs], [d['accuracy'] for d in fs],
                marker='s', markersize=10, linewidth=2.5, label='Few-Shot ICL', color='blue', alpha=0.8)
        _annotate_line(ax, fs, 'cost', offset_y=5)
    if a1:
        ax.plot([d['cost'] for d in a1], [d['accuracy'] for d in a1],
                marker='^', markersize=10, linewidth=2.5, label='Approach 1 (Instructions)', color='green', alpha=0.8)
        _annotate_line(ax, a1, 'cost', offset_y=-15)
    ax.set_xlabel('Cost per Inference (USD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} - Cost-Performance Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, 'cost_performance')

    # ── 3. K-value Impact ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    zero_acc = zs[0]['accuracy'] if zs else None

    def _with_zero(data_list):
        if zero_acc is not None:
            return [0] + [d['k'] for d in data_list], [zero_acc] + [d['accuracy'] for d in data_list]
        return [d['k'] for d in data_list], [d['accuracy'] for d in data_list]

    if fs:
        k_vals, accs = _with_zero(fs)
        ax.plot(k_vals, accs, marker='o', markersize=10, linewidth=2.5, label='Few-Shot ICL', color='blue')
    if a1:
        k_vals, accs = _with_zero(a1)
        ax.plot(k_vals, accs, marker='s', markersize=10, linewidth=2.5, label='Approach 1 (Instructions)', color='green')
    ax.set_xlabel('K (Number of Examples/Instructions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} - Impact of K on Accuracy', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, 'k_value_impact')


def create_rerank_visualizations(results, model_name, output_dir):
    """
    Rerank graphs: zero-shot (baseline) / few-shot+rerank / approach1+rerank.
    Output filenames are prefixed with 'rerank_'.
    Uses distinct colors (orange / purple) so plots are visually different.
    """
    if not results:
        print("No results to visualize")
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    b   = _bucket_results(results)
    zs  = b.get('zero_shot', [])
    fsr = b.get('few_shot_rerank', [])
    a1r = b.get('approach1_rerank', [])

    def _save(fig, name):
        path = os.path.join(output_dir, f'{model_name}_rerank_{name}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(path)}")

    # ── 1. Accuracy vs Tokens ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    if zs:
        ax.scatter([zs[0]['tokens']], [zs[0]['accuracy']], s=200, c='red',
                   marker='o', label='Zero-Shot (baseline)', zorder=5,
                   edgecolors='black', linewidths=1.5)
    if fsr:
        ax.plot([d['tokens'] for d in fsr], [d['accuracy'] for d in fsr],
                marker='D', markersize=10, linewidth=2.5,
                label='Few-Shot ICL + Rerank', color='darkorange', alpha=0.85)
        _annotate_line(ax, fsr, 'tokens', offset_y=5)
    if a1r:
        ax.plot([d['tokens'] for d in a1r], [d['accuracy'] for d in a1r],
                marker='P', markersize=10, linewidth=2.5,
                label='Instruction Distillation + Rerank', color='purple', alpha=0.85)
        _annotate_line(ax, a1r, 'tokens', offset_y=-15)
    ax.set_xlabel('Tokens per Inference', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} [Rerank] - Accuracy vs Token Consumption',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, 'accuracy_vs_tokens')

    # ── 2. Cost-Performance ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    if zs:
        ax.scatter([zs[0]['cost']], [zs[0]['accuracy']], s=200, c='red',
                   marker='o', label='Zero-Shot (baseline)', zorder=5,
                   edgecolors='black', linewidths=1.5)
    if fsr:
        ax.plot([d['cost'] for d in fsr], [d['accuracy'] for d in fsr],
                marker='D', markersize=10, linewidth=2.5,
                label='Few-Shot ICL + Rerank', color='darkorange', alpha=0.85)
        _annotate_line(ax, fsr, 'cost', offset_y=5)
    if a1r:
        ax.plot([d['cost'] for d in a1r], [d['accuracy'] for d in a1r],
                marker='P', markersize=10, linewidth=2.5,
                label='Instruction Distillation + Rerank', color='purple', alpha=0.85)
        _annotate_line(ax, a1r, 'cost', offset_y=-15)
    ax.set_xlabel('Cost per Inference (USD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} [Rerank] - Cost-Performance Tradeoff',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, 'cost_performance')

    # ── 3. K-value Impact ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    zero_acc = zs[0]['accuracy'] if zs else None

    def _with_zero(data_list):
        if zero_acc is not None:
            return [0] + [d['k'] for d in data_list], [zero_acc] + [d['accuracy'] for d in data_list]
        return [d['k'] for d in data_list], [d['accuracy'] for d in data_list]

    if fsr:
        k_vals, accs = _with_zero(fsr)
        ax.plot(k_vals, accs, marker='D', markersize=10, linewidth=2.5,
                label='Few-Shot ICL + Rerank', color='darkorange')
    if a1r:
        k_vals, accs = _with_zero(a1r)
        ax.plot(k_vals, accs, marker='P', markersize=10, linewidth=2.5,
                label='Instruction Distillation + Rerank', color='purple')
    ax.set_xlabel('K (Number of Examples/Instructions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} [Rerank] - Impact of K on Accuracy',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, 'k_value_impact')

def create_hybrid_visualizations(results, model_name, output_dir):
    """
    Hybrid graphs: zero-shot (baseline) / few-shot+rerank / approach1+rerank / hybrid.
    Output filenames are prefixed with 'hybrid_'.
    Hybrid uses teal; rerank series keep their existing orange/purple colours.
    """
    if not results:
        print("No results to visualize")
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    b   = _bucket_results(results)
    zs  = b.get('zero_shot', [])
    fsr = b.get('few_shot_rerank', [])
    a1r = b.get('approach1_rerank', [])
    hyb = b.get('hybrid', [])

    def _save(fig, name):
        path = os.path.join(output_dir, f'{model_name}_hybrid_{name}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(path)}")

    # ── 1. Accuracy vs Tokens ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    if zs:
        ax.scatter([zs[0]['tokens']], [zs[0]['accuracy']], s=200, c='red',
                   marker='o', label='Zero-Shot (baseline)', zorder=5,
                   edgecolors='black', linewidths=1.5)
    if fsr:
        ax.plot([d['tokens'] for d in fsr], [d['accuracy'] for d in fsr],
                marker='D', markersize=10, linewidth=2.5,
                label='Few-Shot ICL + Rerank', color='darkorange', alpha=0.85)
        _annotate_line(ax, fsr, 'tokens', offset_y=5)
    if a1r:
        ax.plot([d['tokens'] for d in a1r], [d['accuracy'] for d in a1r],
                marker='P', markersize=10, linewidth=2.5,
                label='Approach 1 + Rerank', color='purple', alpha=0.85)
        _annotate_line(ax, a1r, 'tokens', offset_y=-15)
    if hyb:
        ax.plot([d['tokens'] for d in hyb], [d['accuracy'] for d in hyb],
                marker='*', markersize=14, linewidth=2.5,
                label='Hybrid (Images + Instructions)', color='teal', alpha=0.9)
        _annotate_line(ax, hyb, 'tokens', offset_y=5)
    ax.set_xlabel('Tokens per Inference', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} [Hybrid] - Accuracy vs Token Consumption',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, 'accuracy_vs_tokens')

    # ── 2. Cost-Performance ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    if zs:
        ax.scatter([zs[0]['cost']], [zs[0]['accuracy']], s=200, c='red',
                   marker='o', label='Zero-Shot (baseline)', zorder=5,
                   edgecolors='black', linewidths=1.5)
    if fsr:
        ax.plot([d['cost'] for d in fsr], [d['accuracy'] for d in fsr],
                marker='D', markersize=10, linewidth=2.5,
                label='Few-Shot ICL + Rerank', color='darkorange', alpha=0.85)
        _annotate_line(ax, fsr, 'cost', offset_y=5)
    if a1r:
        ax.plot([d['cost'] for d in a1r], [d['accuracy'] for d in a1r],
                marker='P', markersize=10, linewidth=2.5,
                label='Approach 1 + Rerank', color='purple', alpha=0.85)
        _annotate_line(ax, a1r, 'cost', offset_y=-15)
    if hyb:
        ax.plot([d['cost'] for d in hyb], [d['accuracy'] for d in hyb],
                marker='*', markersize=14, linewidth=2.5,
                label='Hybrid (Images + Instructions)', color='teal', alpha=0.9)
        _annotate_line(ax, hyb, 'cost', offset_y=5)
    ax.set_xlabel('Cost per Inference (USD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} [Hybrid] - Cost-Performance Tradeoff',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, 'cost_performance')

    # ── 3. K-value Impact ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    zero_acc = zs[0]['accuracy'] if zs else None

    def _with_zero(data_list):
        if zero_acc is not None:
            return [0] + [d['k'] for d in data_list], [zero_acc] + [d['accuracy'] for d in data_list]
        return [d['k'] for d in data_list], [d['accuracy'] for d in data_list]

    if fsr:
        k_vals, accs = _with_zero(fsr)
        ax.plot(k_vals, accs, marker='D', markersize=10, linewidth=2.5,
                label='Few-Shot ICL + Rerank', color='darkorange')
    if a1r:
        k_vals, accs = _with_zero(a1r)
        ax.plot(k_vals, accs, marker='P', markersize=10, linewidth=2.5,
                label='Approach 1 + Rerank', color='purple')
    if hyb:
        k_vals, accs = _with_zero(hyb)
        ax.plot(k_vals, accs, marker='*', markersize=14, linewidth=2.5,
                label='Hybrid (Images + Instructions)', color='teal')
    ax.set_xlabel('K (Image examples; Instructions = 5K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('VQA Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name.upper()} [Hybrid] - Impact of K on Accuracy',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, 'k_value_impact')


def create_summary_statistics(results, model_name, output_dir, rerank=False, hybrid=False):
    """
    Create summary statistics JSON and print comparison table.
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

    suffix = '_hybrid' if hybrid else ('_rerank' if rerank else '')
    summary_path = os.path.join(output_dir, f'{model_name}{suffix}_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved summary statistics to {summary_path}")

    label = '[HYBRID] ' if hybrid else ('[RERANK] ' if rerank else '')
    print("\n" + "="*90)
    print(f"{model_name.upper()} {label}- PERFORMANCE & COST COMPARISON")
    print("="*90)
    print(f"{'Method':<30} {'Accuracy':<12} {'Total Tokens':<15} {'Cost (USD)':<12} {'Tokens/Accuracy':<15}")
    print("-"*90)

    for method in methods:
        acc = results[method]['evaluation']['total_accuracy']
        tokens = results[method]['token_cost']['total_tokens']
        cost = results[method]['token_cost']['estimated_cost_usd']
        efficiency = tokens / acc if acc > 0 else float('inf')
        print(f"{method:<30} {acc:<12.4f} {tokens:<15,} ${cost:<11.2f} {efficiency:<15,.0f}")

    if hybrid:
        # Hybrid-specific: compare all three methods at K=3
        print("\n" + "="*90)
        print("HYBRID vs RERANK COMPARISON (K=3)")
        print("="*90)
        for key, label_str in [('few_shot_rerank_k3', 'Few-Shot + Rerank K=3'),
                                ('approach1_rerank_k3', 'Approach 1 + Rerank K=3'),
                                ('hybrid_k3', 'Hybrid K=3')]:
            if key in results:
                acc = results[key]['evaluation']['total_accuracy']
                tok = results[key]['token_cost']['total_tokens']
                print(f"{label_str:<30}  accuracy={acc:.4f}  tokens={tok:,}")
    elif rerank:
        # Rerank-specific: compare the two rerank methods at K=3
        if 'few_shot_rerank_k3' in results and 'approach1_rerank_k3' in results:
            fsr_acc = results['few_shot_rerank_k3']['evaluation']['total_accuracy']
            a1r_acc = results['approach1_rerank_k3']['evaluation']['total_accuracy']
            fsr_tok = results['few_shot_rerank_k3']['token_cost']['total_tokens']
            a1r_tok = results['approach1_rerank_k3']['token_cost']['total_tokens']
            savings_pct = ((fsr_tok - a1r_tok) / fsr_tok) * 100 if fsr_tok > 0 else 0
            print("\n" + "="*90)
            print("FEW-SHOT RERANK vs APPROACH 1 RERANK (K=3)")
            print("="*90)
            print(f"Few-Shot + Rerank K=3:    accuracy={fsr_acc:.4f}  tokens={fsr_tok:,}")
            print(f"Approach 1 + Rerank K=3:  accuracy={a1r_acc:.4f}  tokens={a1r_tok:,}")
            print(f"Token savings (A1R vs FSR): {fsr_tok - a1r_tok:,} ({savings_pct:.1f}%)")
    else:
        # Baseline: legacy token savings block
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

def analyze_model(model_name, config, rerank=False, hybrid=False):
    """
    Main analysis function for a model.
    Pass rerank=True for rerank-only plots.
    Pass hybrid=True for hybrid comparison plots (hybrid + rerank methods + zero-shot).
    """
    mode_label = 'HYBRID' if hybrid else ('RERANK' if rerank else 'BASELINE')
    print("="*70)
    print(f"ANALYZING RESULTS - MODEL: {model_name.upper()} [{mode_label}]")
    print("="*70)

    results = collect_results(model_name, config, rerank=rerank, hybrid=hybrid)

    if not results:
        print(f"\nNo evaluation results found for {model_name}")
        print("Make sure you have run eval.py first to generate metrics files")
        return

    output_dir = os.path.join(config['output']['figures_dir'], model_name)

    print("\nCreating visualizations...")
    if hybrid:
        create_hybrid_visualizations(results, model_name, output_dir)
    elif rerank:
        create_rerank_visualizations(results, model_name, output_dir)
    else:
        create_visualizations(results, model_name, output_dir)

    print("\nGenerating summary statistics...")
    create_summary_statistics(results, model_name, output_dir, rerank=rerank, hybrid=hybrid)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Analyze VQA experiment results')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gemma', 'qwen', 'phi', 'llama', 'llava'],
                       help='Model to analyze')
    parser.add_argument('--rerank', action='store_true',
                       help='Analyse rerank results only (few_shot_rerank / approach1_rerank). '
                            'Zero-shot is included as baseline reference.')
    parser.add_argument('--hybrid', action='store_true',
                       help='Analyse hybrid results alongside rerank methods. '
                            'Plots hybrid / few_shot_rerank / approach1_rerank + zero-shot baseline.')
    args = parser.parse_args()

    config = load_config()
    analyze_model(args.model, config, rerank=args.rerank, hybrid=args.hybrid)

if __name__ == "__main__":
    main()
