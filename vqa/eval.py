#!/usr/bin/env python3

import os
import json
import re
import argparse

def normalize_answer(answer):
    """
    Normalize answer following VQA evaluation protocol.
    """
    answer = answer.lower().strip()
    
    # Remove periods except if it occurs as decimal
    answer = re.sub(r'(?<!\d)\.(?!\d)', '', answer)
    
    # Convert number words to digits
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in number_words.items():
        answer = re.sub(r'\b' + word + r'\b', digit, answer)
    
    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
    
    # Add apostrophe to contractions
    contractions = {
        'dont': "don't", 'doesnt': "doesn't", 'didnt': "didn't",
        'isnt': "isn't", 'arent': "aren't", 'wasnt': "wasn't",
        'werent': "weren't", 'wont': "won't", 'wouldnt': "wouldn't",
        'couldnt': "couldn't", 'shouldnt': "shouldn't", 'cant': "can't",
        'hasnt': "hasn't", 'havent': "haven't", 'hadnt': "hadn't"
    }
    for wrong, correct in contractions.items():
        answer = re.sub(r'\b' + wrong + r'\b', correct, answer)
    
    # Replace punctuation with space (except apostrophe and colon)
    answer = re.sub(r"[^\w\s':]+", ' ', answer)
    
    # Remove commas between digits
    answer = re.sub(r'(\d),(\d)', r'\1\2', answer)
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer

def compute_vqa_accuracy(predicted_answer, ground_truth_answers):
    """
    Compute VQA accuracy: min(# humans who gave that answer / 3, 1)
    """
    predicted_normalized = normalize_answer(predicted_answer)
    
    # Normalize ground truth answers
    gt_normalized = [normalize_answer(ans) for ans in ground_truth_answers]
    
    # Count how many humans gave this answer
    match_count = gt_normalized.count(predicted_normalized)
    
    # VQA accuracy formula
    accuracy = min(match_count / 3.0, 1.0)
    
    return accuracy

def load_predictions(filepath):
    """Load prediction file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prediction file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate_predictions(predictions_path, output_path):
    """
    Evaluate predictions and compute VQA accuracy metrics.
    """
    print("="*70)
    print("VQA EVALUATION")
    print("="*70)
    print(f"\nPredictions file: {predictions_path}")
    
    # Load predictions
    predictions = load_predictions(predictions_path)
    print(f"Total predictions: {len(predictions)}")
    
    # Initialize results
    results = {
        'predictions_file': predictions_path,
        'total_questions': len(predictions),
        'successful_predictions': 0,
        'error_count': 0,
        'total_accuracy': 0.0,
        'by_question_type': {}
    }
    
    # Track per-question-type metrics
    qtype_metrics = {}
    
    total_score = 0.0
    successful_count = 0
    error_count = 0
    
    # Evaluate each prediction
    for pred in predictions:
        # Skip errors
        if pred.get('predicted_answer') == 'ERROR' or 'error' in pred:
            error_count += 1
            continue
        
        # Compute VQA accuracy
        accuracy = compute_vqa_accuracy(
            pred['predicted_answer'],
            pred['ground_truth_answers']
        )
        
        total_score += accuracy
        successful_count += 1
        
        # Per question type
        qtype = pred.get('question_type', 'unknown')
        if qtype not in qtype_metrics:
            qtype_metrics[qtype] = {
                'count': 0,
                'total_accuracy': 0.0
            }
        
        qtype_metrics[qtype]['count'] += 1
        qtype_metrics[qtype]['total_accuracy'] += accuracy
    
    # Compute overall accuracy
    if successful_count > 0:
        results['total_accuracy'] = total_score / successful_count
    
    results['successful_predictions'] = successful_count
    results['error_count'] = error_count
    
    # Compute per-type accuracy
    for qtype, metrics in qtype_metrics.items():
        if metrics['count'] > 0:
            avg_accuracy = metrics['total_accuracy'] / metrics['count']
            results['by_question_type'][qtype] = {
                'count': metrics['count'],
                'accuracy': avg_accuracy
            }
    
    # Save results
    print(f"\nSaving results to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: round(x, 4))
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Total questions: {results['total_questions']}")
    print(f"Successful predictions: {results['successful_predictions']}")
    print(f"Errors: {results['error_count']}")
    print(f"\nOverall VQA Accuracy: {results['total_accuracy']:.4f}")
    
    print("\n" + "-"*70)
    print("Per-Question-Type Accuracy")
    print("-"*70)
    print(f"{'Question Type':<20} {'Accuracy':<12}")
    print("-"*70)
    
    for qtype in sorted(results['by_question_type'].keys()):
        metrics = results['by_question_type'][qtype]
        print(f"{qtype:<20} {metrics['accuracy']:<12.4f}")
    
    print("\n" + "="*70)
    print(f"Results saved to: {output_path}")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Evaluate VQA predictions')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save evaluation results JSON')
    args = parser.parse_args()
    
    evaluate_predictions(args.predictions, args.output)

if __name__ == "__main__":
    main()