#!/usr/bin/env python3

import os
import json
import re
import argparse


def normalize_label(label):
    """
    Canonical form for comparing Oxford-IIIT Pets labels:
    lowercase, underscores/hyphens -> spaces, collapsed whitespace.
    So ``american_bulldog`` matches ``American Bulldog`` and ``american bulldog``.
    """
    if label is None:
        return ""
    label = str(label).lower().strip()
    label = label.replace("\u2019", "'")
    label = label.replace("_", " ").replace("-", " ")
    label = re.sub(r"\s+", " ", label)
    return label.strip()



def load_predictions(filepath):
    """
    Load prediction file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prediction file not found: {filepath}")

    with open(filepath, "r") as f:
        return json.load(f)


def compute_mean_per_class_accuracy(class_metrics):
    """
    Mean per-class accuracy (macro average). Standard for Oxford-IIIT Pets
    and other fine-grained classification with class imbalance.
    """
    class_accuracies = []

    for cls, metrics in class_metrics.items():
        if metrics["count"] > 0:
            acc = metrics["correct"] / metrics["count"]
            class_accuracies.append(acc)

    if not class_accuracies:
        return 0.0

    return sum(class_accuracies) / len(class_accuracies)


def evaluate_predictions(predictions_path, output_path):

    print("=" * 70)
    print("OXFORD-IIIT PETS EVALUATION")
    print("=" * 70)

    print(f"\nPredictions file: {predictions_path}")

    predictions = load_predictions(predictions_path)

    print(f"Total predictions: {len(predictions)}")

    total = len(predictions)
    correct = 0
    error_count = 0

    class_metrics = {}

    for pred in predictions:

        pl = pred.get("predicted_label")
        pc = pred.get("predicted_class")
        if (
            pl == "ERROR"
            or pc == "ERROR"
            or pred.get("error") is not None
        ):
            error_count += 1
            continue

        gt = normalize_label(pred.get("ground_truth", ""))

        raw_pred = pl if pl not in (None, "") else pc
        pred_label = normalize_label(raw_pred)

        is_correct = gt == pred_label

        if is_correct:
            correct += 1

        if gt not in class_metrics:
            class_metrics[gt] = {
                "count": 0,
                "correct": 0
            }

        class_metrics[gt]["count"] += 1

        if is_correct:
            class_metrics[gt]["correct"] += 1

    # Top-1 overall accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Mean per-class accuracy (primary metric on Oxford-IIIT Pets)
    mean_per_class_acc = compute_mean_per_class_accuracy(class_metrics)

    results = {
        "predictions_file": predictions_path,
        "total_images": total,
        "correct_predictions": correct,
        "error_count": error_count,
        "top1_accuracy": accuracy,
        "mean_per_class_accuracy": mean_per_class_acc,   # primary metric
        "num_classes_evaluated": len(class_metrics),
        "per_class_accuracy": {}
    }

    for cls, metrics in class_metrics.items():

        acc = metrics["correct"] / metrics["count"]

        results["per_class_accuracy"][cls] = {
            "count": metrics["count"],
            "correct": metrics["correct"],
            "accuracy": acc
        }

    # Sort per-class results by accuracy (ascending) for easy inspection
    results["per_class_accuracy"] = dict(
        sorted(
            results["per_class_accuracy"].items(),
            key=lambda x: x[1]["accuracy"]
        )
    )

    print(f"\nSaving results to: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: round(x, 4))

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"Total images:             {results['total_images']}")
    print(f"Correct predictions:      {results['correct_predictions']}")
    print(f"Errors:                   {results['error_count']}")
    print(f"Classes evaluated:        {results['num_classes_evaluated']}")
    print(f"\nTop-1 Accuracy:           {results['top1_accuracy']:.4f}")
    print(f"Mean Per-Class Accuracy:  {results['mean_per_class_accuracy']:.4f}  <-- primary metric")

    # Print bottom-5 and top-5 classes for quick diagnostics
    sorted_classes = list(results["per_class_accuracy"].items())

    print("\n--- Lowest 5 Classes ---")
    for cls, m in sorted_classes[:5]:
        print(f"  {cls:<40} acc={m['accuracy']:.2f}  ({m['correct']}/{m['count']})")

    print("\n--- Highest 5 Classes ---")
    for cls, m in sorted_classes[-5:]:
        print(f"  {cls:<40} acc={m['accuracy']:.2f}  ({m['correct']}/{m['count']})")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_path}")
    print("=" * 70)


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate Oxford-IIIT Pets predictions (top-1 and mean per-class accuracy)"
    )

    parser.add_argument(
        "--p",
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )

    parser.add_argument(
        "--o",
        type=str,
        required=True,
        help="Path to save evaluation results JSON"
    )

    args = parser.parse_args()

    evaluate_predictions(args.p, args.o)


if __name__ == "__main__":
    main()