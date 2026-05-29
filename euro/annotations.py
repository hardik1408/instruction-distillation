#!/usr/bin/env python3

import os
import json
import csv
import argparse
from tqdm import tqdm


def load_label_map(label_map_path):

    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    # Case 1: {"0": "AnnualCrop"}
    try:
        return {int(k): v for k, v in label_map.items()}
    except ValueError:
        pass

    # Case 2: {"AnnualCrop": 0}
    reversed_map = {v: k for k, v in label_map.items()}

    return reversed_map


def load_csv_annotations(csv_path, image_root):

    annotations = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        image_id = 0

        for row in tqdm(reader, desc=f"Processing {os.path.basename(csv_path)}"):

            filename = row.get("Filename")
            label = row.get("Label")
            class_name = row.get("ClassName")

            # Skip bad rows
            if filename is None or label is None or class_name is None:
                continue

            if filename.strip() == "" or label.strip() == "":
                continue

            class_id = int(label)

            image_path = os.path.abspath(
                os.path.join(image_root, filename)
            )

            image_name = os.path.basename(filename)

            annotations.append({
                "image_id": image_id,
                "image_name": image_name,
                "image_path": image_path,
                "class_id": class_id,
                "class_name": class_name
            })

            image_id += 1

    return annotations


def save_annotations(annotations, label_list, dataset_name, output_path):

    data = {
        "dataset": dataset_name,
        "num_images": len(annotations),
        "num_classes": len(label_list),
        "label_list": label_list,
        "data": annotations
    }

    output_dir = os.path.dirname(output_path)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(annotations)} annotations → {output_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_root",
        default="EuroSAT/",
        help="Root directory where images are stored"
    )

    parser.add_argument(
        "--train_csv",
        default="EuroSAT/train.csv"
    )

    parser.add_argument(
        "--test_csv",
        default="EuroSAT/test.csv"
    )

    parser.add_argument(
        "--label_map",
        default="EuroSAT/label_map.json"
    )

    parser.add_argument(
        "--output_dir",
        default="data"
    )

    args = parser.parse_args()

    label_map = load_label_map(args.label_map)

    label_list = [
        label_map[i] for i in sorted(label_map.keys())
    ]

    # TRAIN annotations
    train_annotations = load_csv_annotations(
        args.train_csv,
        args.image_root
    )

    train_output = os.path.join(
        args.output_dir,
        "eurosat_train_annotations.json"
    )

    save_annotations(
        train_annotations,
        label_list,
        "eurosat_train",
        train_output
    )

    # TEST annotations
    test_annotations = load_csv_annotations(
        args.test_csv,
        args.image_root
    )

    test_output = os.path.join(
        args.output_dir,
        "eurosat_test_annotations.json"
    )

    save_annotations(
        test_annotations,
        label_list,
        "eurosat_test",
        test_output
    )


if __name__ == "__main__":
    main()