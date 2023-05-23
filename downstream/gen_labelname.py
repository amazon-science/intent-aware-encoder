# Original Copyright (c) Microsoft Corporation. Licensed under the MIT license.
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import os, json
import argparse

def generate_labels(input_fn, output_fn):
    all_labels = []
    with open(input_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            label = json.loads(line)["label"]
            if label not in all_labels:
                all_labels.append(label)
    with open(output_fn, mode="w", encoding="utf-8") as fp:
        fp.writelines([x + "\n" for x in all_labels])
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)

    args = parser.parse_args()

    for dataset in ['BANKING77', 'HWU64', 'Liu', 'OOS']:
        input_path = os.path.join(args.data_dir, dataset, 'full.jsonl')
        output_path = os.path.join(args.data_dir, dataset, 'labels.txt')
        generate_labels(input_path, output_path)