# Intent Role Label Annotations
This directory contains annotations used for training an intent role labeling (IRL) model.

* Annotation span offsets are provided in the `irl-annotations.jsonl` file
* The train/dev/test split used in the paper is provided in `splits.json`

## Prerequisites
* Python ≥ 3.9

## Data Processing
To generate annotations with corresponding text from the
[SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
dataset, tokenize and create a train/dev/test split, run the script [run_process_data.sh](run_process_data.sh):
```bash
cd data
bash run_process_data.sh
```

This script does the following:
* runs `anchor_spans.py` to get corresponding text for `irl-annotations.jsonl`.
* runs `convert_spans.py` to tokenize, apply IOB labeling, and prepare train/dev/test split

## Model Training
To train an intent role labeling model on the resulting annotations, run the [run_train_model.sh](run_train_model.sh):
```bash
cd data
bash run_train_model.sh
```