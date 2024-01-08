# Pre-training IAE Model

This directory is about how to preprocess pre-training datasets, how to pre-train an encoder using the pre-training datasets, and how to validate the trained model.

## Prerequisites
To pre-process data for pre-training, you will first need to prepare an intent role labeling model.
Instructions for this can be found in [README.md](/data/README.md).

## Pre-training Dataset Creation

To create pre-training dataset, run the following script.

```
bash ./scripts/create_dataset.sh
```

## Pretraining

To pre-train an encoder, run the following script.
```
bash ./scripts/run_pretrain.sh
```

## Evaluation

To evaluate the pre-trained encoder, run the following script.
```
bash ./scripts/run_eval.sh
```