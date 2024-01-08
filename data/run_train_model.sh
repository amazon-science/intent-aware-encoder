#!/usr/bin/env bash

MODEL_NAME='irl-model-sgd-08-16-2022.tar.gz'

set -euxo pipefail
cd "$(dirname "$0")" || exit

if [ ! -d "irl-model-train" ]; then
  python3 -m venv irl-model-train
fi

# Install dependencies
source irl-model-train/bin/activate
pip install -r requirements-train-model.txt

# download training script
if [ ! -f "run_ner.py" ]; then
  wget https://raw.githubusercontent.com/huggingface/transformers/v4.18.0/examples/pytorch/token-classification/run_ner.py
fi

# i/o
train_file='train.json'
validation_file='dev.json'
test_file='test.json'
output_dir='results'
label_column_name='labels'

# parameters
base_model='roberta-base'
max_seq_length=256
train_batch_size=16
eval_batch_size=32
lr=2e-5
weight_decay=0.01
lr_scheduler_type='linear'
num_train_epochs=8
warmup_ratio=0.06
seed=1

# run training
WANDB_DISABLED="true" python run_ner.py \
--output_dir $output_dir \
--train_file $train_file \
--validation_file $validation_file \
--test_file $test_file \
--model_name_or_path $base_model \
--max_seq_length $max_seq_length \
--evaluation_strategy epoch \
--per_device_train_batch_size $train_batch_size \
--per_device_eval_batch_size $eval_batch_size \
--learning_rate $lr \
--weight_decay $weight_decay \
--lr_scheduler_type $lr_scheduler_type \
--num_train_epochs $num_train_epochs \
--warmup_ratio $warmup_ratio \
--seed $seed \
--label_column_name $label_column_name \
--do_train \
--do_eval \
--do_predict

# package model
PYTHONPATH="$(pwd)/../pretraining/preprocess" python convert_to_allennlp.py results
mkdir -p ../models/irl_model/
mv archive/model.tar.gz ../models/irl_model/$MODEL_NAME
