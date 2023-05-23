DATA_DIR=../data/downstream
OUTPUT_DIR=../output/finetuned_iae_model
MODEL_PATH=../models/iae_model
dataset=OOS
n_classes=5 # N-way
n_support=1 # K-shot
MODEL_NAME=iae_${dataset}_${n_classes}_${n_support}
python run_protonet.py \
    --train-path ${DATA_DIR}/${dataset}/few_shot \
    --valid-path ${DATA_DIR}/${dataset}/few_shot \
    --test-path ${DATA_DIR}/${dataset}/few_shot \
    --model-name-or-path ${MODEL_PATH} \
    --n-support ${n_support} \
    --n-query 5 \
    --n-classes ${n_classes} \
    --n-augment 0 \
    --lr 1e-6 \
    --encoder sentbert \
    --evaluate-every 100 \
    --n-test-episodes 600 \
    --max-iter 10000 \
    --early-stop 20 \
    --log-every 10 \
    --seed 42 \
    --metric euclidean \
    --output-path ${OUTPUT_DIR}/protonet/${MODEL_NAME} \
    --label_fn ${DATA_DIR}/${dataset}/labels.txt