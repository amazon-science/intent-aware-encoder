DATA_DIR=../data/downstream
OUTPUT_DIR=../output/eval
MODEL_PATH=../models/iae_model
dataset=OOS
n_classes=50 # N-way (50-way)
n_support=1 # K-shot (1-shot)
OUTPUT_NAME=iae_${dataset}_${n_classes}_${n_support}
python run_eval.py \
    --test-path ${DATA_DIR}/${dataset}/few_shot \
    --output-path ${OUTPUT_DIR}/${dataset}/${OUTPUT_NAME} \
    --encoder sentbert \
    --model-name-or-path ${MODEL_PATH} \
    --load_ckpt False \
    --n-test-episodes 600 --n-support ${n_support} --n-classes ${n_classes} --n-query 5 --metric euclidean --pooling avg \
    --label_fn ${DATA_DIR}/${dataset}/labels.txt