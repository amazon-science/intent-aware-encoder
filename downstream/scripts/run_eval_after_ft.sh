DATA_DIR=../data/downstream
OUTPUT_DIR=../output/eval
MODEL_PATH=../models/iae_model
CKPT_DIR=../output/finetuned_iae_model
dataset=OOS
n_classes=50 # N-way (50-way) (e.g. 5)
n_support=1  # K-shot (1-shot)
ckpt_n_classes=5 # assume that ckpt was created in 5-way
CKPT_NAME=protaugment_${dataset}_${ckpt_n_classes}_${n_support}
OUTPUT_NAME=iae_protaugment_${dataset}_${n_classes}_${n_support}
python run_eval.py \
    --test-path ${DATA_DIR}/${dataset}/few_shot \
    --output-path ${OUTPUT_DIR}/${dataset}/${OUTPUT_NAME} \
    --encoder sentbert \
    --model-name-or-path ${MODEL_PATH} \
    --load_ckpt True \
    --ckpt_path ${CKPT_DIR}/protaugment/${CKPT_NAME} \
    --n-test-episodes 6 --n-support ${n_support} --n-classes ${n_classes} --n-query 5 --metric euclidean --pooling avg \
    --label_fn ${DATA_DIR}/${dataset}/labels.txt