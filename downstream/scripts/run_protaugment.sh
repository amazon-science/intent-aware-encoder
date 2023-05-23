DATA_DIR=../data/downstream
OUTPUT_DIR=../output/finetuned_iae_model
MODEL_PATH=../models/iae_model
dataset=OOS
n_classes=5 # N-way
n_support=1 # K-shot
MODEL_NAME=protaugment_${dataset}_${n_classes}_${n_support}
python run_protaugment.py \
    --data-path ${DATA_DIR}/${dataset}/full.jsonl \
    --train-labels-path ${DATA_DIR}/${dataset}/few_shot \
    --valid-labels-path ${DATA_DIR}/${dataset}/few_shot \
    --test-labels-path ${DATA_DIR}/${dataset}/few_shot \
    --unlabeled-path ${DATA_DIR}/${dataset}/raw.txt \
    --n-support ${n_support} \
    --n-query 5 \
    --n-classes ${n_classes} \
    --evaluate-every 100 \
    --n-test-episodes 600 \
    --max-iter 10 \
    --early-stop 20 \
    --log-every 10 \
    --seed 42 \
    --n-unlabeled 5 \
    --augmentation-data-path ${DATA_DIR}/${dataset}/paraphrases/DBS-unigram-flat-1.0/paraphrases.jsonl \
    --metric euclidean \
    --lr 1e-6 \
    --encoder sentbert \
    --supervised-loss-share-power 1 \
    --model-name-or-path ${MODEL_PATH} \
    --output-path ${OUTPUT_DIR}/protaugment/${MODEL_NAME} \
    --label_fn ${DATA_DIR}/${dataset}/labels.txt