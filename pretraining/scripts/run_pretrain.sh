output_name=iae_model
python run_pretrain.py \
    --model_name_or_path sentence-transformers/paraphrase-mpnet-base-v2 \
    --train_path ../data/pretraining/train.txt \
    --val_path ../data/pretraining/val.txt \
    --output_dir ../models/${output_name} \
    --epoch 1 \
    --train_batch_size 50 \
    --learning_rate 1e-6 \
    --max_length 50 \
    --infoNCE_tau 0.05 \
    --dropout_rate 0.1 \
    --drophead_rate 0.0 \
    --random_span_mask 5 \
    --random_seed 42 \
    --agg_mode mean_std \
    --amp \
    --parallel \
    --use_cuda \
    --pseudo_weight 2