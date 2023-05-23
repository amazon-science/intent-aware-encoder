python preprocess/create_pretrain_dataset.py \
    --irl_model_path ../models/irl_model/irl-model-sgd-08-16-2022.tar.gz \
    --top1_dir ../data/sources/topv1 \
    --top2_dir ../data/sources/topv2 \
    --dstc11t2_dir ../data/sources/dstc11t2 \
    --sgd_dir ../data/sources/sgd \
    --multiwoz_dir ../data/sources/multiwoz2.2 \
    --output_dir ../data/pretraining