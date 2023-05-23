output_name=iae_model
python run_eval.py \
    --test_path ../data/pretraining/val.txt \
    --model_name_or_path ../output/${output_name} \
    --distance_metric cosine \
    --verbose