# Downstream

This directory is about how to evaluate the pre-trained model on each of four intent classification datasets (BANKING77, HWU64, Liu, OOS), and how to fine-tune the model using ProtoNet or ProtAugment.

## Download Datasets & Get Label Names
For convenience, you can download the intent dataset splits from [this repo](https://github.com/tdopierre/ProtAugment/tree/main/data) and put the downloaded data into `data` folder.
Before evaluating or fine-tuning model, label name files should be generated to use label names as support example.

```
DATA_DIR=../data/downstream
python gen_labelname.py --data-dir ${DATA_DIR}
```

## Eval the Model Before Fine-tuning
To evaluate the pre-trained model on the intent classification dataset, run the following script.
```
bash ./scripts/run_eval_before_ft.sh
```

## ProtoNet
To fine-tune the pre-trained model using ProtoNet, run the following script.
```
bash ./scripts/run_protonet.sh
```

## ProtAugment
To fine-tune the pre-trained model using ProtAugment, run the following script.
```  
bash ./scripts/run_protaugment.sh
```

## Eval the Model After Fine-tuning

To evaluate the fine-tuned model, run the following script.
```
bash ./scripts/run_eval_after_ft.sh
```