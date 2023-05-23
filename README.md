# 🥧 Pre-Training Intent-Aware Encoders

This repository is used for **P**re-training **I**ntent-Aware **E**ncoders (PIE) and evaluating on four intent classification datasets 
([BANKING77](https://arxiv.org/abs/2003.04807), [HWU64](https://arxiv.org/abs/1903.05566),
[Liu54](https://arxiv.org/abs/1903.05566), and [CLINC150](https://aclanthology.org/D19-1131/)).

## Environment setup

### Option 1: Docker
```
image_name=pie
code_path=/path/to/intent-aware-encoder
docker build  -t $image_name .
nvidia-docker run -it -v ${code_path}:/code $image_name
cd code
```

### Option 2: Conda
```
conda create -n pie python=3.8
conda activate pie
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

## Pre-training
See the readme in the `pretraining` directory.
```
cd pretraining
```

## Fine-tuning and Evaluation
See the readme in the `downstream` directory.
```
cd downstream
```

## Acknowledgement
Parts of the code are modified from [mirror-bert](https://github.com/cambridgeltl/mirror-bert), [IDML](https://github.com/microsoft/KC/tree/main/papers/IDML), and [ProtAugment](https://github.com/tdopierre/ProtAugment). We appreciate the authors for open sourcing their projects.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
