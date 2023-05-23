# Intent Role Label Annotations
This directory contains annotations used for training an intent role labeling (IRL) model.

Annotation span offsets are provided in the `irl-annotations.jsonl` file.
To generate annotations with corresponding text from the 
[SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) dataset,
use the `anchor_spans.py` script:
```sh
python anchor_spans.py --sgd-dir path/to/dstc8-schema-guided-dialogue --offsets-file path/to/irl-annotations.jsonl
```

