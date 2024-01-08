#!/usr/bin/env bash

set -euxo pipefail
cd "$(dirname "$0")" || exit

if [ ! -d "irl-data" ]; then
  python3 -m venv irl-data
fi

# Install dependencies
source irl-data/bin/activate
pip install -r requirements-process-data.txt

# Add text from SGD to IRL annotations
if [ ! -d "dstc8-schema-guided-dialogue" ]; then
  git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git
fi

python3 anchor_spans.py --sgd-dir dstc8-schema-guided-dialogue --offsets-file irl-annotations.jsonl

# Tokenize text and convert spans to IOB labels corresponding to each token, split into train/dev/tet
python3 convert_spans.py irl-annotations.anchored.jsonl splits.json
