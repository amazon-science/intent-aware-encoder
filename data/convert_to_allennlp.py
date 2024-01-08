"""
This script converts from a transformers IRL model to an AllenNLP predictor (sequence tagger).
"""
import argparse
import json
from pathlib import Path

import torch
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import archive_model
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.initializers import PretrainedModelInitializer

from irl.intent_role_labelers import IntentRoleLabeler
from irl.irl_tagger import TurnTagger

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=Path)


def _vocabulary(model_name: Path):
    from allennlp.common import cached_transformers
    tokenizer = cached_transformers.get_tokenizer(str(model_name))
    vocab = Vocabulary(non_padded_namespaces=['tokenizer', 'labels'], oov_token='<unk>')
    # add tokenizer vocabulary
    vocab.add_transformer_vocab(tokenizer, 'tokenizer')
    # add label vocabulary
    id2label = json.loads((model_name / 'config.json').read_text(encoding='utf-8'))['id2label']
    for k, v in id2label.items():
        vocab.add_token_to_namespace(v, "labels")
    return vocab


def _model(model_name: Path, vocab: Vocabulary):
    text_field_embedder = BasicTextFieldEmbedder(token_embedders={
        "bert": PretrainedTransformerMismatchedEmbedder(str(model_name)),
    })
    model = TurnTagger(
        vocab=vocab,
        text_field_embedder=text_field_embedder,
        encoder=PassThroughEncoder(768),
        initializer=InitializerApplicator([(
            # map to parameter names expected by AllenNLP
            'tag_projection_layer.*', PretrainedModelInitializer(
                weights_file_path=str(model_name / 'pytorch_model.bin'),
                parameter_name_overrides={
                    'tag_projection_layer._module.bias': 'classifier.bias',
                    'tag_projection_layer._module.weight': 'classifier.weight',
                }))
        ])
    )
    return model


def _config():
    # hard-coded configuration for AllenNLP inference
    return {
        "dataset_reader": {
            "type": "tagger_dataset_reader",
            "tagger_preprocessor": {
                "tokenizer": {
                    "type": "spacy",
                    "pos_tags": True
                }
            },
            "token_indexers": {
                "bert": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": "roberta-base",
                    "namespace": "tokens"
                }
            }
        },
        "model": {
            "type": "turn_tagger",
            "encoder": {
                "type": "pass_through",
                "input_dim": 768
            },
            "text_field_embedder": {
                "token_embedders": {
                    "bert": {
                        "type": "pretrained_transformer_mismatched",
                        "model_name": "roberta-base"
                    }
                }
            }
        }}


def main():
    args = parser.parse_args()
    model_path = args.model_path
    archive_path = model_path.parent / 'archive'
    archive_path.mkdir(parents=True, exist_ok=True)
    # prepare the vocabulary
    vocab = _vocabulary(model_path)
    # prepare the model
    model = _model(model_path, vocab)
    # add weights
    torch.save(model.state_dict(), archive_path / 'weights.th')
    # meta file indicating allenNLP version number
    (archive_path / 'meta.json').write_text(json.dumps({'version': '2.9.3'}), encoding='utf-8')
    # add vocabulary
    vocab.save_to_files(str(archive_path / 'vocabulary'))
    # add config
    (archive_path / 'config.json').write_text(json.dumps(_config()), encoding='utf-8')
    # write archive to model.tar.gz
    archive_model(str(archive_path), 'weights.th')

    # prepare the dataset reader / tokenizer
    predictor = IntentRoleLabeler.from_params(Params({
        'type': 'tagger_based_intent_role_labeler',
        'model_path': archive_path / 'model.tar.gz',
        'cuda_device': -1,
    }))

    # sanity check
    text_sample = "I'm looking to purchase movie tickets."
    print(predictor.label_batch([text_sample]))


if __name__ == '__main__':
    main()
