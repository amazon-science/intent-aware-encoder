"""
Predictors applied to dialogs/dialog turns, providing facades over AllenNLP models for inference.

Essentially a lot of boilerplate code to load AllenNLP predictor wrapper.
"""
import logging
from typing import Dict, Iterable, List, Tuple, Any

import torch
from allennlp.common import JsonDict, Registrable
from allennlp.common.util import lazy_groups_of
from allennlp.data import DatasetReader, Instance, Vocabulary, TextFieldTensors, Token, Tokenizer, TokenIndexer
from allennlp.data.fields import SequenceLabelField, TextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import Model, SimpleTagger, load_archive
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.conditional_random_field import allowed_transitions, ConditionalRandomField
from allennlp.predictors import Predictor

from irl.data import LabeledSpan
from irl.utils import labels_to_spans

logger = logging.getLogger(__name__)


class TaggerConstants:
    TAGS = 'tags'
    TOKENS = 'tokens'
    WORDS = 'words'
    TEXT = 'text'
    BEGIN_ = 'B-'
    IN_ = 'I-'
    OUT = 'O'


@Predictor.register('tagger_predictor')
class TaggerPredictor(Predictor):
    """
    Sequence tagger of `DialogTurn`s, used as a facade over AllenNLP models. Provides conversion
    logic of IOB tags to spans.
    """

    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 frozen: bool = True,
                 batch_size: int = 128) -> None:
        """
        Initialize a predictor used for inference on dialog turns.
        Args:
            model: AllenNLP model
            dataset_reader: dataset reader used for feature extraction
            frozen: whether model is frozen or not
            batch_size: batch size for inference
        """
        super().__init__(model, dataset_reader, frozen)
        self._batch_size = batch_size

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        dialog_turn = json_dict[TaggerConstants.TEXT]
        return self._dataset_reader.text_to_instance(dialog_turn, [])

    def predict_instance(self, instance: Instance) -> JsonDict:
        prediction = super().predict_instance(instance)
        return self._update_prediction(instance, prediction)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        results = []
        for model_input_json, prediction in zip(instances,
                                                super().predict_batch_instance(instances)):
            results.append(self._update_prediction(model_input_json, prediction))
        return results

    def tag_batch(self, dialog_turns: Iterable[str]) -> List[List[LabeledSpan]]:
        instances = self._batch_json_to_instances(
            [{TaggerConstants.TEXT: dialog_turn} for dialog_turn in dialog_turns]
        )

        results = []
        for batch_instance in lazy_groups_of(instances, self._batch_size):
            for model_input_json, prediction in zip(batch_instance,
                                                    self.predict_batch_instance(
                                                        batch_instance)):
                results.append(prediction)

        results = [self.prediction_to_spans(pred) for pred in results]
        return results

    @staticmethod
    def _update_prediction(instance: Instance, prediction: JsonDict) -> JsonDict:
        tags = prediction[TaggerConstants.TAGS]
        # noinspection PyUnresolvedReferences
        tokens = instance[TaggerConstants.TOKENS].tokens
        # noinspection PyUnresolvedReferences
        return {
            TaggerConstants.TOKENS: tokens,
            TaggerConstants.TAGS: tags[:len(tokens)],
            TaggerConstants.TEXT: instance.fields[TaggerConstants.TEXT].metadata
        }

    @staticmethod
    def prediction_to_spans(prediction: JsonDict) -> List[LabeledSpan]:
        """
        Convert tagger predictions to `LabeledSpan` which consists of start and end
        offsets in original text.
        Args:
            prediction: predict output

        Returns: list of labeled spans

        """
        tokens = prediction[TaggerConstants.TOKENS]
        tags = prediction[TaggerConstants.TAGS]
        labeled_spans = []
        for name, start, end in labels_to_spans(tags):
            start_idx = tokens[start].idx
            end_idx = tokens[end - 1].idx_end
            text = prediction[TaggerConstants.TEXT][start_idx:end_idx]
            labeled_spans.append(LabeledSpan(name, start_idx, end_idx, text=text))
        return labeled_spans


@Model.register('turn_tagger')
class TurnTagger(SimpleTagger):
    """
    Override `SimpleTagger` mostly to add **kwargs to the forward method, allowing us
    to include the original UID for inputs in output without causing an assertion error.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 label_encoding='BIO',
                 viterbi_decoding=False,
                 **kwargs) -> None:
        super().__init__(vocab,
                         text_field_embedder,
                         encoder,
                         calculate_span_f1=True,
                         label_encoding=label_encoding,
                         **kwargs)
        if viterbi_decoding:
            logger.info(f'Initializing tagger with {label_encoding}'
                        f'-constrained Viterbi decoding enabled')
        self.viterbi_decoding = viterbi_decoding
        constraints = allowed_transitions(
            constraint_type=label_encoding,
            labels=self.vocab.get_index_to_token_vocabulary(self.label_namespace)
        )
        self._crf = ConditionalRandomField(
            num_tags=self.vocab.get_vocab_size(self.label_namespace),
            constraints=constraints,
        )
        self._crf.transitions.requires_grad_(False).fill_(0)

    def forward(
        self,
        tokens: TextFieldTensors,
        tags: torch.LongTensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        output_dict = super().forward(tokens,
                                      tags,
                                      kwargs.get(TaggerConstants.WORDS))
        return {**output_dict, **kwargs}

    def _decode(
        self, output_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform IOB-constrained Viterbi decoding.
        """
        viterbi_tags = self._crf.viterbi_tags(
            logits=output_dict['logits'],
        )
        all_tags = []
        for indices, score in viterbi_tags:
            tags = [
                self.vocab.get_token_from_index(x, namespace=self.label_namespace)
                for x in indices
            ]
            all_tags.append(tags)
        output_dict[TaggerConstants.TAGS] = all_tags
        return output_dict

    def make_output_human_readable(
        self,
        output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        readable_output = super().make_output_human_readable(output_dict)

        return {
            TaggerConstants.TAGS: readable_output[TaggerConstants.TAGS]
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return super().get_metrics(reset)


class TaggerPreprocessor(Registrable):
    default_implementation = 'transformer_tagger_preprocessor'

    def text_to_tokens_and_tags(
        self,
        text: str,
        spans: List[LabeledSpan]
    ) -> Tuple[List[Token], List[str]]:
        raise NotImplementedError


@TaggerPreprocessor.register('transformer_tagger_preprocessor')
class TransformerTaggerPreprocessor(TaggerPreprocessor):
    """
    Tagger pre-processor that extracts labels and corresponding tokens for Transformer-based models.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def text_to_tokens_and_tags(
        self,
        text: str,
        spans: List[LabeledSpan]
    ) -> Tuple[List[Token], List[str]]:
        text = text.lower()
        utterance_tokens = self.tokenizer.tokenize(text=text)

        annotation_map = {}
        for span in spans:
            for i in range(span.start, span.exclusive_end):
                annotation_map[i] = span
        prev_annotation_start = -1
        labels = []
        for tok in utterance_tokens:
            label, annotation_start = TaggerConstants.OUT, -1
            if tok.idx in annotation_map:
                span = annotation_map[tok.idx]
                annotation_start = span.start
                tag = (TaggerConstants.BEGIN_ if annotation_start != prev_annotation_start
                       else TaggerConstants.IN_)
                label = f'{tag}{span.label}'
            # truncate tokens above the max length
            prev_annotation_start = annotation_start
            labels.append(label)

        return utterance_tokens, labels


@DatasetReader.register('tagger_dataset_reader')
class TaggerDatasetReader(DatasetReader):
    """
    Dataset reader that extracts token-level labels from each turn in a `Dialog`.
    """

    def __init__(self,
                 tagger_preprocessor: TransformerTaggerPreprocessor,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._tagger_preprocessor = tagger_preprocessor
        self._token_indexers = token_indexers or {TaggerConstants.TOKENS: SingleIdTokenIndexer()}

    def text_to_instance(self, turn: str, labels: List[LabeledSpan]):
        tokens, labels = self._tagger_preprocessor.text_to_tokens_and_tags(turn, labels)
        tokens_field = TextField(tokens=tokens, token_indexers=self._token_indexers)
        fields = {
            TaggerConstants.WORDS: MetadataField(
                {
                    TaggerConstants.WORDS: [x.text for x in tokens]
                }
            ),
            TaggerConstants.TOKENS: tokens_field,
            TaggerConstants.TAGS: SequenceLabelField(labels, tokens_field),
            TaggerConstants.TEXT: MetadataField(turn)
        }

        return Instance(fields)

    def _read(self, file_path) -> Iterable[Instance]:
        with open(file_path) as lines:
            for line in lines:
                yield self.text_to_instance(line, [])


def load_irl_model(model_path: str, override: Dict = None, cuda_device: int = 0) -> TaggerPredictor:
    """Load IRL model as TaggerPredictor"""
    if not override:
        override = {}
    archive = load_archive(model_path, cuda_device=cuda_device, overrides=override)
    return TaggerPredictor.from_archive(archive, predictor_name='tagger_predictor')
