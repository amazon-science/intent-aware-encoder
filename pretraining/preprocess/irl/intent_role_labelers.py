"""
I want to find and reserve a room.
I want to reserve a flight and a hotel.
"""
from collections import defaultdict
from dataclasses import replace
from typing import List, Dict, Tuple

from allennlp.common import Registrable
from spacy.tokens import Span, Doc

from irl.data import LabeledUtterance, Frame
from irl.irl_tagger import load_irl_model


class IntentRoleLabeler(Registrable):
    def label(self, utterance: str) -> LabeledUtterance:
        """
        Predict intent role labels for a single input utterance.
        :param utterance: input utterance
        :return: predicted role labels
        """
        return self.label_batch([utterance])[0]

    def label_batch(self, utterances: List[str]) -> List[LabeledUtterance]:
        """
        Predict intent role labels for a batch of input utterances.
        :param utterances: batch of input utterances
        :return: predicted role labels for each input utterance
        """
        raise NotImplementedError


@IntentRoleLabeler.register('tagger_based_intent_role_labeler')
class TaggerBasedIntentRoleLabeler(IntentRoleLabeler):
    def __init__(
        self,
        model_path: str,
        spacy_model: str = 'en_core_web_md',
        cuda_device: int = 0
    ) -> None:
        super().__init__()
        self._predictor = load_irl_model(model_path, cuda_device=cuda_device)
        from spacy import load
        self._nlp = load(spacy_model)

    @staticmethod
    def _convert_to_labeled_utterance(
        predictions: List[LabeledUtterance],
        doc: Doc
    ) -> LabeledUtterance:
        frames = []
        for prediction, sent in zip(predictions, doc.sents):
            for prop in prediction.frames:
                spans = [
                    replace(
                        span,
                        start=span.start + sent.start_char,
                        exclusive_end=span.exclusive_end + sent.start_char
                    ) for span in prop.spans]
                if not spans:
                    continue
                frames.append(Frame(spans))
        return LabeledUtterance(doc.text, frames)

    @staticmethod
    def convert_to_labeled_utterances(
        predictions: List[LabeledUtterance],
        inputs: List[Span],
        sent_idx_to_utterance_idx: Dict[int, int],
        parsed: List[Doc],
    ):
        predictions_by_utterance = defaultdict(list)
        for i, (prediction, parse) in enumerate(zip(predictions, inputs)):
            utterance_idx = sent_idx_to_utterance_idx[i]
            predictions_by_utterance[utterance_idx].append(prediction)

        result = [
            TaggerBasedIntentRoleLabeler._convert_to_labeled_utterance(predictions, parsed[idx])
            for idx, predictions in predictions_by_utterance.items()
        ]
        return result

    def label_batch(self, utterances: List[str]) -> List[LabeledUtterance]:
        # split into sentences and parse
        sent_idx_to_utterance_idx, inputs, parsed = parse_utterances(utterances, self._nlp)

        sentences = [sent.text for sent in inputs]
        predictions = []
        for utterance, spans in zip(sentences, self._predictor.tag_batch(sentences)):
            predictions.append(
                LabeledUtterance(
                    utterance,
                    [Frame(spans)]
                )
            )

        # regroup predictions by utterance
        result = self.convert_to_labeled_utterances(
            predictions,
            inputs,
            sent_idx_to_utterance_idx,
            parsed,
        )
        return result


def parse_utterances(
    utterances: List[str], nlp, batch_size: int = 64
) -> Tuple[Dict[int, int], List[Span], List[Doc]]:
    sent_idx_to_utterance_idx = {}
    inputs = []
    parsed = list(nlp.pipe(utterances, batch_size=batch_size, n_process=1, disable=["ner"]))
    for i, parse in enumerate(parsed):
        for _ in parse.sents:
            sent_idx_to_utterance_idx[len(sent_idx_to_utterance_idx)] = i
        inputs.extend(parse.sents)
    return sent_idx_to_utterance_idx, inputs, parsed
