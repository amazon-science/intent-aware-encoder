"""
Convert JSONL anchored/labeled spans to training format (IOB) and splits.

Usage:
```
python3 -m venv irl-data
source irl-data/bin/activate
pip install -r requirements-process-data.txt
python convert_spans.py irl-annotations.anchored.jsonl splits.json
```
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from spacy.tokens import Doc
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('spans', type=Path)
parser.add_argument('splits', type=Path)


@dataclass(frozen=True)
class LabeledSpan:
    label: str
    start: int
    exclusive_end: int
    text: Optional[str] = None

    @classmethod
    def from_dict(cls, json_dict: Dict) -> 'LabeledSpan':
        return LabeledSpan(**json_dict)


@dataclass(frozen=True)
class LabeledUtterance:
    text: str
    spans: List[LabeledSpan]
    uid: str = ''

    @classmethod
    def from_dict(cls, json_dict: Dict) -> 'LabeledUtterance':
        return LabeledUtterance(
            text=json_dict['text'],
            spans=[LabeledSpan.from_dict(span) for span in json_dict['spans']],
            uid=f"{json_dict['dialogue']}.{json_dict['turn']}"
        )

    @staticmethod
    def read_lines(path: Path) -> List['LabeledUtterance']:
        result = []
        with path.open() as lines:
            for line in lines:
                result.append(LabeledUtterance.from_dict(json.loads(line)))
        return result


def _init_parser(name="en_core_web_md"):
    import spacy
    from spacy.attrs import ORTH
    from spacy import Language

    nlp = spacy.load(name)

    @Language.component("set_custom_boundaries")
    def set_custom_boundaries(_doc):
        for _token in _doc[:-1]:
            if _token.text == "\n":
                _doc[_token.i + 1].is_sent_start = True
            else:
                _doc[_token.i + 1].is_sent_start = False
        return _doc

    #  special cases for tokenizer to avoid sentence splitting issues
    special_cases = [
        ("ride?How", [{ORTH: "ride"}, {ORTH: "?"}, {ORTH: "How"}]),
        ("travel?I", [{ORTH: "travel"}, {ORTH: "?"}, {ORTH: "I"}]),
        ("cab?it", [{ORTH: "cab"}, {ORTH: "?"}, {ORTH: "it"}]),
        ("events?2", [{ORTH: "events"}, {ORTH: "?"}, {ORTH: "2"}]),
        ("Great!Buy", [{ORTH: "Great"}, {ORTH: "!"}, {ORTH: "Buy"}]),
        ("there/", [{ORTH: "there"}, {ORTH: "/"}]),
        ("calendar/", [{ORTH: "calendar"}, {ORTH: "/"}]),
        ("then-", [{ORTH: "then"}, {ORTH: "-"}]),
        ("Stovall'S.", [{ORTH: "Stovall"}, {ORTH: "'S"}, {ORTH: "."}]),
    ]
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        special_cases.append((f'{letter}..', [{ORTH: f'{letter}.'}, {ORTH: '.'}]))

    for key, val in special_cases:
        nlp.tokenizer.add_special_case(key, val)
    nlp.add_pipe("set_custom_boundaries", first=True)
    return nlp


def convert_to_labels_and_tokens(
    spans: List[LabeledSpan],
    parse: Doc,
) -> Tuple[List[str], List[str]]:
    ann_map = {}  # resolved spans by word index
    for span in spans:
        character_span = parse.char_span(span.start, span.exclusive_end)
        for word in character_span:
            ann_map[word.i] = span

    tokens = []
    labels = []
    prev_ann_start = -1
    for tok in parse:
        label, ann_start = 'O', -1
        if tok.i in ann_map:
            span = ann_map[tok.i]
            ann_start = span.start
            label = f'{"B" if ann_start != prev_ann_start else "I"}-{span.label}'
        labels.append(label)
        tokens.append(tok.text.strip())
        prev_ann_start = ann_start
    return labels, tokens


def _convert_utterance(labeled_utterance: LabeledUtterance, parse: Doc) -> Optional[Dict]:
    spans = labeled_utterance.spans
    spans = sorted(spans, key=lambda x: (x.start, x.exclusive_end))
    labels, tokens = convert_to_labels_and_tokens(spans, parse)
    return dict(tokens=tokens, labels=labels, uid=labeled_utterance.uid)


def _tokenize_and_apply_iob_labels(utterances: List[LabeledUtterance]) -> Dict[str, Dict]:
    # tokenize
    nlp = _init_parser()
    parses = tqdm(nlp.pipe([utt.text for utt in utterances], batch_size=64), total=len(utterances))
    # convert to IOB
    tokenized_utterances = {}
    for parse, utterance in zip(parses, utterances):
        converted = _convert_utterance(utterance, parse)
        tokenized_utterances[converted['uid']] = converted
    return tokenized_utterances


def main(in_pth: Path, splits_pth: Path, out_pth: Path):
    # read anchored utterances
    utterances = LabeledUtterance.read_lines(in_pth)

    # tokenize utterances and convert spans to IOB labels (e.g. B-Query, I-Query, O)
    tokenized_utterances = _tokenize_and_apply_iob_labels(utterances)

    # split into train/dev/test using splits from paper
    split_to_uids = json.loads(splits_pth.read_text('utf-8'))
    splits = {}
    for split, uids in split_to_uids.items():
        print(f'{split}: {len(uids)}')
        splits[split] = [tokenized_utterances[uid] for uid in uids]

    # write splits in JSONL format, with a line per sentence
    out_pth.mkdir(exist_ok=True, parents=True)
    for split, utterances in splits.items():
        with (out_pth / f'{split}.json').open(mode='w') as out:
            for conv in utterances:
                out.write(json.dumps(conv) + '\n')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.spans, args.splits, args.spans.parent)
