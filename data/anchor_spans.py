"""
Script to anchor labeled spans in dialogues from the SGD dataset
(https://github.com/google-research-datasets/dstc8-schema-guided-dialogue).

Usage:
python anchor_spans.py --sgd-dir path/to/dstc8-schema-guided-dialogue --offsets-file path/to/annotations.jsonl

(Will write anchored annotations at path/to/annotations.anchored.jsonl)
"""
import argparse
import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, List

parser = argparse.ArgumentParser()
parser.add_argument('--sgd-dir',
                    help='Path to SGD repository root directory from'
                         ' https://github.com/google-research-datasets/dstc8-schema-guided-dialogue', required=True)
parser.add_argument('--offsets-file',
                    help='Path to file defining labeled spans with offsets in SGD conversations', required=True)


@dataclass
class LabeledSpan:
    label: str
    start: int
    exclusive_end: int
    text: str = None

    @staticmethod
    def from_dict(json_dict: Dict):
        return LabeledSpan(
            label=json_dict['label'],
            start=json_dict['start'],
            exclusive_end=json_dict['exclusive_end'],
            text=json_dict.get('text'),
        )


@dataclass
class LabeledUtterance:
    spans: List[LabeledSpan]
    dialogue: str
    turn: int
    text: str = None

    @staticmethod
    def from_dict(json_dict: Dict):
        return LabeledUtterance(
            [LabeledSpan.from_dict(span) for span in json_dict['spans']],
            json_dict.get('dialogue'),
            json_dict.get('turn'),
            json_dict.get('text')
        )

    @staticmethod
    def read_lines(path: Path):
        with path.open() as lines:
            lines = [line.strip() for line in lines if line.strip()]
        result = []
        for line in lines:
            result.append(LabeledUtterance.from_dict(json.loads(line)))
        return result


def write_utterances(path: Path, utterances: List[LabeledUtterance]):
    with path.open(mode='w') as out:
        for utterance in utterances:
            out.write(json.dumps(asdict(utterance)) + '\n')


def anchor_spans_from_sgd(sgd_dir: Path, offsets_file: Path) -> List[LabeledUtterance]:
    """
    :param sgd_dir: Path to SGD repository root directory
    :param offsets_file: Path to file defining labeled spans with offsets in SGD conversations
    :return: utterances with spans anchored in text from SGD
    """
    result = []
    uid_to_utterance = {
        (utt.dialogue, utt.turn): utt for utt in LabeledUtterance.read_lines(offsets_file)
    }
    for dialogues_path in sorted(sgd_dir.glob('train/dialogues*.json')):
        dialogues = json.loads(dialogues_path.read_text(encoding='utf-8'))
        for dialogue in dialogues:
            for i, turn in enumerate(dialogue['turns']):
                text = turn['utterance']
                uid = (dialogue["dialogue_id"], i)
                if uid not in uid_to_utterance:
                    continue
                utterance = uid_to_utterance[uid]
                anchored_utterance = replace(
                    utterance,
                    spans=[replace(span, text=text[span.start:span.exclusive_end]) for span in utterance.spans],
                    text=text
                )
                result.append(anchored_utterance)
    return result


if __name__ == '__main__':
    args = parser.parse_args()
    anchored = anchor_spans_from_sgd(Path(args.sgd_dir), Path(args.offsets_file))
    write_utterances(Path(args.offsets_file).with_suffix('.anchored.jsonl'), anchored)

