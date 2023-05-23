import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass(frozen=True, eq=True)
class LabeledSpan:
    label: str
    start: int
    exclusive_end: int
    text: str

    @staticmethod
    def from_dict(json_dict: Dict):
        return LabeledSpan(
            json_dict['label'],
            json_dict['start'],
            json_dict['exclusive_end'],
            json_dict['text']
        )


@dataclass(frozen=True, eq=True)
class Frame:
    spans: List[LabeledSpan]

    @staticmethod
    def from_dict(json_dict: Dict):
        return Frame([LabeledSpan.from_dict(span) for span in json_dict['spans']])


@dataclass
class LabeledUtterance:
    text: str
    frames: List[Frame]

    @staticmethod
    def from_dict(json_dict: Dict):
        return LabeledUtterance(json_dict['text'], [Frame.from_dict(frame) for frame in json_dict['frames']])

    @staticmethod
    def read_lines(path: Path):
        with path.open() as lines:
            lines = [line.strip() for line in lines if line.strip()]
        result = []
        for line in lines:
            result.append(LabeledUtterance.from_dict(json.loads(line)))
        return result