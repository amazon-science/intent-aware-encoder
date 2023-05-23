from typing import List, Tuple, Iterable


class BioesConfig:
    BEGIN = "B"
    BEGIN_ = "B-"
    END = "E"
    END_ = "E-"
    SINGLE = "S"
    SINGLE_ = "S-"
    IN = "I"
    IN_ = "I-"
    OUT = "O"
    DELIM = "-"


def labels_to_spans(labeling: Iterable[str]) -> List[Tuple[str, int, int]]:
    """
    Given an IOB/BESIO chunking produce a list of labeled spans--triples of (label, start index,
    end index exclusive).
    >>> labels_to_spans(['O', 'B-PER', 'I-PER', 'O', 'B-ORG'])
    [('PER', 1, 3), ('ORG', 4, 5)]

    :param labeling: list of IOB/BESIO labels
    :return: list of spans
    """

    def _start_of_chunk(curr):
        curr_tag, _ = _get_val_and_tag(curr)
        return curr_tag in {BioesConfig.SINGLE, BioesConfig.BEGIN}

    def _end_of_chunk(curr):
        curr_tag, _ = _get_val_and_tag(curr)
        return curr_tag in {BioesConfig.END, BioesConfig.SINGLE}

    besio = chunk(labeling, besio=True)

    result = []
    curr_label, start = None, None
    for index, label in enumerate(besio):
        if _start_of_chunk(label):
            if curr_label:
                result.append((curr_label, start, index))
            curr_label, start = _get_val_and_tag(label)[1], index
        if _end_of_chunk(label):
            result.append((curr_label, start, index + 1))
            curr_label = None
    if curr_label:
        result.append((curr_label, start, len(besio)))

    return result


def chunk(labeling: Iterable[str], besio=False) -> List[str]:
    """
    Convert an IO/BIO/BESIO-formatted sequence of labels to BIO, BESIO, or CoNLL-2005 formatted.
    :param labeling: original labels
    :param besio: (optional) convert to BESIO format, `False` by default
    :return: converted labels
    """
    result = []
    prev_type = None
    curr = []
    for label in labeling:
        if label == BioesConfig.OUT:
            state, chunk_type = BioesConfig.OUT, ''
        else:
            split_index = label.index(BioesConfig.DELIM)
            state, chunk_type = label[:split_index], label[split_index + 1:]
        if state == BioesConfig.IN and chunk_type != prev_type:  # new chunk of different type
            state = BioesConfig.BEGIN
        if state in [BioesConfig.BEGIN, BioesConfig.OUT] and curr:  # end of chunk
            result += _to_besio(curr) if besio else curr
            curr = []
        if state == BioesConfig.OUT:
            result.append(state)
        else:
            curr.append(state + BioesConfig.DELIM + chunk_type)
        prev_type = chunk_type
    if curr:
        result += _to_besio(curr) if besio else curr
    return result


def _to_besio(iob_labeling):
    if len(iob_labeling) == 1:
        return [BioesConfig.SINGLE + iob_labeling[0][1:]]
    return iob_labeling[:-1] + [BioesConfig.END + iob_labeling[-1][1:]]


def _get_val_and_tag(label):
    if not label:
        return '', ''
    if label == BioesConfig.OUT:
        return label, ''
    return label.split(BioesConfig.DELIM, 1)
