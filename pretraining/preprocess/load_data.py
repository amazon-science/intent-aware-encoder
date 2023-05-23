from utils import camel_terms, filter_text, process_text, sample_min_num
from glob import glob
import csv
import json
from tqdm import tqdm

def load_top(min_num = 1000, single_sent=False, top1_data_dir="", top2_data_dir=""):
    print("load top")
    intent_correction = {
        'gettimer': 'get timer'
    }

    paths = glob(f'{top1_data_dir}/*.tsv')
    paths2 = glob(f'{top2_data_dir}/*.tsv')
    paths += paths2
    intent2utterances = {}
    utterances = {} # to detect duplication
    for path in paths:
        with open(path) as f:
            reader = csv.reader(f, delimiter='\t', quotechar='\"')
            next(reader, None)
            for row in reader:
                _, utterance, tag = row
                utterance = utterance.strip()
                if not filter_text(utterance, single_sent):
                    continue
                utterance = process_text(utterance)
            
                if tag.count('IN:') != 1:
                    continue

                for t in tag.split():
                    if 'IN:' in t:
                        intent = t[t.index("IN:")+3:].replace("_"," ").lower()
                    break
                if 'unsupported' in intent or 'unintelligible' in intent:
                    continue
                if intent in intent_correction:
                    intent = intent_correction[intent]
                if intent not in intent2utterances:
                    intent2utterances[intent] = []
                
                # deduplicate
                if utterance in utterances:
                    continue
                utterances[utterance] = ""

                intent2utterances[intent].append(utterance)
    return sample_min_num(intent2utterances, min_num)

def load_dstc11t2(min_num = 1000, single_sent=False, data_dir=""):
    print("load dstc11t2")
    intent_correction = {
        'getbankstatement': 'get bank statement',
        'getloaninfo': 'get loan info',
        'netincome': 'net income'
    }

    intent2utterances = {}
    utterances = {} # to detect duplication
    path = f'{data_dir}/dstc11t2-intent-data.tsv'
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t', quotechar='\"')
        next(reader, None)
        for row in reader:
            source, intent, utterance = row
            utterance = utterance.strip()
            if not filter_text(utterance, single_sent):
                continue
            utterance = process_text(utterance)

            intent = ' '.join(camel_terms(intent)).lower()
            if 'faq' in intent:
                continue

            if intent in intent_correction:
                intent = intent_correction[intent]

            if intent not in intent2utterances:
                intent2utterances[intent] = []

            # deduplicate
            if utterance in utterances:
                continue
            utterances[utterance] = ""

            intent2utterances[intent].append(utterance)

    return sample_min_num(intent2utterances, min_num)


def load_sgd(min_num=100, single_sent=False, data_dir=""):
    intent2utterances = {}
    intent_names = {}
    utterances = {} # to detect duplication

    for datatype in ['train', 'dev', 'test']:
        paths = glob(f'{data_dir}/{datatype}/dialogues_*.json')
        for path in tqdm(paths):
            with open(path) as f:
                data = json.load(f)
                for dialogue in data:
                    turns = dialogue['turns']
                    for turn in turns[:1]: # constraint: first turn 
                        utterance = turn['utterance'].strip()
                        if not filter_text(utterance, single_sent):
                            continue
                        utterance = process_text(utterance)
            
                        frames = turn['frames']
                        for frame in frames:
                            actions = frame['actions']
                            for action in actions:
                                slot = action['slot']
                                canonical_values = action['canonical_values']
                                if slot == 'intent':
                                    for canonical_value in canonical_values:
                                        if canonical_value in intent_names:
                                            intent = intent_names[canonical_value]
                                        else:
                                            intent = ' '.join(camel_terms(canonical_value)).lower()
                                            intent_names[canonical_value] = intent

                                        if intent not in intent2utterances:
                                            intent2utterances[intent] = []

                                        # deduplicate
                                        if utterance in utterances:
                                            continue
                                        utterances[utterance] = ""

                                        intent2utterances[intent].append(utterance) 
    return sample_min_num(intent2utterances, min_num)                             

def load_multiwoz(min_num=100, data_dir=""):
    intent2utterances = {}
    utterances = {} # to detect duplication

    for datatype in ['train', 'dev', 'test']:
        paths = glob(f'{data_dir}/{datatype}/dialogues_*.json')
        for path in tqdm(paths):
            with open(path) as f:
                data = json.load(f)
                
                for dialogue in data:
                    turns = dialogue['turns']
                    for turn in turns[:1]: # constraint: first turn 
                        utterance = turn['utterance'].strip()
                        if not filter_text(utterance):
                            continue
                        utterance = process_text(utterance)
            
                        frames = turn['frames']
                        for frame in frames:
                            if 'state' not in frame:
                                continue
                            if not frame['slots']:
                                continue
                            intent = frame['state']['active_intent']
                            if intent == 'NONE':
                                continue

                            intent = intent.replace("_", " ")

                            # deduplicate
                            if utterance in utterances:
                                continue
                            utterances[utterance] = ""
                            
                            if intent not in intent2utterances:
                                intent2utterances[intent] = []
                            intent2utterances[intent].append(utterance)           
    return sample_min_num(intent2utterances, min_num)                             
