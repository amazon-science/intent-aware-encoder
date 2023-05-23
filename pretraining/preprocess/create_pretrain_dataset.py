import os

from allennlp.common import Params
import random
from utils import construct_irl_text
from irl.intent_role_labelers import IntentRoleLabeler
import argparse
from dataclasses import asdict
from load_data import load_top, load_dstc11t2, load_sgd, load_multiwoz

def process_irl(utterances, intents, role_labeler):
    irls = role_labeler.label_batch(utterances)

    output = []
    new_utterances = {}
    new_intents = {}
    new_irl_texts = {}
    for utterance, intent, irl in zip(utterances, intents, irls):
        irl = asdict(irl) # irl to dict
        if utterance.lower() in new_utterances: # deduplicate
            continue

        irl_text, has_irl = construct_irl_text(irl)
        if has_irl:
            output.append('||'.join([utterance, intent, irl_text]))
            new_utterances[utterance.lower()] = ""
            new_intents[intent.lower()] = ""
            new_irl_texts[irl_text.lower()] = ""
            
    print("len(new_utterances):", len(new_utterances))
    print("len(new_intents):", len(new_intents))
    print("len(new_irl_texts):", len(new_irl_texts))

    return output

def save_data(output, path):
    # stat
    utterances = {}
    intents ={}
    irl_texts ={}
    for o in output:
        utterance, intent, irl = o.split("||")
        utterances[utterance.lower()] = ""
        intents[intent.lower()] = ""
        irl_texts[irl.lower()] = ""

    print("total len(utterances):", len(utterances))
    print("total len(intents):", len(intents))
    print("total len(irl_texts):", len(irl_texts))

    random.shuffle(output)
    with open(path, 'w') as f:
        for line in output:
            f.write(line + "\n")
    print(path)

def main(args):
    # load irl model
    role_labeler = IntentRoleLabeler.from_params(Params({
        'type': 'tagger_based_intent_role_labeler',
        'model_path': args.irl_model_path,
        'cuda_device': 0
    }))

    # preprocess train dataset
    train_data = []
    utterances, intents = load_top(1000, top1_data_dir=args.top1_dir, top2_data_dir=args.top2_dir)
    train_data += process_irl(utterances, intents, role_labeler)
    utterances, intents = load_dstc11t2(1000, data_dir=args.dstc11t2_dir)
    train_data += process_irl(utterances, intents, role_labeler)
    utterances, intents = load_sgd(100, single_sent=True, data_dir=args.sgd_dir)
    train_data += process_irl(utterances, intents, role_labeler)
    train_data = list(set(train_data)) # deduplicate

    # preprocess validation dataset
    val_data = []
    utterances, intents = load_multiwoz(100, data_dir=args.multiwoz_dir)
    val_data += process_irl(utterances, intents, role_labeler)

    # save
    os.makedirs(args.output_dir, exist_ok = True)
    train_path = os.path.join(args.output_dir, 'train.txt')
    val_path = os.path.join(args.output_dir, 'val.txt')
    save_data(train_data, train_path)
    save_data(val_data, val_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--irl_model_path', type=str, required=True)
    parser.add_argument('--top1_dir', type=str, required=True)
    parser.add_argument('--top2_dir', type=str, required=True)
    parser.add_argument('--dstc11t2_dir', type=str, required=True)
    parser.add_argument('--sgd_dir', type=str, required=True)
    parser.add_argument('--multiwoz_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()

    print(args)
    main(args)

