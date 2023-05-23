from sentence_transformers import SentenceTransformer, util
import argparse
import random
random.seed(0)
import logging
import torch
from utils import init_logging

LOGGER = logging.getLogger()

def load_data(test_path, draft=False):
    utterances = []
    labels = []

    with open(test_path) as f:
        intent2count = {}
        for row in f:
            utterance, label, _ = row.strip().split("||")

            if label not in intent2count:
                intent2count[label] = 0 
            intent2count[label] += 1

            if draft and intent2count[label] > 3:
                    continue

            utterances.append(utterance)
            labels.append(label)

    return utterances, labels

def find_top1_intent_idxs(model, utterances, intent_embeddings, distance_metric='cosine'):
    utterance_embeddings = model.encode(utterances)
    # calc distance/score
    if distance_metric == 'cosine':
        consine_scores = util.cos_sim(utterance_embeddings, intent_embeddings)
        top1_intent_idxs = torch.argmax(consine_scores, dim=1)
    else: # euclidean
        raise NotImplementedError
    
    return top1_intent_idxs

def run_eval(test_path, model_name_or_path="", distance_metric='cosine', draft=False, verbose=False):
    utterances, labels = load_data(test_path, draft)
    unique_intents = list(set(labels))
    LOGGER.info(f"the number of unique intents)={len(unique_intents)}")

    model = SentenceTransformer(model_name_or_path, device='cuda')
    
    with torch.no_grad():
        intent_embeddings = model.encode(unique_intents)
        hit = 0
        total = 0
        batch_size = 100
        for i in range(0, len(utterances), batch_size):
            b_utterances = utterances[i:i+batch_size]
            b_labels = labels[i:i+batch_size]

            top1_intent_idxs = find_top1_intent_idxs(model, b_utterances, intent_embeddings, distance_metric)
            
            for label, top1_intent_idx in zip(b_labels, top1_intent_idxs):
                top1_intent = unique_intents[top1_intent_idx]
                if label == top1_intent:
                    hit+=1
                total +=1
    
    acc = hit/total*100
    if verbose:
        LOGGER.info(f"acc={acc}")
    return acc

def main(args):
    run_eval(
        test_path=args.test_path,
        model_name_or_path=args.model_name_or_path,
        distance_metric=args.distance_metric,
        draft=args.draft,
        verbose=args.verbose
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--distance_metric', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    init_logging(LOGGER)
    print(args)
    main(args)
