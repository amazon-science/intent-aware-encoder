# Original Copyright (c) 2021 Cambridge Language Technology Lab. Licensed under the MIT License.
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import numpy as np
import random
from torch.utils.data import Dataset
import logging

LOGGER = logging.getLogger(__name__)

def erase_and_mask(s, tokenizer, mask_len=5):
    """
    Randomly replace a span in input s with "[MASK]".
    """
    if len(s) <= mask_len: return s
    if len(s) < 30: return s # if too short, no span masking
    ind = np.random.randint(len(s)-mask_len)
    left, right = s.split(s[ind:ind+mask_len], 1)
    return " ".join([left, tokenizer.mask_token, right]) 

# 2022-09-09: Amazon modification
class ContrastiveLearningDataset(Dataset):
    def __init__(self, path, tokenizer, random_span_mask=0, pairwise=False, triplewise=False, masking_strat='opt1', draft=False): 
        with open(path, 'r') as f:
            lines = f.readlines()
        self.sent_pairs = []
        self.pairwise = pairwise
        self.intent2label = {}
        self.utterance2label = {}

        for line in lines:
            line = line.rstrip("\n")
            try:
                utterance, intent, irl = line.split("||")
            except:
                continue

            if intent not in self.intent2label:
                self.intent2label[intent] = len(self.intent2label)
            self.utterance2label[utterance] = self.intent2label[intent]
            self.sent_pairs.append((utterance, intent, irl))

        if draft:
            self.sent_pairs = self.sent_pairs[:1000]
            
        self.tokenizer = tokenizer
        self.random_span_mask = random_span_mask
        self.masking_strat = masking_strat

        self.intent2utterances = {}
        for utterance, intent, irl in self.sent_pairs:
            if intent not in self.intent2utterances:
                self.intent2utterances[intent] = []
            
            self.intent2utterances[intent].append(utterance)

    def __getitem__(self, idx):
        # batch_x1: input utterance
        # batch_x2: gold intent
        # batch_x3: gold utterance
        # batch_x4: pseudo intent

        utterance = self.sent_pairs[idx][0]
        gold_intent = self.sent_pairs[idx][1]
        pseudo_intent = self.sent_pairs[idx][2]

        # gold_utterances: utternaces with the same gold intent as the input utterance
        gold_utterances = [u for u in self.intent2utterances[gold_intent] if u != utterance]

        if not gold_utterances:
            gold_utterance = random.sample(gold_utterances,k = 1)[0]
        else: # random masking
            gold_utterance = erase_and_mask(utterance, self.tokenizer, mask_len=int(self.random_span_mask))

        if idx < 5:
            print(f"{idx},input utterance=",utterance)
            print(f"{idx},gold intent=",gold_intent)
            print(f"{idx},gold utterance=",gold_utterance)
            print(f"{idx},pseudo intent=",pseudo_intent)

        return utterance, gold_intent, gold_utterance, pseudo_intent
        
    def __len__(self):
        assert (len(self.sent_pairs) !=0)
        return len(self.sent_pairs)
# End of Amazon modification