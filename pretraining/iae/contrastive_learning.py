# Original Copyright (c) 2021 Cambridge Language Technology Lab. Licensed under the MIT License.
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from pytorch_metric_learning import losses
from transformers import (
    AdamW,
)
LOGGER = logging.getLogger(__name__)


class ContrastiveLearningPairwise(nn.Module):
    def __init__(self, encoder, learning_rate, weight_decay, use_cuda=True, \
     agg_mode="cls", infoNCE_tau="0.04"):

        LOGGER.info(f"ContrastiveLearningPairwise! learning_rate={learning_rate} weight_decay={weight_decay} " \
            f"agg_mode={agg_mode} infoNCE_tau={infoNCE_tau}")
        super(ContrastiveLearningPairwise, self).__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.agg_mode = agg_mode
        self.optimizer = AdamW([{'params': self.encoder.parameters()},], 
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.infoNCE_tau = infoNCE_tau # sentence & phrase: 0.04, word: 0.2  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        self.loss = losses.NTXentLoss(temperature=self.infoNCE_tau)
        
        print ("loss:", self.loss)
    
    @autocast() 
    def forward(self, query_toks1, query_toks2, labels=None):
        outputs1 = self.encoder(**query_toks1, return_dict=True, output_hidden_states=False)
        outputs2 = self.encoder(**query_toks2, return_dict=True, output_hidden_states=False)
        last_hidden_state1 = outputs1.last_hidden_state
        last_hidden_state2 = outputs2.last_hidden_state

        if self.agg_mode=="cls":
            query_embed1 = last_hidden_state1[:,0]  
            query_embed2 = last_hidden_state2[:,0]  
        elif self.agg_mode == "mean": # include padded tokens
            query_embed1 = last_hidden_state1.mean(1)  
            query_embed2 = last_hidden_state2.mean(1)
        elif self.agg_mode == "mean_std":
            query_embed1 = (last_hidden_state1 * query_toks1['attention_mask'].unsqueeze(-1)).sum(1) / query_toks1['attention_mask'].sum(-1).unsqueeze(-1)
            query_embed2 = (last_hidden_state2 * query_toks2['attention_mask'].unsqueeze(-1)).sum(1) / query_toks2['attention_mask'].sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()

        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        
        if labels is None:
            labels = torch.arange(query_embed1.size(0))
        labels = torch.cat([labels, labels], dim=0)

        return self.loss(query_embed, labels) 