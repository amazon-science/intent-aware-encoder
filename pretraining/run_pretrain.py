# Original Copyright (c) 2021 Cambridge Language Technology Lab. Licensed under the MIT License.
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

#!/usr/bin/env python
import argparse
import copy
import logging
import os
import time
import shutil
import torch
from run_eval import run_eval
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from iae.contrastive_learning import ContrastiveLearningPairwise
from iae.data_loader import ContrastiveLearningDataset
from iae.drophead import set_drophead
from iae.iae_model import IAEModel
from utils import init_logging

LOGGER = logging.getLogger()

def train(args, data_loader, model, scaler=None, iae_model=None, step_global=0):
    LOGGER.info("train!")
    train_loss = 0
    train_steps = 0
    model.cuda()
    model.train()
    best_acc = 0
    best_model = copy.deepcopy(iae_model)
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()
        # batch_x1: input utterance
        # batch_x2: gold intent
        # batch_x3: gold utterance
        # batch_x4: pseudo intent

        batch_x1, batch_x2, batch_x3, batch_x4 = data
        batch_x_cuda1, batch_x_cuda2, batch_x_cuda3, batch_x_cuda4 = {},{},{},{}
        for k,v in batch_x1.items():
            batch_x_cuda1[k] = v.cuda()
        for k,v in batch_x2.items():
            batch_x_cuda2[k] = v.cuda()
        for k,v in batch_x3.items():
            batch_x_cuda3[k] = v.cuda()
        for k,v in batch_x4.items():
            batch_x_cuda4[k] = v.cuda()

        if args.amp:
            with autocast():
                loss = torch.tensor(0.0, requires_grad=True).cuda()
                loss += model(batch_x_cuda1, batch_x_cuda2)
                loss += model(batch_x_cuda1, batch_x_cuda3)
                if args.pseudo_weight: loss += args.pseudo_weight * model(batch_x_cuda1, batch_x_cuda4)
        else:
            loss = torch.tensor(0.0, requires_grad=True).cuda()
            loss += model(batch_x_cuda1, batch_x_cuda2)
            loss += model(batch_x_cuda1, batch_x_cuda3)
            if args.pseudo_weight: loss += args.pseudo_weight * model(batch_x_cuda1, batch_x_cuda4)

        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
        
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()

        train_loss += loss.item()
        train_steps += 1
        step_global += 1

        if args.eval_during_training and (step_global % args.eval_step == 0):
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_tmp")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            iae_model.save_model(checkpoint_dir)
            
            acc = run_eval(test_path=args.val_path,
                     model_name_or_path=checkpoint_dir,
                     distance_metric=args.distance_metric)

            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(iae_model)
            
            LOGGER.info(f"step:{step_global}, val_acc:{acc}")

    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global, best_model, best_acc
    
def main(args):
    init_logging(LOGGER)
    print(args)

    torch.manual_seed(args.random_seed) 
    # by default 42 is used, also tried 33, 44, 55
    # results don't seem to change too much
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load BERT tokenizer, dense_encoder
    iae_model = IAEModel()
    encoder, tokenizer = iae_model.load_model(
        path=args.model_name_or_path,
        max_length=args.max_length,
        use_cuda=args.use_cuda,
        return_model=True
    )

    # adjust dropout rates
    encoder.embeddings.dropout = torch.nn.Dropout(p=args.dropout_rate)
    for i in range(len(encoder.encoder.layer)):
        # hotfix
        try:
            encoder.encoder.layer[i].attention.self.dropout = torch.nn.Dropout(p=args.dropout_rate)
            encoder.encoder.layer[i].attention.output.dropout = torch.nn.Dropout(p=args.dropout_rate)
        except:
            encoder.encoder.layer[i].attention.attn.dropout = torch.nn.Dropout(p=args.dropout_rate)
            encoder.encoder.layer[i].attention.dropout = torch.nn.Dropout(p=args.dropout_rate)

        encoder.encoder.layer[i].output.dropout  = torch.nn.Dropout(p=args.dropout_rate)

    # set drophead rate
    if args.drophead_rate != 0:
        set_drophead(encoder, args.drophead_rate)

    def collate_fn_batch_encoding(batch):
        sent1, sent2, sent3, sent4 = zip(*batch)
        sent1_toks = tokenizer.batch_encode_plus(
            list(sent1), 
            max_length=args.max_length, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        sent2_toks = tokenizer.batch_encode_plus(
            list(sent2), 
            max_length=args.max_length, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        sent3_toks = tokenizer.batch_encode_plus(
            list(sent3), 
            max_length=args.max_length, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        sent4_toks = tokenizer.batch_encode_plus(
            list(sent4), 
            max_length=args.max_length, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")

        return sent1_toks, sent2_toks, sent3_toks, sent4_toks

    train_set = ContrastiveLearningDataset(
        args.train_path,
        tokenizer=tokenizer,
        random_span_mask=args.random_span_mask,
        draft=args.draft
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn_batch_encoding,
        drop_last=True
    )
    model = ContrastiveLearningPairwise(
        encoder=encoder,
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        use_cuda=args.use_cuda,
        infoNCE_tau=args.infoNCE_tau,
        agg_mode=args.agg_mode
    )
    if args.parallel:
        model.encoder = torch.nn.DataParallel(model.encoder)
        LOGGER.info("using nn.DataParallel")
    # mixed precision training 
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    start = time.time()
    step_global = 0
    best_acc = 0
    best_model = copy.deepcopy(iae_model)
    for epoch in range(1,args.epoch+1):
        LOGGER.info(f"Epoch {epoch}/{args.epoch}")

        # train
        train_loss, step_global, ep_best_model, ep_best_acc = train(args, data_loader=train_loader, model=model, 
                scaler=scaler, iae_model=iae_model, step_global=step_global)
        LOGGER.info(f'loss/train_per_epoch={train_loss}/{epoch}')
        if ep_best_acc > best_acc:
            best_model = ep_best_model
            best_acc = ep_best_acc
        
        # eval after one epoch
        tmp_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_tmp")
        if not os.path.exists(tmp_checkpoint_dir):
            os.makedirs(tmp_checkpoint_dir)
        iae_model.save_model(tmp_checkpoint_dir)
        ep_acc = run_eval(test_path=args.val_path,
                       model_name_or_path=tmp_checkpoint_dir,
                       distance_metric=args.distance_metric
        )
        # remove tmp directory
        shutil.rmtree(tmp_checkpoint_dir)

        if ep_acc > best_acc:
            best_acc = ep_acc
            best_model = copy.deepcopy(iae_model)
        
        LOGGER.info(f"step:{step_global}, val_acc:{ep_acc}")

    best_model.save_model(args.output_dir)

    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info(f"Training Time!{training_hour} hours {training_minute} minutes {training_second} seconds")
    LOGGER.info(f"Best val acc={best_acc}")
    
if __name__ == '__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='train IAE Model')

    # Required
    parser.add_argument('--train_path', type=str, required=True, help='training set directory')
    parser.add_argument('--val_path', type=str, help='validation set directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output')

    parser.add_argument('--model_name_or_path', type=str, \
        help='Directory for pretrained model', \
        default="roberta-base")
    parser.add_argument('--max_length', default=50, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--train_batch_size', default=200, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--infoNCE_tau', default=0.04, type=float) 
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean|mean_std}") 
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--checkpoint_step', type=int, default=10000000)
    parser.add_argument('--parallel', action="store_true") 
    parser.add_argument('--amp', action="store_true", \
        help="automatic mixed precision training")
    parser.add_argument('--random_seed', default=42, type=int)

    # data augmentation config
    parser.add_argument('--dropout_rate', default=0.1, type=float) 
    parser.add_argument('--drophead_rate', default=0.0, type=float)
    parser.add_argument('--random_span_mask', default=5, type=int, 
            help="number of chars to be randomly masked on one side of the input") 

    parser.add_argument('--distance_metric', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--eval_during_training', action='store_true')
    parser.add_argument('--eval_step', default=200, type=int)
    parser.add_argument('--pseudo_weight', default=2, type=float)
    args = parser.parse_args()

    main(args)
