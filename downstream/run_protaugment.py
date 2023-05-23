# Original Copyright (c) 2021, Thomas Dopierre. Licensed under Apache License 2.0
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import json
import argparse

from encoders.bert_encoder import BERTEncoder
from encoders.sbert_encoder import SentEncoder

from protaugment.paraphrase.utils.data import FewShotDataset, FewShotSSLFileDataset
from protaugment.utils.python import now, set_seeds
import collections
import os
from typing import Callable, Union
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import warnings
import logging
import copy
from protaugment.utils.math import euclidean_dist, cosine_similarity
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_label_file(label_fn):
    class2name_mp = {}
    replace_pair = [("lightdim", "light dim"), ("lightchange", "light change"), ("lightup", "light up"),
                    ("commandstop", "command stop"), ("lighton", "light on"), ("dontcare", "don't care"),
                    ("lightoff", "light off"), ("querycontact", "query contact"), ("addcontact", "add contact"),
                    ("sendemail", "send email"), ("createoradd", "create or add"), ("qa", "what")]

    with open(label_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip("\n")
            tmp = line.replace("_", " ").replace("/", " ")
            for x, y in replace_pair:
                tmp = tmp.replace(x, y)
            class2name_mp[line] = tmp
    return class2name_mp

class ProtAugmentNet(nn.Module):
    def __init__(self, encoder, metric="euclidean", label_fn=None, zero_shot=False):
        super(ProtAugmentNet, self).__init__()

        self.encoder = encoder
        self.metric = metric
        self.class2name_mp = load_label_file(label_fn) if label_fn else None
        self.zero_shot = zero_shot
        assert self.metric in ('euclidean', 'cosine')

    def loss(self, sample, supervised_loss_share: float = 0, classes=None):
        """
        :param supervised_loss_share: share of supervised loss in total loss
        :param sample: {
            "xs": [
                [support_A_1, support_A_2, ...],
                [support_B_1, support_B_2, ...],
                [support_C_1, support_C_2, ...],
                ...
            ],
            "xq": [
                [query_A_1, query_A_2, ...],
                [query_B_1, query_B_2, ...],
                [query_C_1, query_C_2, ...],
                ...
            ],
            "x_augment":[
                {
                    "src_text": str,
                    "tgt_texts: List[str]
                }, .
            ]
        }
        :return:
        """
        xs = sample['xs']  # support
        xq = sample['xq']  # query

        n_class = len(xs)
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False).to(device)

        # x_augment is not always present in `sample`
        # Indeed, at evaluation / test time, the network is judged on a regular meta-learning episode (i.e. only samples and query points)
        has_augment = "x_augment" in sample

        if has_augment:
            augmentations = sample["x_augment"]

            # expection: n_augmentations_samples == n_classes
            # actual: n_augmentations_samples == n_unlabeled
            n_augmentations_samples = len(sample["x_augment"])
            n_augmentations_per_sample = [len(item['tgt_texts']) for item in augmentations]
            assert len(set(n_augmentations_per_sample)) == 1
            n_augmentations_per_sample = n_augmentations_per_sample[0]

            supports = [item["sentence"] for xs_ in xs for item in xs_]
            queries = [item["sentence"] for xq_ in xq for item in xq_]
            augmentations_supports = [[item2 for item2 in item["tgt_texts"]] for item in sample["x_augment"]]
            augmentation_queries = [item["src_text"] for item in sample["x_augment"]]

            # Encode
            x = supports + queries + [item2 for item1 in augmentations_supports for item2 in item1] + augmentation_queries
            z = self.encoder.embed_sentences(x)
            z_dim = z.size(-1)

            # Dispatch
            z_support = z[:len(supports)].view(n_class, n_support, z_dim)
            z_query = z[len(supports):len(supports) + len(queries)]
            z_aug_support = (z[len(supports) + len(queries):len(supports) + len(queries) + n_augmentations_per_sample * n_augmentations_samples]
                             .view(n_augmentations_samples, n_augmentations_per_sample, z_dim).mean(dim=[1]))
            z_aug_query = z[-len(augmentation_queries):]
        else:
            # When not using augmentations
            supports = [item["sentence"] for xs_ in xs for item in xs_]
            queries = [item["sentence"] for xq_ in xq for item in xq_]

            # Encode
            x = supports + queries
            z = self.encoder.embed_sentences(x)
            z_dim = z.size(-1)

            # Dispatch
            z_support = z[:len(supports)].view(n_class, n_support, z_dim)
            z_query = z[len(supports):len(supports) + len(queries)]

        if self.class2name_mp:
            class_names = [self.class2name_mp[classes[i]] for i in range(len(xs))]
            z_class = self.encoder.embed_sentences(class_names).view(n_class, 1, z_dim)
            if self.zero_shot:
                z_support = z_class
            else:
                z_support = torch.cat([z_support, z_class], dim=1)

        # avg pooling
        z_support = z_support.mean(dim=[1])
        if self.metric == "euclidean":
            supervised_dists = euclidean_dist(z_query, z_support)
            if has_augment:
                unsupervised_dists = euclidean_dist(z_aug_query, z_aug_support)
        elif self.metric == "cosine":
            supervised_dists = (-cosine_similarity(z_query, z_support) + 1) * 5
            if has_augment:
                unsupervised_dists = (-cosine_similarity(z_aug_query, z_aug_support) + 1) * 5
        else:
            raise NotImplementedError

        from torch.nn import CrossEntropyLoss
        supervised_loss = CrossEntropyLoss()(-supervised_dists, target_inds.reshape(-1))
        _, y_hat_supervised = (-supervised_dists).max(1)
        acc_val_supervised = torch.eq(y_hat_supervised, target_inds.reshape(-1)).float().mean()

        if has_augment:
            # Unsupervised loss
            unsupervised_target_inds = torch.range(0, n_augmentations_samples - 1).to(device).long()
            unsupervised_loss = CrossEntropyLoss()(-unsupervised_dists, unsupervised_target_inds)
            _, y_hat_unsupervised = (-unsupervised_dists).max(1)
            acc_val_unsupervised = torch.eq(y_hat_unsupervised, unsupervised_target_inds.reshape(-1)).float().mean()

            # Final loss
            assert 0 <= supervised_loss_share <= 1
            final_loss = (supervised_loss_share) * supervised_loss + (1 - supervised_loss_share) * unsupervised_loss

            return final_loss, {
                "metrics": {
                    "supervised_acc": acc_val_supervised.item(),
                    "unsupervised_acc": acc_val_unsupervised.item(),
                    "supervised_loss": supervised_loss.item(),
                    "unsupervised_loss": unsupervised_loss.item(),
                    "supervised_loss_share": supervised_loss_share,
                    "final_loss": final_loss.item(),
                },
                "supervised_dists": supervised_dists,
                "unsupervised_dists": unsupervised_dists,
                "target": target_inds
            }

        return supervised_loss, {
            "metrics": {
                "acc": acc_val_supervised.item(),
                "loss": supervised_loss.item(),
            },
            "dists": supervised_dists,
            "target": target_inds
        }

    def train_step(self, optimizer, episode, supervised_loss_share: float, classes):
        self.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss, loss_dict = self.loss(episode, supervised_loss_share=supervised_loss_share, classes=classes)
        loss.backward()
        optimizer.step()

        return loss, loss_dict

    def test_step(self, dataset: FewShotDataset, n_episodes: int = 1000):
        metrics = collections.defaultdict(list)

        self.eval()
        for i in range(n_episodes):
            episode, classes = dataset.get_episode()

            with torch.no_grad():
                loss, loss_dict = self.loss(episode, supervised_loss_share=1, classes=classes)

            for k, v in loss_dict["metrics"].items():
                metrics[k].append(v)

        return {
            key: np.mean(value) for key, value in metrics.items()
        }


def run_protaugment(
        # Compulsory!
        data_path: str,
        train_labels_path: str,
        model_name_or_path: str,

        # Few-shot Stuff
        n_support: int,
        n_query: int,
        n_classes: int,
        metric: str = "euclidean",

        # Optional path to augmented data
        unlabeled_path: str = None,

        # Path training data ONLY (optional)
        train_path: str = None,

        # Validation & test
        valid_labels_path: str = None,
        test_labels_path: str = None,
        evaluate_every: int = 100,
        n_test_episodes: int = 1000,

        # Logging & Saving
        output_path: str = f'runs/{now()}',
        log_every: int = 10,
        lr: float = 2e-5,

        # Training stuff
        max_iter: int = 10000,
        early_stop: int = None,

        # Augmentation & paraphrase
        n_unlabeled: int = 5,
        paraphrase_model_name_or_path: str = None,
        paraphrase_tokenizer_name_or_path: str = None,
        paraphrase_num_beams: int = None,
        paraphrase_beam_group_size: int = None,
        paraphrase_diversity_penalty: float = None,
        paraphrase_filtering_strategy: str = None,
        paraphrase_drop_strategy: str = None,
        paraphrase_drop_chance_speed: str = None,
        paraphrase_drop_chance_auc: float = None,
        supervised_loss_share_fn: Callable[[int, int], float] = lambda x, y: 1 - (x / y),

        paraphrase_generation_method: str = None,

        augmentation_data_path: str = None,

        encoder: str = 'bert',
        draft: bool = False,
        zero_shot: bool = False,
        label_fn: str = None,
):
    # --------------------
    # Creating Log Writers
    # --------------------
    os.makedirs(output_path,exist_ok=True)
    log_dict = dict(train=list())

    # ----------
    # Load model
    # ----------
    if encoder == 'bert':
        bert = BERTEncoder(model_name_or_path).to(device)
    elif encoder == 'sentbert':
        bert = SentEncoder(model_name_or_path).to(device)
    else:
        raise NotImplementedError

    protonet: ProtAugmentNet = ProtAugmentNet(encoder=bert, metric=metric, label_fn=label_fn, zero_shot=zero_shot)
    optimizer = torch.optim.Adam(protonet.parameters(), lr=lr)

    # ------------------
    # Load Train Dataset
    # ------------------
    if augmentation_data_path:
        # If an augmentation data path is provided, uses those pre-generated augmentations
        train_dataset = FewShotSSLFileDataset(
            data_path=train_path if train_path else data_path,
            labels_path=train_labels_path,
            n_classes=n_classes,
            n_support=n_support,
            n_query=n_query,
            n_unlabeled=n_unlabeled,
            unlabeled_file_path=augmentation_data_path,
        )

    else:
        raise NotImplementedError
    logger.info(f"Train dataset has {len(train_dataset)} items")

    # ---------
    # Load data
    # ---------
    logger.info(f"train labels: {train_dataset.data.keys()}")
    valid_dataset: FewShotDataset = None
    if valid_labels_path:
        log_dict["valid"] = list()
        valid_dataset = FewShotDataset(data_path=data_path, labels_path=valid_labels_path, n_classes=n_classes, n_support=n_support, n_query=n_query)
        logger.info(f"valid labels: {valid_dataset.data.keys()}")
        assert len(set(valid_dataset.data.keys()) & set(train_dataset.data.keys())) == 0

    test_dataset: FewShotDataset = None
    if test_labels_path:
        log_dict["test"] = list()
        test_dataset = FewShotDataset(data_path=data_path, labels_path=test_labels_path, n_classes=n_classes, n_support=n_support, n_query=n_query)
        logger.info(f"test labels: {test_dataset.data.keys()}")
        assert len(set(test_dataset.data.keys()) & set(train_dataset.data.keys())) == 0

    train_metrics = collections.defaultdict(list)
    n_eval_since_last_best = 0
    best_valid_acc = 0.0
    best_protonet = copy.deepcopy(protonet)

    for step in range(max_iter):
        episode, classes = train_dataset.get_episode()

        supervised_loss_share = supervised_loss_share_fn(step, max_iter)
        loss, loss_dict = protonet.train_step(optimizer=optimizer, episode=episode, supervised_loss_share=supervised_loss_share, classes=classes)

        for key, value in loss_dict["metrics"].items():
            train_metrics[key].append(value)

        # Logging
        if (step + 1) % log_every == 0:
            # TODO! logging
            # for key, value in train_metrics.items():
                
            logger.info(f"train | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in train_metrics.items()]))
            train_metrics = collections.defaultdict(list)

        if valid_labels_path or test_labels_path:
            if (step + 1) % evaluate_every == 0:
                for labels_path, set_type, set_dataset in zip(
                        [valid_labels_path, test_labels_path],
                        ["valid", "test"],
                        [valid_dataset, test_dataset]
                ):
                    if set_dataset:

                        set_results = protonet.test_step(
                            dataset=set_dataset,
                            n_episodes=n_test_episodes
                        )

                        # TODO! logging
                        # for key, val in set_results.items():
                        logger.info(f"{set_type} | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in set_results.items()]))
                        if set_type == "valid":
                            if set_results["acc"] > best_valid_acc:
                                best_valid_acc = set_results["acc"]
                                best_protonet = copy.deepcopy(protonet)
                                n_eval_since_last_best = 0
                                logger.info(f"Better eval results!")
                                # TODO! logging
                            else:
                                n_eval_since_last_best += 1
                                logger.info(f"Worse eval results ({n_eval_since_last_best}/{early_stop})")

                if draft:
                    break

                if early_stop and n_eval_since_last_best >= early_stop:
                    logger.warning(f"Early-stopping.")
                    break
    
    # save model in encoder.pkl
    with open(os.path.join(output_path,'encoder.pkl'), 'wb') as handle:
        torch.save(best_protonet.state_dict(), handle)
    print(os.path.join(output_path,'encoder.pkl'))

def main(args):
    logger.debug(f"Received args: {json.dumps(args.__dict__, sort_keys=True, ensure_ascii=False, indent=1)}")

    # # Check if data path(s) exist
    # for arg in [args.data_path, args.train_labels_path, args.valid_labels_path, args.test_labels_path]:
    #     if arg and not os.path.exists(arg):
    #         raise FileNotFoundError(f"Data @ {arg} not found.")

    # Create supervised_loss_share_fn
    def get_supervised_loss_share_fn(supervised_loss_share_power: Union[int, float]) -> Callable[[int, int], float]:
        def _supervised_loss_share_fn(current_step: int, max_steps: int) -> float:
            assert current_step <= max_steps
            return 1 - (current_step / max_steps) ** supervised_loss_share_power

        return _supervised_loss_share_fn

    supervised_loss_share_fn = get_supervised_loss_share_fn(args.supervised_loss_share_power)

    if args.n_support == 0:
        args.n_support = 1
        args.zero_shot = True

    orig_args = copy.deepcopy(args)
    cvs = args.cv.split(",")
    # for cv in ['01', '02', '03', '04', '05']:
    for cv in cvs:
        args.train_labels_path = os.path.join(orig_args.train_labels_path,cv,'labels.train.txt')
        args.valid_labels_path = os.path.join(orig_args.valid_labels_path,cv,'labels.valid.txt')
        args.test_labels_path = os.path.join(orig_args.test_labels_path,cv,'labels.test.txt')
        args.output_path = os.path.join(orig_args.output_path, cv)
        
        set_seeds(args.seed)
        # Run
        run_protaugment(
            data_path=args.data_path,
            train_labels_path=args.train_labels_path,
            train_path=args.train_path,
            model_name_or_path=args.model_name_or_path,
            n_support=args.n_support,
            n_query=args.n_query,
            n_classes=args.n_classes,
            metric=args.metric,

            valid_labels_path=args.valid_labels_path,
            test_labels_path=args.test_labels_path,
            evaluate_every=args.evaluate_every,
            n_test_episodes=args.n_test_episodes,

            output_path=args.output_path,
            log_every=args.log_every,
            max_iter=args.max_iter,
            early_stop=args.early_stop,

            unlabeled_path=args.unlabeled_path,
            n_unlabeled=args.n_unlabeled,

            # Paraphrase generation model
            paraphrase_model_name_or_path=args.paraphrase_model_name_or_path,
            paraphrase_tokenizer_name_or_path=args.paraphrase_tokenizer_name_or_path,
            paraphrase_num_beams=args.paraphrase_num_beams,
            paraphrase_beam_group_size=args.paraphrase_beam_group_size,
            paraphrase_filtering_strategy=args.paraphrase_filtering_strategy,
            paraphrase_drop_strategy=args.paraphrase_drop_strategy,
            paraphrase_drop_chance_speed=args.paraphrase_drop_chance_speed,
            paraphrase_drop_chance_auc=args.paraphrase_drop_chance_auc,
            supervised_loss_share_fn=supervised_loss_share_fn,

            # Other paraphrase generation method
            paraphrase_generation_method=args.paraphrase_generation_method,

            # Or just path to augmented data
            augmentation_data_path=args.augmentation_data_path,

            encoder=args.encoder,
            draft=args.draft,
            lr=args.lr,
            zero_shot=args.zero_shot,
            label_fn=args.label_fn
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the full data")
    parser.add_argument("--train-labels-path", type=str, required=True, help="Path to train labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--train-path", type=str, help="Path to training data (if provided, picks training data from this path instead of --data-path")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Language Model PROTAUGMENT initializes from")

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=5, help="Number of support points for each class")
    parser.add_argument("--n-query", type=int, default=5, help="Number of query points for each class")
    parser.add_argument("--n-classes", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance function to use", choices=("euclidean", "cosine"))

    # Validation & test
    parser.add_argument("--valid-labels-path", type=str, required=True, help="Path to valid labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--test-labels-path", type=str, required=True, help="Path to test labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--evaluate-every", type=int, default=100, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--n-test-episodes", type=int, default=1000, help="Number of episodes during evaluation (valid, test)")

    # Logging & Saving
    parser.add_argument("--output-path", type=str, default=f'runs/{now()}')
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")

    # Training stuff
    parser.add_argument("--max-iter", type=int, default=10000, help="Max number of training episodes")
    parser.add_argument("--early-stop", type=int, default=0, help="Number of worse evaluation steps before stopping. 0=disabled")

    # Augmentation & Paraphrase
    parser.add_argument("--unlabeled-path", type=str, help="Path to raw data (one sentence per line), to generate paraphrases from.")
    parser.add_argument("--n-unlabeled", type=int, help="Number of rows to draw from `--unlabeled-path` at each episode", default=5)

    # If you are using a paraphrase generation model
    parser.add_argument("--paraphrase-model-name-or-path", type=str, help="Name or path to the paraphrase model")
    parser.add_argument("--paraphrase-tokenizer-name-or-path", type=str, help="Name or path to the paraphrase model's tokenizer")
    parser.add_argument("--paraphrase-num-beams", type=int, help="Total number of beams in the Beam Search algorithm")
    parser.add_argument("--paraphrase-beam-group-size", type=int, help="Size of each group of beams")
    parser.add_argument("--paraphrase-diversity-penalty", type=float, help="Diversity penalty (float) to use in Diverse Beam Search")
    parser.add_argument("--paraphrase-filtering-strategy", type=str, choices=["bleu", "clustering"], help="Filtering strategy to apply to a group of generated paraphrases to choose the one to pick. `bleu` takes the sentence which has the highest bleu_score w/r to the original sentence.")
    parser.add_argument("--paraphrase-drop-strategy", type=str, choices=["bigram", "unigram"], help="Drop strategy to use to contraint the paraphrase generation. If not set, no words are forbidden.")
    parser.add_argument("--paraphrase-drop-chance-speed", type=str, choices=["flat", "down", "up"], help="Curve of drop probability depending on token position in the sentence")
    parser.add_argument("--paraphrase-drop-chance-auc", type=float, help="Area of the drop chance probability w/r to the position in the sentence. When --paraphrase-drop-chance-speed=flat (same chance for all tokens to be forbidden no matter the position in the sentence), this parameter equals to p_{mask}")

    # If you want to use another augmentation technique, e.g. EDA (https://github.com/jasonwei20/eda_nlp/)
    parser.add_argument("--paraphrase-generation-method", type=str, choices=["eda"])

    # Augmentation file path (optional, but if provided it will be used)
    parser.add_argument("--augmentation-data-path", type=str, help="Path to a .jsonl file containing augmentations. Refer to `back-translation.jsonl` for an example")

    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")

    # Supervised loss share
    parser.add_argument("--supervised-loss-share-power", default=1.0, type=float, help="supervised_loss_share = 1 - (x/y) ** <param>")
    
    parser.add_argument("--encoder", type=str, default="bert", help="Metric to use", choices=("bert", "sentbert"))
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--zero-shot", action='store_true')
    parser.add_argument("--label_fn", type=str)
    parser.add_argument("--cv", type=str, default='01,02,03,04,05')

    args = parser.parse_args()

    main(args)
