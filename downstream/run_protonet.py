# Original Copyright (c) 2021, Thomas Dopierre. Licensed under Apache License 2.0
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import json
import argparse
from encoders.bert_encoder import BERTEncoder
from encoders.sbert_encoder import SentEncoder
from protaugment.utils.data import FewShotDataLoader
from protaugment.utils.python import now, set_seeds
import collections
import os
from typing import Callable, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.autograd import Variable
import warnings
import logging
import copy
from protaugment.utils.few_shot import create_ARSC_train_episode
from protaugment.utils.math import euclidean_dist, cosine_similarity

logging.basicConfig()
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

class ProtoNet(nn.Module):
    def __init__(self, encoder, metric="euclidean", label_fn=None, zero_shot=False):
        super(ProtoNet, self).__init__()

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

        has_augmentations = ("x_augment" in sample)

        if has_augmentations:
            # When using augmentations
            augmentations = sample["x_augment"]

            n_augmentations_samples = len(sample["x_augment"])
            n_augmentations_per_sample = [len(item['augmentations']) for item in augmentations]
            assert set(n_augmentations_per_sample) == {5}
            n_augmentations_per_sample = n_augmentations_per_sample[0]

            supports = [item["sentence"] for xs_ in xs for item in xs_]
            queries = [item["sentence"] for xq_ in xq for item in xq_]
            augmentations_supports = [[item2["text"] for item2 in item["augmentations"]] for item in sample["x_augment"]]
            augmentation_queries = [item["sentence"] for item in sample["x_augment"]]

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

        # import pdb; pdb.set_trace()
        
        # avg pooling
        z_support = z_support.mean(dim=[1])
        if self.metric == "euclidean":
            supervised_dists = euclidean_dist(z_query, z_support)
            if has_augmentations:
                unsupervised_dists = euclidean_dist(z_aug_query, z_aug_support)
        elif self.metric == "cosine":
            supervised_dists = (-cosine_similarity(z_query, z_support) + 1) * 5
            if has_augmentations:
                unsupervised_dists = (-cosine_similarity(z_aug_query, z_aug_support) + 1) * 5
        else:
            raise NotImplementedError

        # Supervised loss
        # -- legacy
        # log_p_y = torch_functional.log_softmax(-supervised_dists, dim=1).view(n_class, n_query, -1)
        # dists.view(n_class, n_query, -1)
        # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        # -- NEW
        from torch.nn import CrossEntropyLoss
        supervised_loss = CrossEntropyLoss()(-supervised_dists, target_inds.reshape(-1))
        _, y_hat_supervised = (-supervised_dists).max(1)
        acc_val_supervised = torch.eq(y_hat_supervised, target_inds.reshape(-1)).float().mean()

        if has_augmentations:
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

    def loss_softkmeans(self, sample):
        xs = sample['xs']  # support
        xq = sample['xq']  # query
        xu = sample['xu']  # unlabeled

        n_class = len(xs)
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False).to(device)

        x = [item["sentence"] for xs_ in xs for item in xs_] + [item["sentence"] for xq_ in xq for item in xq_] + [item["sentence"] for item in xu]
        z = self.encoder.embed_sentences(x)
        z_dim = z.size(-1)

        zs = z[:n_class * n_support]
        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support: (n_class * n_support) + (n_class * n_query)]
        zu = z[(n_class * n_support) + (n_class * n_query):]

        distances_to_proto = euclidean_dist(
            torch.cat((zs, zu)),
            z_proto
        )

        distances_to_proto_normed = torch.nn.Softmax(dim=-1)(-distances_to_proto)

        refined_protos = list()
        for class_ix in range(n_class):
            z = torch.cat(
                (zs[class_ix * n_support: (class_ix + 1) * n_support], zu)
            )
            d = torch.cat(
                (torch.ones(n_support).to(device),
                 distances_to_proto_normed[(n_class * n_support):, class_ix])
            )
            refined_proto = ((z.t() * d).sum(1) / d.sum())
            refined_protos.append(refined_proto.view(1, -1))
        refined_protos = torch.cat(refined_protos)

        if self.metric == "euclidean":
            dists = euclidean_dist(zq, refined_protos)
        elif self.metric == "cosine":
            dists = (-cosine_similarity(zq, refined_protos) + 1) * 5
        else:
            raise NotImplementedError

        log_p_y = torch_functional.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        dists.view(n_class, n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            "metrics": {
                "acc": acc_val.item(),
                "loss": loss_val.item(),
            },
            'dists': dists,
            'target': target_inds
        }

    def loss_consistency(self, sample):
        x_augment = sample["x_augment"]
        n_samples = len(x_augment)

        # x_augment = [(A, [A_1, A_2, ..., A_n]), (B, [B_1, B_2, ..., B_m])]
        lengths = [1 + len(augments) for sentence, augments in x_augment]

        x = list()
        for sentence, augs in x_augment:
            x.append(sentence)
            x += augs

        z = self.encoder.embed_sentences(x)
        assert len(z) == sum(lengths)

        i = 0
        original_embeddings = list()
        augmented_embeddings = list()
        for length in lengths:
            original_embeddings.append(z[i])
            augmented_embeddings.append(z[i + 1:i + length + 1])
            i += length

        augmented_embeddings = [a.mean(0) for a in augmented_embeddings]
        if self.metric == "euclidean":
            dists = euclidean_dist(original_embeddings, augmented_embeddings)
        elif self.metric == "cosine":
            dists = (-cosine_similarity(original_embeddings, augmented_embeddings) + 1) * 5
        else:
            raise NotImplementedError

        log_p_y = torch_functional.log_softmax(-dists, dim=1).view(n_samples, n_samples, -1)

        # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        # _, y_hat = log_p_y.max(2)
        # acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        #
        # return loss_val, {
        #     'loss': loss_val.item(),
        #     'acc': acc_val.item(),
        #     'dists': dists,
        #     'target': target_inds
        # }

    def train_step(self, optimizer, episode, supervised_loss_share: float, unlabeled: bool = False, classes=None):
        self.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        if unlabeled:
            loss, loss_dict = self.loss_softkmeans(episode, classes=classes)
        else:
            loss, loss_dict = self.loss(episode, supervised_loss_share=supervised_loss_share, classes=classes)
        loss.backward()
        optimizer.step()

        return loss, loss_dict

    def test_step(self,
                  data_loader: FewShotDataLoader,
                  n_support: int,
                  n_query: int,
                  n_classes: int,
                  n_unlabeled: int = 0,
                  n_episodes: int = 1000):
        metrics = collections.defaultdict(list)

        self.eval()
        for i in range(n_episodes):
            episode, classes = data_loader.create_episode(
                n_support=n_support,
                n_query=n_query,
                n_unlabeled=n_unlabeled,
                n_classes=n_classes,
                n_augment=0
            )

            with torch.no_grad():
                if n_unlabeled:
                    loss, loss_dict = self.loss_softkmeans(episode)
                else:
                    loss, loss_dict = self.loss(episode, supervised_loss_share=0, classes=classes)

            for k, v in loss_dict["metrics"].items():
                metrics[k].append(v)

        return {
            key: np.mean(value) for key, value in metrics.items()
        }

    def train_step_ARSC(self, data_path: str, optimizer, n_unlabeled: int):
        episode = create_ARSC_train_episode(prefix=data_path, n_support=5, n_query=5, n_unlabeled=n_unlabeled)

        self.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        if n_unlabeled:
            loss, loss_dict = self.loss_softkmeans(episode)
        else:
            loss, loss_dict = self.loss(episode)
        loss.backward()
        optimizer.step()

        return loss, loss_dict



def run_proto(
        train_path: str,
        model_name_or_path: str,
        n_support: int,
        n_query: int,
        n_classes: int,
        unlabeled_path: str = None,
        valid_path: str = None,
        test_path: str = None,
        n_unlabeled: int = 0,
        n_augment: int = 0,
        output_path: str = f'runs/{now()}',
        max_iter: int = 10000,
        evaluate_every: int = 100,
        early_stop: int = None,
        n_test_episodes: int = 1000,
        log_every: int = 10,
        lr: float = 2e-5,
        metric: str = "euclidean",
        encoder: str = 'bert',
        draft: bool = False,
        zero_shot: bool = False,
        label_fn: str = None,
        arsc_format: bool = False,
        data_path: str = None,
        supervised_loss_share_fn: Callable[[int, int], float] = lambda x, y: 1 - (x / y)
):
    # if output_path:
    #     if os.path.exists(output_path) and len(os.listdir(output_path)):
    #         raise FileExistsError(f"Output path {output_path} already exists. Exiting.")

    # --------------------
    # Creating Log Writers
    # --------------------
    os.makedirs(output_path,exist_ok=True)
    log_dict = dict(train=list())


    # Load model
    if encoder == 'bert':
        bert = BERTEncoder(model_name_or_path).to(device)
    elif encoder == 'sentbert':
        bert = SentEncoder(model_name_or_path).to(device)
    else:
        raise NotImplementedError

    protonet = ProtoNet(encoder=bert, metric=metric, label_fn=label_fn, zero_shot=zero_shot)
    optimizer = torch.optim.Adam(protonet.parameters(), lr=lr)

    # Load data
    if not arsc_format:
        train_data_loader = FewShotDataLoader(train_path, unlabeled_file_path=unlabeled_path)
        logger.info(f"train labels: {train_data_loader.data_dict.keys()}")

        if valid_path:
            valid_data_loader = FewShotDataLoader(valid_path)
            logger.info(f"valid labels: {valid_data_loader.data_dict.keys()}")
        else:
            valid_data_loader = None

        if test_path:
            test_data_loader = FewShotDataLoader(test_path)
            logger.info(f"test labels: {test_data_loader.data_dict.keys()}")
        else:
            test_data_loader = None
    else:
        raise NotImplementedError
        train_data_loader = None
        valid_data_loader = None
        test_data_loader = None

    train_metrics = collections.defaultdict(list)
    n_eval_since_last_best = 0
    best_valid_acc = 0.0
    best_protonet = copy.deepcopy(protonet)

    for step in range(max_iter):
        if not arsc_format:
            episode, classes = train_data_loader.create_episode(
                n_support=n_support,
                n_query=n_query,
                n_classes=n_classes,
                n_unlabeled=n_unlabeled,
                n_augment=n_augment
            )
        else:
            raise NotImplementedError
            # episode, classes = create_ARSC_train_episode(n_support=5, n_query=5)

        supervised_loss_share = supervised_loss_share_fn(step, max_iter)
        loss, loss_dict = protonet.train_step(optimizer=optimizer, episode=episode, unlabeled=(n_unlabeled > 0), supervised_loss_share=supervised_loss_share, classes=classes)

        for key, value in loss_dict["metrics"].items():
            train_metrics[key].append(value)

        # Logging
        if (step + 1) % log_every == 0:
            # TODO! logging
            # for key, value in train_metrics.items():

            logger.info(f"train | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in train_metrics.items()]))

            train_metrics = collections.defaultdict(list)

        if valid_path or test_path:
            if (step + 1) % evaluate_every == 0:
                for path, set_type, set_data_loader in zip(
                        [valid_path, test_path],
                        ["valid", "test"],
                        [valid_data_loader, valid_data_loader]
                ):
                    if path:
                        if not arsc_format:
                            set_results = protonet.test_step(
                                data_loader=set_data_loader,
                                n_unlabeled=n_unlabeled,
                                n_support=n_support,
                                n_query=n_query,
                                n_classes=n_classes,
                                n_episodes=n_test_episodes
                            )
                        else:
                            raise NotImplementedError
                            set_results = protonet.test_step_ARSC(
                                data_path=data_path,
                                n_unlabeled=n_unlabeled,
                                n_episodes=n_test_episodes,
                                set_type={"valid": "dev", "test": "test"}[set_type]
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


def main(args):
    logger.debug(f"Received args: {json.dumps(args.__dict__, sort_keys=True, ensure_ascii=False, indent=1)}")

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

    for cv in cvs:
        args.train_path = os.path.join(orig_args.train_path,cv,'train.jsonl')
        args.valid_path = os.path.join(orig_args.valid_path,cv,'valid.jsonl')
        args.test_path = os.path.join(orig_args.test_path,cv,'test.jsonl')
        args.output_path = os.path.join(orig_args.output_path, cv)
        
        set_seeds(args.seed)
        # Run
        run_proto(
            train_path=args.train_path,
            valid_path=args.valid_path,
            test_path=args.test_path,
            output_path=args.output_path,
            unlabeled_path=args.unlabeled_path,

            model_name_or_path=args.model_name_or_path,
            n_unlabeled=args.n_unlabeled,

            n_support=args.n_support,
            n_query=args.n_query,
            n_classes=args.n_classes,
            n_test_episodes=args.n_test_episodes,
            n_augment=args.n_augment,

            max_iter=args.max_iter,
            evaluate_every=args.evaluate_every,

            metric=args.metric,
            early_stop=args.early_stop,
            arsc_format=args.arsc_format,
            data_path=args.data_path,
            log_every=args.log_every,

            supervised_loss_share_fn=supervised_loss_share_fn,

            encoder=args.encoder,
            draft=args.draft,
            lr=args.lr,
            zero_shot=args.zero_shot,
            label_fn=args.label_fn
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--valid-path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--test-path", type=str, default=None, help="Path to testing data")
    parser.add_argument("--unlabeled-path", type=str, default=None, help="Path to data containing augmentations used for consistency")
    parser.add_argument("--data-path", type=str, default=None, help="Path to data (ARSC only)")

    parser.add_argument("--output-path", type=str, default=f'runs/{now()}')
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Transformer model to use")
    parser.add_argument("--n-unlabeled", type=int, help="Number of unlabeled data points per class (proto++)", default=0)
    parser.add_argument("--max-iter", type=int, default=10000, help="Max number of training episodes")
    parser.add_argument("--evaluate-every", type=int, default=100, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")
    parser.add_argument("--early-stop", type=int, default=0, help="Number of worse evaluation steps before stopping. 0=disabled")

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=5, help="Number of support points for each class")
    parser.add_argument("--n-query", type=int, default=5, help="Number of query points for each class")
    parser.add_argument("--n-classes", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--n-augment", type=int, default=5, help="Number of augmented samples to take")
    parser.add_argument("--n-test-episodes", type=int, default=1000, help="Number of episodes during evaluation (valid, test)")

    # Metric to use in proto distance calculation
    parser.add_argument("--metric", type=str, default="euclidean", help="Metric to use", choices=("euclidean", "cosine"))

    # Supervised loss share
    parser.add_argument("--supervised-loss-share-power", default=1.0, type=float, help="supervised_loss_share = 1 - (x/y) ** <param>")

    # ARSC data
    parser.add_argument("--arsc-format", default=False, action="store_true", help="Using ARSC few-shot format")

    parser.add_argument("--encoder", type=str, default="bert", help="Metric to use", choices=("bert", "sentbert"))
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--zero-shot", action='store_true')
    parser.add_argument("--label_fn", type=str)
    parser.add_argument("--cv", type=str, default='01,02,03,04,05')
    args = parser.parse_args()

    main(args)
