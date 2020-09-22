""" 
Created at 12/04/2019

@author: dimitris.michailidis
"""

import torch


def get_recall(indices, targets):
    """
    Calculates the recall score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
    """

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall


def get_mrr(indices, targets):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        mrr (float): the mrr score
    """

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    # Returns it in a form [matrix_index, index_of_hit]
    # so if k = 3 then the format is
    # 0, index_of_hit
    # 1, index_of_hit
    # 2, index_of_hit
    hits = (targets == indices).nonzero()
    # hits[:, -1] all rows, only last column (so only index_of_hit)
    # to get the ranks we add 1
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr


def evaluate(logits, targets, k=20):
    """
    Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.
        k: determines Recall@K, MRR@K, etc.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    # if k is greater than the number of items, torch.topk throws an error.
    # this is mostly for testing cases, as in the other cases it won't have an effect anyway
    if k > logits.shape[1]:
        k = logits.shape[1]
    # logits here are basically item indices
    _, logits = torch.topk(logits, k, -1)

    recall = get_recall(logits, targets)
    mrr = get_mrr(logits, targets)

    targets = targets.view(-1, 1).expand_as(logits)
    hits = (targets == logits).nonzero()

    return recall, mrr, hits
