import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy(padding_idx):
    # Encapsulate the padding idx in the criterion function
    return nn.CrossEntropyLoss(ignore_index=padding_idx)
