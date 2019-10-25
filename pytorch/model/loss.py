from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import nll_loss


def nll_loss(output: Tensor, target: Tensor) -> Tensor:
    return nll_loss(output, target)


def cross_entropy(padding_idx: int) -> CrossEntropyLoss:
    # Encapsulate the padding idx in the criterion function
    return CrossEntropyLoss(ignore_index=padding_idx)
