from torch.nn.init import normal_, uniform_

from base import BaseModel


def uniform(m: BaseModel, a, b):
    """Initialize the weights from a uniform distribution """
    for name, param in m.named_parameters():
        uniform_(param.data, a, b)


def normal(m: BaseModel, mean, std):
    """Initialize the weights from a normal distribution """
    for name, param in m.named_parameters():
        normal_(param.data, mean=mean, std=std)
