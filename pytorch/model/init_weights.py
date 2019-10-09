from torch.nn.init import constant_, normal_, uniform_

from base import BaseModel


def uniform(m: BaseModel, a, b):
    """Initialize the weights from a uniform distribution """
    for name, param in m.named_parameters():
        uniform_(param.data, a, b)


def normal(m: BaseModel, mean, std):
    """Initialize the weights from a normal distribution """
    for name, param in m.named_parameters():
        normal_(param.data, mean=mean, std=std)


def normal_with_bias(m: BaseModel, mean, std, bias):
    """Initialize the weights from a normal distribution, and bias with constant value """
    for name, param in m.named_parameters():
        if 'weight' in name:
            normal_(param.data, mean=mean, std=std)
        else:
            constant_(param.data, bias)
