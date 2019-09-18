import numpy as np

import torch
from copy_task import train_copy_task
from plotting.perplexity import plot_perplexity


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_cuda():
    gpu = torch.cuda.is_available()
    if (gpu):
        device = torch.device("cuda:0")
        print("[CUDA] ", gpu)
        print(device)
    else:
        raise Exception("GPU: no cuda device available")


if __name__ == '__main__':
    set_seeds()
    set_cuda()

    # Simple task: Given a random set of input symbols from a small vocabulary,
    # the goal is to generate back those same symbols.

    perplexities = train_copy_task(use_cuda=True)
    plot_perplexity(perplexities)
