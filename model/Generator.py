from torch.nn import Linear, Module
from torch.nn.functional import log_softmax


class Generator(Module):
    """
    Define standard linear + softmax generation step.
    This transforms the output of the RNN to probabilities in our vocabulary.
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super(Generator, self).__init__()
        self.proj = Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
