from torch import bmm, tanh
from torch.nn import Linear, Module
from torch.nn.functional import softmax


class AdditiveAttention(Module):
    """
    Implements Bahdanau/Additive attention
    https://arxiv.org/abs/1409.0473

    """

    def __init__(self, hidden_size: int, key_size: int = None,
                 query_size: int = None):
        super(AdditiveAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = Linear(key_size, hidden_size, bias=False)
        self.query_layer = Linear(query_size, hidden_size, bias=False)
        self.energy_layer = Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)

        # Calculate scores.
        scores = self.energy_layer(tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = softmax(scores, dim=-1)
        self.alphas = alphas

        # The context vector is the weighted sum of the values.
        context = bmm(alphas, value)

        # context shape: [Batch, 1, 2Dim], alphas shape: [Batch, 1, M]
        return context, alphas
