from torch import cat
from torch.nn import GRU, Module
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = GRU(input_size, hidden_size, num_layers,
                       batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """

        # make all elements in the batch x the same length by padding with zero
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        # [num_layers, batch, 2*dim]
        final = cat([fwd_final, bwd_final], dim=2)

        return output, final
