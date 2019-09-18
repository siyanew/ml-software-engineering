import numpy as np

from model.Batch import Batch
from torch import LongTensor, from_numpy


def data_gen(num_words=11, batch_size=16, num_batches=100, length=10,
             pad_index=0, sos_index=1, use_cuda=True):
    """Generate random data for a src-tgt copy task."""
    for i in range(num_batches):
        data = from_numpy(
            np.random.randint(1, num_words, size=(batch_size, length)))

        data = data.type(LongTensor)
        data[:, 0] = sos_index
        data = data.cuda() if use_cuda else data
        src = data[:, 1:]
        trg = data
        src_lengths = [length-1] * batch_size
        trg_lengths = [length] * batch_size
        yield Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)
