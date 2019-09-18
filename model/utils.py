import math
import time

from model.AdditiveAttention import AdditiveAttention
from model.Decoder import Decoder
from model.Encoder import Encoder
from model.EncoderDecoder import EncoderDecoder
from model.Generator import Generator
from torch.nn import Embedding


def make_model(src_vocab, tgt_vocab, emb_size: int = 256, hidden_size: int = 512,
               num_layers: int = 1, dropout: float = 0.1, use_cuda: bool = True):
    " Construct a model from hyperparameters."

    attention = AdditiveAttention(hidden_size)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention,
                num_layers=num_layers, dropout=dropout),
        Embedding(src_vocab, emb_size),
        Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab))

    return model.cuda() if use_cuda else model


def run_epoch(data_iter, model, loss_compute, print_every: int = 50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):

        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))
