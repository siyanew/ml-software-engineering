from typing import Callable, List

from torch import LongTensor, Tensor
from torchtext.data import Field, Iterator


class BaseTextIterator:
    def __init__(self, train: Iterator, valid: Iterator, test: Iterator,
                 SRC: Field, TRG: Field, src_tokenize: Callable, trg_tokenize: Callable):
        self.train = train
        self.valid = valid
        self.test = test
        self.SRC = SRC
        self.src_tokenize = src_tokenize
        self.TRG = TRG
        self.trg_tokenize = trg_tokenize

    def split_validation(self) -> Iterator:
        return self.valid

    def split_train(self) -> Iterator:
        return self.train

    def split_test(self) -> Iterator:
        return self.test

    def tokens_to_tensor(self, src: List[str]) -> (Tensor, Tensor):
        # TODO: this function assumes that we use packed sentences

        tokenize = ['<sos>'] + src + ['<eos>']
        numerical = [self.SRC.vocab.stoi[t] for t in tokenize]
        length = LongTensor([len(numerical)])
        tensor = LongTensor(numerical).unsqueeze(1)

        return tensor, length

    def tensor_to_tokens(self, translation: Tensor) -> List[str]:
        translation = [self.TRG.vocab.itos[t] for t in translation]

        return translation
