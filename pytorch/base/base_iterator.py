from torchtext.data import Field, Iterator


class BaseTextIterator:
    def __init__(self, train: Iterator, valid: Iterator, test: Iterator,
                 SRC: Field, TRG: Field):
        self.train = train
        self.valid = valid
        self.test = test
        self.SRC = SRC
        self.TRG = TRG

    def split_validation(self) -> Iterator:
        return self.valid

    def split_train(self) -> Iterator:
        return self.train

    def split_test(self) -> Iterator:
        return self.test
