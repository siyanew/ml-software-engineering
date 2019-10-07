import spacy
from base import BaseDataLoader
from torchtext.data import BucketIterator, Field
from torchtext.datasets import Multi30k
from torchvision import datasets as vdatasets, transforms


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = vdatasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class LanguageDataLoader:
    def __init__(self, data_dir, batch_sizes):
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('de')

        def tokenize_de(text):
            """
            Tokenizes German text from a string into a list of strings (tokens) and reverses it
            """
            return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

        def tokenize_en(text):
            """
            Tokenizes English text from a string into a list of strings (tokens)
            """
            return [tok.text for tok in spacy_en.tokenizer(text)]

        SRC = Field(tokenize=tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)

        TRG = Field(tokenize=tokenize_en,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)

        train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                            fields=(SRC, TRG),
                                                            root=data_dir
                                                            )
        print(f"Number of training examples: {len(train_data.examples)}")
        print(f"Number of validation examples: {len(valid_data.examples)}")
        print(f"Number of testing examples: {len(test_data.examples)}")
        SRC.build_vocab(train_data, min_freq=2)
        TRG.build_vocab(train_data, min_freq=2)
        print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
        print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

        # padding all the sentences to same length, replacing words by its index,
        # bucketing (minimizes the amount of padding by grouping similar length sentences)
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                              batch_size=batch_sizes)
        self.valid_iterator = valid_iterator
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator

    def split_validation(self):
        return self.valid_iterator

    def split_train(self):
        return self.train_iterator

    def split_test(self):
        return self.test_iterator
