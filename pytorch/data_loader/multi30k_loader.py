from typing import List, Tuple

import spacy
import torch
from torchtext.data import BucketIterator, Field
from torchtext.datasets import Multi30k

from base import BaseTextIterator

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text: str) -> List[str]:
    """
    Tokenizes German text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_de_reverse(text: str) -> List[str]:
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text: str) -> List[str]:
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


class LanguageDataLoader(BaseTextIterator):
    def __init__(self, data_dir: str, reverse_src: bool, packed: bool, batch_sizes: Tuple[int, int, int]):

        # Define torch text fields for processing text
        # Include the sentence length for source
        if reverse_src:
            SRC = Field(tokenize=tokenize_de_reverse,
                        init_token='<sos>',
                        eos_token='<eos>',
                        include_lengths=packed,
                        lower=True)
        else:
            SRC = Field(tokenize=tokenize_de,
                        init_token='<sos>',
                        eos_token='<eos>',
                        include_lengths=packed,
                        lower=True)

        TRG = Field(tokenize=tokenize_en,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)

        # Load the Multi30k sentence dataset in de and en
        train_data, valid_data, test_data = Multi30k.splits(
            exts=('.de', '.en'),
            fields=(SRC, TRG),
            root=data_dir)

        # Build vocabs
        SRC.build_vocab(train_data, min_freq=2)
        TRG.build_vocab(train_data, min_freq=2)

        print(f"Number of training examples: {len(train_data.examples)}")
        print(f"Number of validation examples: {len(valid_data.examples)}")
        print(f"Number of testing examples: {len(test_data.examples)}")
        print(f"Unique tokens in source (de) training vocabulary: {len(SRC.vocab)}")
        print(f"Unique tokens in target (en) training vocabulary: {len(TRG.vocab)}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Bucketing (minimizes the amount of padding by grouping similar length sentences)
        # Sort the sequences based on their non-padded length
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_sizes=batch_sizes,
            sort_within_batch=packed,
            sort_key=lambda x: len(x.src) if packed else None,
            device=device)

        super().__init__(train_iterator, valid_iterator, test_iterator, SRC, TRG)
