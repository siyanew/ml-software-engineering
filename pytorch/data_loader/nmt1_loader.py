from typing import List, Tuple

import torch
from torchtext.data import BucketIterator, Field
from torchtext.datasets import TranslationDataset

from base import BaseTextIterator
from data_loader.utils import has_vocabs, load_vocabs, save_vocabs


def tokenize_diff(text: str) -> List[str]:
    return text.split(" ")


def tokenize_msg(text: str) -> List[str]:
    return text.split(" ")


class NMT1Loader(BaseTextIterator):
    def __init__(self, data_dir: str, packed: bool,
                 vocab_max_sizes: Tuple[int, int], vocab_min_freqs: Tuple[int, int],
                 batch_sizes: Tuple[int, int, int], test: bool = False):
        print(f"Creating DataLoader for {'testing' if test else 'training'}")

        # Rebuild the vocabs during testin, as the saved can be build from a different config
        if test:
            vocab_exists = False
        else:
            vocab_exists = has_vocabs(data_dir, vocab_max_sizes, vocab_min_freqs)

        # Define torch text fields for processing text
        if vocab_exists:
            print("Loading fields and vocabs...")
            SRC, TRG = load_vocabs(data_dir, vocab_max_sizes, vocab_min_freqs)
        else:
            print("Building fields...")

            # Include the sentence length for source
            SRC = Field(tokenize=tokenize_diff,
                        init_token='<sos>',
                        eos_token='<eos>',
                        include_lengths=packed,
                        lower=True)

            TRG = Field(tokenize=tokenize_msg,
                        init_token='<sos>',
                        eos_token='<eos>',
                        lower=True)

        print("Loading commit data...")
        train_data, valid_data, test_data = TranslationDataset.splits(
            exts=('.diff', '.msg'),
            train='TrainingSet/train.26208',
            validation='TrainingSet/valid.3000',
            test='TestSet/test.3000',
            fields=(SRC, TRG),
            path=data_dir)

        if not vocab_exists:
            # Build vocabs
            print("Building vocabulary...")
            specials = ['<unk>', '<pad>', '<sos>', '<eos>']
            SRC.build_vocab(train_data, min_freq=vocab_min_freqs[0], max_size=vocab_max_sizes[0], specials=specials)
            TRG.build_vocab(train_data, min_freq=vocab_min_freqs[1], max_size=vocab_max_sizes[1], specials=specials)

            if not test:
                save_vocabs(data_dir, SRC, TRG, vocab_max_sizes, vocab_min_freqs)

        print(f"Number of training examples: {len(train_data.examples)}")
        print(f"Number of validation examples: {len(valid_data.examples)}")
        print(f"Number of testing examples: {len(test_data.examples)}")
        print(f"Unique tokens in source (diff) training vocabulary: {len(SRC.vocab)}")
        print(f"Unique tokens in target (msg) training vocabulary: {len(TRG.vocab)}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Bucketing (minimizes the amount of padding by grouping similar length sentences)
        # Sort the sequences based on their non-padded length
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_sizes=batch_sizes,
            sort_within_batch=packed,
            sort_key=lambda x: len(x.src) if packed else None,
            device=device)

        super().__init__(train_iterator, valid_iterator, test_iterator,
                         SRC, TRG, tokenize_diff, tokenize_msg)
