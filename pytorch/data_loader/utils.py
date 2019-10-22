import os
from pathlib import Path
from typing import List, Tuple

import dill
from torchtext.data import Dataset, Field

SRC_FILE = 'src'
TRG_FILE = 'trg'


def get_vocab_paths(data_dir: str, sizes: Tuple[int, int], freqs: Tuple[int, int]) -> (Path, Path):
    src = get_vocab_path(data_dir, SRC_FILE, sizes[0], freqs[0])
    trg = get_vocab_path(data_dir, TRG_FILE, sizes[1], freqs[1])
    return src, trg


def get_vocab_path(data_dir: str, name: str, size: int, freq: int) -> str:
    save_path = Path(data_dir) / 'processed'
    file = f"{name}_{size}_{freq}.vocab"
    return save_path / file


def has_vocabs(data_dir: str, sizes: Tuple[int, int], freqs: Tuple[int, int]) -> bool:
    src_path, trg_path = get_vocab_paths(data_dir, sizes, freqs)

    return os.path.exists(src_path) and os.path.exists(trg_path)


def load_vocabs(data_dir: str, sizes: Tuple[int, int], freqs: Tuple[int, int]) -> (Field, Field):
    src_path, trg_path = get_vocab_paths(data_dir, sizes, freqs)
    print(f"Loading source vocabs from ${src_path}...")
    print(f"Loading target vocabs from ${trg_path}...")

    with open(src_path, 'rb') as f:
        SRC = dill.load(f)

    with open(trg_path, 'rb') as f:
        TRG = dill.load(f)

    return SRC, TRG


def save_vocabs(data_dir: str, SRC: Field, TRG: Field, sizes: Tuple[int, int], freqs: Tuple[int, int]):
    save_path = Path(data_dir) / 'processed'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    src_path, trg_path = get_vocab_paths(data_dir, sizes, freqs)
    print(f"Writing preprocessed vocabs to ${src_path}...")
    with open(src_path, 'wb+') as f:
        dill.dump(SRC, f)

    print(f"Writing preprocessed vocabs to ${trg_path}...")
    with open(trg_path, 'wb+') as f:
        dill.dump(TRG, f)


def tokenize_diff(text: str) -> List[str]:
    return text.split(" ")


def tokenize_msg(text: str) -> List[str]:
    return text.split(" ")


def save_dataset(data_dir: str, dataset: Dataset, fname="test"):
    save_path = Path(data_dir)
    with open(save_path / f"{fname}.diff", 'w+') as src_file, \
            open(save_path / f"{fname}.msg", 'w+') as trg_file:
        for example in dataset.examples:
            print(" ".join(example.src), file=src_file)
            print(" ".join(example.trg), file=trg_file)
