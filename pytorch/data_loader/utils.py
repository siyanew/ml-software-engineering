import os
from pathlib import Path

import dill
from torchtext.data import Field

src_file = 'src_vocab'
trg_file = 'trg_vocab'


def has_vocabs(data_dir: str) -> bool:
    save_path = Path(data_dir) / 'processed'

    src_path = save_path / src_file
    trg_path = save_path / trg_file

    return os.path.exists(src_path) and os.path.exists(trg_path)


def load_vocabs(data_dir: str) -> (Field, Field):
    save_path = Path(data_dir) / 'processed'
    print(f"Loading preprocessed vocabs from ${save_path}...")

    src_path = save_path / src_file
    trg_path = save_path / trg_file

    with open(src_path, 'rb') as f:
        SRC = dill.load(f)

    with open(trg_path, 'rb') as f:
        TRG = dill.load(f)

    return SRC, TRG


def save_vocabs(data_dir: str, SRC: Field, TRG: Field):
    save_path = Path(data_dir) / 'processed'
    print(f"Writing preprocessed vocabs to ${save_path}...")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path / src_file, 'wb+') as f:
        dill.dump(SRC, f)

    with open(save_path / trg_file, 'wb+') as f:
        dill.dump(TRG, f)
