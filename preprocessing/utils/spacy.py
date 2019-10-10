from typing import Iterable

from spacy.tokens import Token


def tokens_to_string(tokens: Iterable[Token]) -> str:
    # Glue tokens together, and expand contractions (e.g. "wouldn't" to "would not")
    return ' '.join([token.norm_ for token in tokens])
