from typing import Iterable

from spacy.tokens import Token


def tokens_to_string(tokens: Iterable[Token]) -> str:
    # TODO: can use token.norm_ here to expand (e.g.) "n't" to "not", but this also lowercases all words
    return ' '.join(map(str, tokens))
