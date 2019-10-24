from typing import List

from spacy.language import Language
from spacy.tokens import Token

from preprocessing import constants
from preprocessing.utils.nlp import is_sha1, tokenize_diff

import nltk

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def parse_commit_message(msg: str, nlp: Language) -> List[Token]:
    """
    Processes (cleaned) commit message more thorougly:
        1. Extracts first sentence from message.
        2. Processes sentence with SpaCy (tokenize, PoS-tag)
        3. Removes any unwanted PoS-tagged tokens and trailing punctuation
    """

    # Split sentences with NLTK
    sents = sent_detector.tokenize(msg)
    if len(sents) <= 0:
        return []

    # Run full SpaCy pipeline on first sentence
    span = nlp(sents[0])
    tokens: List[Token] = []

    # Bail early if no first sentence
    if len(span) == 0:
        return tokens

    last_non_punct_idx = 0
    i = 0

    # Generate a list of tokens we want to keep
    for token in span:
        # Remove unwanted PoS-tags
        if token.pos in constants.PREPROCESS_IGNORE_POS:
            continue

        # Remove commit hashes
        if is_sha1(token.text):
            continue

        i += 1

        if token.text not in ['.', '..', '...', '?', '!', ';', ':', ',']:
            last_non_punct_idx = i

        tokens.append(token)

    # Bail if no valid tokens in message
    if len(tokens) == 0:
        return []

    # Remove trailing punctuation
    tokens = tokens[0:last_non_punct_idx]

    return tokens


def parse_diff(diff: str, meta: dict) -> (List[str], dict):
    """
    Tokenizes diff and filters diffs that are too large.
    """

    # Split diff into tokens
    tokens: List[str] = tokenize_diff(diff)

    meta['total_tokens'] = len(tokens) if tokens else 0

    # Filter on the number of tokens in the diff
    if len(tokens) >= constants.PREPROCESS_DIFF_MAX_TOKENS:
        tokens = []

    return tokens, meta
