from typing import List

from spacy.language import Language
from spacy.symbols import VERB, NOUN
from spacy.tokens import Token

from preprocessing import constants
from preprocessing.utils.nlp import tokens_to_string


def filter_message_pre(msg: str) -> bool:
    """
    Decides whether or not to keep this commit message.
    Run on raw commit message, without any preprocessing (just str.strip()).
    """

    # Filter on message length
    if len(msg) < constants.PREPROCESS_COMMIT_MSG_MIN_LEN:
        return False

    # Filter 'git revert' commits
    # Example: Revert "Added exclamation point to story one." (#94)
    if msg[:6] in ["Revert", "revert"]:
        return False

    # Filter 'git merge' commits
    if msg[:6] == "Merge:" or "Merge branch" in msg or "Merge pull request" in msg:
        return False

    return True


def filter_message_post(tokens: List[Token], nlp: Language) -> bool:
    """
    Decide whether or not to keep this commit message.
    Run on preprocessed commit message.
    """

    # Discard if processing the message was unsuccessful
    if not tokens:
        return False

    # Discard if too few or many tokens
    if not constants.PREPROCESS_COMMIT_MSG_MIN_TOKENS <= len(tokens) < constants.PREPROCESS_COMMIT_MSG_MAX_TOKENS:
        return False

    # Check if first word is a verb
    if tokens[0].pos == VERB:
        return True

    elif tokens[0].pos == NOUN:

        # Second check to handle incorrect PoS-tags for 'commit style' sentences
        # E.g. "Support setting this" --> "Support" is classified as NOUN instead of VERB
        # By prepending 'I ', the classification is correct
        check = nlp('I ' + tokens_to_string(tokens))

        return True if check[1].pos == VERB else False

    return False


def filter_diff_pre(diff: bytes) -> bool:
    """
    Decides whether or not to keep this diff.
    Run on raw diff, without any preprocessing.
    """

    # Filter on diff size
    if not constants.PREPROCESS_DIFF_MIN_BYTES <= len(diff) < constants.PREPROCESS_DIFF_MAX_BYTES:
        return False

    return True
