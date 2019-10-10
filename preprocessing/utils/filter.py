from typing import List

from spacy.language import Language
from spacy.symbols import VERB, NOUN
from spacy.tokens import Token

from preprocessing import constants
from preprocessing.utils.spacy import tokens_to_string


def filter_message_pre(msg: str) -> bool:
    """
    Decides whether or not to keep this commit message.
    Run on raw commit message, without any preprocessing.
    """

    # Filter on message length
    if len(msg) < constants.PREPROCESS_COMMIT_MSG_MIN_LEN:
        return False

    # Check for 'git revert' commits
    # Pattern: Revert "(.*)"
    # Examples:
    #   Revert "Added exclamation point to story one." (#94)
    #   Revert "Revert "Revert "add LimitMempoolSize logic for mempool"""
    #   Revert "Revert "Initial commit""
    if msg[0] in ['R', 'r'] and msg.startswith((
            "Revert \"",
            "revert \"",
    )):
        return False

    # Check for 'git merge' commits
    # Pattern: Merge branch '(.*)'[(.*)]
    # Pattern: Merge pull request #(.*)
    # Examples:
    #   Merge branch 'master' into devel
    #   Merge branch 'docs'
    #   Merge pull request #1998 from rossabaker/merge-0.18.16
    #   Merge pull request #1 from infamousmammal/readme-edits
    #   Merge branch 'master' of https://github.com/phorcys/Taiwu_mods
    # N.B.: Commit msg does not necessarily have to start with these words! Some repos prepend some IDs.
    if msg.startswith("Merge:"):
        return False

    if msg.find("Merge branch '") >= 0 or msg.find("Merge pull request #") >= 0:
        return False

    # TODO: Filter non-English commit messages

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
        # TODO: find and check edge cases
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
