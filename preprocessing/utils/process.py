import re
from typing import List

from spacy.language import Language
from spacy.tokens import Token

from preprocessing import constants

# Compiled regexes
re_git_id = re.compile(
    r'\s?(\((closing\s+)?\s?issue(s?)\s?|\(\s?)?#[0-9]+(\,\s?#[0-9]+)?\s?\)?')  # See: https://regex101.com/r/V1Scal/3
re_label_colon = re.compile(r'^\w* ?\:')  # Matches "{{Label:}} commit message"
re_mention = re.compile(
    r"((thanks\s?)(\s?to\s?)?)?\@[a-z0-9']+\s?[.,!]*")  # Matches user mentions combined with 'thanks'
re_url = re.compile(r"((git|ssh|http(s)?)|(git@[\w\.]+))(:(//)?)([\w\.@\:/\-~#'?,+]+)(\.git)?(/)?")  # Matches urls


def clean_commit_message(msg: str) -> str:
    """
    Prepares commit message for the preliminary filter:
        - Removes whitespace.
        - Strips identifiers (#59)
        - Strips labels ([label], label:)  TODO: maybe don't strip 'Added: this and that' constructions (so with verb)
    """

    msg = msg.strip()

    # Discard everything after the first line
    msg = msg.partition('\n')[0]

    # Remove commit / PR IDs
    msg = re_git_id.sub('', msg)

    # Remove square bracket label at the start of the message
    if msg[0] == '[':
        msg = msg.partition(']')[-1]

    # Remove label with colon (at the start of the message)
    msg = re_label_colon.sub('', msg)

    # Remove thanks messages
    msg = re_mention.sub('', msg)

    # Remove URLs
    msg = re_url.sub('', msg)

    # Remove any remaining starting and trailing whitespace
    msg = msg.strip()

    return msg


def parse_commit_message(msg: str, nlp: Language) -> List[Token]:
    """
    Processes (cleaned) commit message more thorougly:
        - Extract dependencies / word functions from message
    """

    # Run full SpaCy pipeline on message
    doc = nlp(msg)

    # TODO: add special tokens for language constructs (e.g. ==, +=, etc.)

    try:

        # Get first sentence
        span = next(doc.sents)
        tokens: List[Token] = []

        # Bail early if no first sentence
        if len(span) == 0:
            return tokens

        # Generate a list of tokens we want to keep
        for token in span:
            if token.pos in constants.PREPROCESS_IGNORE_POS:
                continue

            tokens.append(token)

        # Bail if no valid tokens in message
        if len(tokens) == 0:
            return []

        # Remove trailing punctuation
        if tokens[-1].string in '.?!;:':
            tokens = tokens[0:-1]

        return tokens
    except StopIteration:
        return None
