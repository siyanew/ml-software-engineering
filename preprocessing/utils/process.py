import pathlib
import re
from typing import List

from spacy.language import Language
from spacy.tokens import Token

from preprocessing import constants

# Compiled regexes
from preprocessing.utils.spacy import tokenize_diff

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


def _diff_line_filter_reduce(lines: List[str]) -> str:
    result = []

    # Processing flags
    ignore_file = False
    in_block = False

    for line in lines:

        if line[0:11] == 'diff --git ':

            # Reset
            in_block = False
            ignore_file = False

            # Parse line for file extension
            path = pathlib.PurePosixPath(line[11:].partition(' ')[0])

            filename = path.name
            filetype = path.suffix[1:]

            if filetype not in constants.PREPROCESS_DIFF_KEEP_EXTENSIONS:

                # Ignore unsupported files
                ignore_file = True

            else:

                # Retain just the filename for valid files
                result.append(filename.lower())

        elif not ignore_file:

            # Skip empty lines
            if not line:
                continue

            if in_block:

                # Process changes/additions
                if line[0] in '+-':

                    change = line[1:].strip()

                    # Discard empty lines
                    if change:
                        if line[0] == '-':
                            result.append(f'{constants.PREPROCESS_DIFF_TOKEN_DEL} {change.lower()}')
                        elif line[0] == '+':
                            result.append(f'{constants.PREPROCESS_DIFF_TOKEN_ADD} {change.lower()}')

            else:

                # Process block starts
                if line[0:2] == '@@':
                    in_block = True

                    # Get context of block (if any)
                    context = line[2:].partition('@@')[-1]

                    # Remove whitespace and trailing accolades / commas
                    context = context.strip().rstrip('{,')

                    if context:  # Keep if not empty
                        result.append(context.lower())

    # Generate output string
    return ' '.join(result)


def clean_diff(diff_raw: bytes) -> str:
    try:

        # Decode byte stream
        lines = diff_raw.decode().split('\n')

        # Process and fiter lines in diff
        return _diff_line_filter_reduce(lines)

    except UnicodeDecodeError as e:
        print('\n')
        print(diff_raw)
        print('\n')
        raise e

    return None


def parse_diff(diff: str) -> List[str]:
    # Split diff into tokens
    tokens: List[str] = tokenize_diff(diff)

    # Limit the number of tokens in diff
    tokens = tokens[:constants.PREPROCESS_DIFF_NUM_TOKEN_CUTOFF]

    return tokens
