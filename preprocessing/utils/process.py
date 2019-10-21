import collections
import pathlib
import re
from typing import List

from spacy.language import Language
from spacy.tokens import Token

from preprocessing import constants
from preprocessing.utils.converter import Converter
from preprocessing.utils.spacy import is_sha1

import nltk

# Compiled regexes
from preprocessing.utils.spacy import tokenize_diff

re_git_id = re.compile(
    r'\s?(\((closing\s+)?\s?issue(s?)\s?|\(\s?)?#[0-9]+(,\s?#[0-9]+)?\s?\)?')  # See: https://regex101.com/r/V1Scal/3
re_label_colon = re.compile(r'^\w* ?:')  # Matches "{{Label:}} commit message"
re_mention = re.compile(
    r"((thanks\s?)(\s?to\s?)?)?@[a-z0-9']+\s?[.,!]*")  # Matches user mentions combined with 'thanks'
re_url = re.compile(r"((git|ssh|http(s)?)|(git@[\w.]+))(:(//)?)([\w.@:/\-~#'?,+]+)(\.git)?(/)?")  # Matches urls
re_no_english = re.compile(r'[^\sa-zA-Z0-9.!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]')  # From ptrgn-commit-msg project
re_version_number = re.compile(r'(\d+)\.(\d+)(?:\.(\d+))?(?:-(\w+)|-)?')

conv: Converter = Converter()

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def clean_commit_message(msg: str) -> str:
    """
    Prepares commit message for the preliminary filter:
        - Removes whitespace.
        - Strips identifiers (#59)
        - Strips labels ([label], label:)  TODO: maybe don't strip 'Added: this and that' constructions (so with verb)
    """

    # Discard everything after the first line
    msg = msg.partition('\n')[0]

    # Remove commit / PR IDs
    msg = re_git_id.sub('', msg)

    # Remove square bracket label at the start of the message
    if len(msg) > 0 and msg[0] == '[':
        msg = msg.partition(']')[-1]

    # Remove label with colon (at the start of the message)
    # DISABLED: too many side effects
    # msg = re_label_colon.sub('', msg)

    # Remove thanks messages
    msg = re_mention.sub('', msg)

    # Remove URLs
    msg = re_url.sub('', msg)

    # Remove "non-English" characters
    msg = re_no_english.sub('', msg)

    # Split sub-tokens
    msg = conv.splitSubTokens(msg)

    # Remove any remaining starting and trailing whitespace
    msg = msg.strip()

    return msg


def parse_commit_message(msg: str, nlp: Language) -> List[Token]:
    """
    Processes (cleaned) commit message more thorougly:
        - Extract dependencies / word functions from message
    """

    # Replace version numbers
    msg = re_version_number.sub(constants.PREPROCESS_DIFF_TOKEN_VERSION, msg)

    # Split sentences with NLTK
    sents = sent_detector.tokenize(msg)
    if len(sents) <= 0:
        return None

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


def _reduce_diff(lines: List[str]) -> (str, dict):
    result = []

    # Processing flags
    ignore_file = False
    in_block = False
    block_lines = []

    # Metadata
    meta = {
        'total_lines': len(lines),
        'lines_kept': 0,
        'additions': 0,
        'deletions': 0,
        'file_count': 0,
        'block_count': 0,
        'ext_count': collections.defaultdict(int)
    }

    for line in lines:

        # New file marker
        if line[0:11] == 'diff --git ':

            meta['file_count'] += 1

            if block_lines:  # Commit changes in last file
                result.append(block_lines)
                meta['lines_kept'] += len(block_lines)

                # Reset
                block_lines = []

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

                meta['ext_count'][filetype] += 1

                # Remove "non-English" characters and split sub-tokens
                filename = conv.splitSubTokens(re_no_english.sub('', filename))

                # Retain just the filename for valid files
                block_lines.append(f'{constants.PREPROCESS_DIFF_TOKEN_FILE} {filename.lower()}')

        elif not ignore_file:

            # Skip empty lines
            if not line:
                continue

            if in_block:

                # Process changes/additions
                if line[0] in '+-':

                    change = line[1:].strip()

                    # Remove "non-English" characters and split sub-tokens
                    change = conv.splitSubTokens(re_no_english.sub('', change))

                    if change:  # Keep if not empty
                        if line[0] == '-':
                            meta['deletions'] += 1
                            block_lines.append(f'{constants.PREPROCESS_DIFF_TOKEN_DEL} {change.lower()}')
                        elif line[0] == '+':
                            meta['additions'] += 1
                            block_lines.append(f'{constants.PREPROCESS_DIFF_TOKEN_ADD} {change.lower()}')

            else:

                # Process block starts
                if line[0:2] == '@@':
                    in_block = True

                    meta['block_count'] += 1

                    # Get context of block (if any)
                    context = line[2:].partition('@@')[-1]

                    # Remove whitespace and trailing accolades / commas
                    context = context.strip().rstrip('{,')

                    # Remove "non-English" characters and split sub-tokens
                    context = conv.splitSubTokens(re_no_english.sub('', context))

                    if context:  # Keep if not empty
                        block_lines.append(context.lower())

    # Commit changes in last file
    if block_lines:
        result.append(block_lines)
        meta['lines_kept'] += len(block_lines)

    # Order files by most changes
    result.sort(key=len, reverse=True)

    # Flatten list (fast and readable, https://stackoverflow.com/a/408281)
    result = [line for block in result for line in block]

    # Don't return anything for diffs that are empty after processing
    result_str = ' '.join(result) if len(result) > 0 else None

    return result_str, meta


def clean_diff(diff_raw: bytes) -> (str, dict):
    # Decode byte stream and split lines
    lines = diff_raw.decode(errors='ignore').split('\n')

    # Process and filter lines in diff
    return _reduce_diff(lines)


def parse_diff(diff: str, meta: dict) -> (List[str], dict):
    # Split diff into tokens
    tokens: List[str] = tokenize_diff(diff)

    meta['total_tokens'] = len(tokens) if tokens else 0

    # Limit the number of tokens in diff
    tokens = tokens[:constants.PREPROCESS_DIFF_NUM_TOKEN_CUTOFF]

    return tokens, meta
