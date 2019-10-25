import collections
import pathlib
import re

from preprocessing import constants
from preprocessing.utils.converter import Converter

re_git_id = re.compile(r'\s?(\((closing\s+)?\s?issue(s?)\s?|\(\s?)?#[0-9]+(,\s?#[0-9]+)?\s?\)?')  # See: https://regex101.com/r/V1Scal/3
re_mention = re.compile(r"((thanks\s?)(\s?to\s?)?)?@[a-z0-9']+\s?[.,!]*")  # Matches user mentions (optionally combined with 'thanks')
re_url = re.compile(r"((git|ssh|http(s)?)|(git@[\w.]+))(:(//)?)([\w.@:/\-~#'?,+]+)(\.git)?(/)?")  # Matches urls
re_no_english = re.compile(r'[^\sa-zA-Z0-9.!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]')  # From ptrgn-commit-msg project
re_version_number = re.compile(r'(\d+)\.(\d+)(?:\.(\d+))?(?:-(\w+)|-)?')

conv: Converter = Converter()


def clean_commit_message(msg: str) -> str:
    """
    Cleans commits message by:
        1. Discarding everything after the first line
        2. Removing GitHub issue IDs
        3. Removing [...] labels at the beginning of messages
        4. Removing @mentions
        5. Removing URLs
        6. Replacing version numbers with a generic identifier
        7. Splitting sub-tokens (camelCase -> camel case)
        8. Removing any remaining whitespace.

    N.B.: Pass a string without leading whitespace.
    """

    # Discard everything after the first line
    msg = msg.partition('\n')[0]

    # Remove commit / PR IDs
    msg = re_git_id.sub('', msg)

    # Remove square bracket label at the start of the message
    if len(msg) > 0 and msg[0] == '[':
        msg = msg.partition(']')[-1]

    # Remove thanks messages
    msg = re_mention.sub('', msg)

    # Remove URLs
    msg = re_url.sub('', msg)

    # Remove "non-English" characters
    msg = re_no_english.sub('', msg)

    # Replace version numbers
    msg = re_version_number.sub(constants.PREPROCESS_DIFF_TOKEN_VERSION, msg)

    # Split sub-tokens
    msg = conv.splitSubTokens(msg)

    # Remove any remaining starting and trailing whitespace
    msg = msg.strip()

    return msg


def clean_diff(diff_raw: bytes) -> (str, dict):
    """
    Cleans diffs by:
        1. Keeping only the changed filename from blocks in the diff
        2. Keeping only lines with changes (discarding same lines)
        3. Removing all files with extensions not pre-approved
        4. Keeping only the context part of the block metadata (and not the location in the file)
        5. Splitting sub-tokens (camelCase -> camel case)

    Finally, all changes are ordered by the files with the most changes.
    All tokens are glued together with a single space (' ') to produce the output string.
    During this process, statistics about how many lines kept etc. are collected to be analyzed after preprocessing.
    """
    result = []

    # Decode byte stream and split lines
    lines = diff_raw.decode(errors='ignore').split('\n')

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

    # Processing flags
    ignore_file = False
    in_block = False
    block_lines = []

    for line in lines:

        # New file marker
        if line[0:11] == 'diff --git ':

            meta['file_count'] += 1

            if block_lines:  # Commit changes of previous file (if any)
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
