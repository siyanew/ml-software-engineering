import pathlib
import time
from typing import List

import spacy
from spacy.language import Language
from spacy.tokens import Token

from preprocessing import constants
from preprocessing.utils.dataset import load_dataset
from preprocessing.utils.filter import filter_message_pre, filter_message_post
from preprocessing.utils.process import clean_commit_message, parse_commit_message
from preprocessing.utils.spacy import tokens_to_string


def process_commit(msg: str, nlp: Language):
    # Clean commit message (remove whitespace, labels, ids, etc.)
    msg = clean_commit_message(msg)

    # Preliminary filter (discard rollback/merge commits, etc.)
    if not filter_message_pre(msg):
        return None

    # Process commit message (split first sentence, NLP, etc.)
    tokens: List[Token] = parse_commit_message(msg, nlp)

    # Final filter (V-DO filter, etc.)
    if not filter_message_post(tokens, nlp):
        return None

    return tokens_to_string(tokens)


def process_diff(diff: str):
    return diff[:10]


if __name__ == "__main__":
    """Preprocess commit + diff datasets."""

    # Read dataset structure
    p = pathlib.Path(constants.DATA_DIR, constants.DATASET)
    dataset, num_ids = load_dataset(p)

    # TODO: partition dataset and multiprocess

    print(f'Now processing dataset "{constants.DATASET}" containing {num_ids} commits.')

    # Load spacy
    print("Loading SpaCy...")
    nlp = spacy.load(constants.PREPROCESS_SPACY_LANGUAGE_MODEL, disable=["ner", "textcat"])

    # Open write handlers for result files
    msg_results = p.joinpath(constants.DATASET + '.processed.msg')
    diff_results = p.joinpath(constants.DATASET + '.processed.diff')
    fh_msg = msg_results.open('a', encoding=constants.DATASET_ENCODING)
    fh_diff = diff_results.open('a', encoding=constants.DATASET_ENCODING)

    # Iterate over repositories (in commit messages folder)
    for repo, ids in dataset.items():

        print(f"Processing repo: {repo}")
        repo_time_start = time.time()

        msg_path = p.joinpath('msg', repo)
        diff_path = p.joinpath('diff', repo)

        # Process commits
        for entryId in ids:
            print(f"   Processing entry: {entryId}", end='\r')

            # Process commit
            with msg_path.joinpath(f'{entryId}.msg').open('r', encoding=constants.DATASET_ENCODING) as f:
                msg = process_commit(f.read(), nlp)

            # Bail early if message cannot be processed
            if not msg:
                continue

            # Process diff
            with diff_path.joinpath(f'{entryId}.diff').open('rb') as f:

                try:
                    data = f.read()
                    diff = process_diff(data)
                except UnicodeDecodeError:
                    print(f)
                    raise

            # Bail if diff cannot be processed
            if not diff:
                continue

            # Write results to file
            if msg and diff:

                if constants.DEBUG:
                    fh_msg.write(f'{entryId}: {msg}\n')
                    fh_diff.write(f'{entryId}: {diff}\n')
                else:
                    fh_msg.write(f'{msg}\n')
                    fh_diff.write(f'{diff}\n')

        print(f'...done in {time.time() - repo_time_start:.1f}s!')

    # Close results file handler
    fh_msg.close()

    print("\n\nPreprocessing finished.")
