import pathlib
import sys
import time
from typing import List
import orjson

import spacy
from spacy.language import Language
from spacy.tokens import Token

from preprocessing import constants
from preprocessing.utils.dataset import read_dataset
from preprocessing.utils.filter import filter_message_pre, filter_message_post, filter_diff_pre
from preprocessing.utils.process import clean_commit_message, parse_commit_message, clean_diff, parse_diff
from preprocessing.utils.spacy import tokens_to_string, _add_special_tokenizer_cases

import multiprocessing as mp


def process_commit(msg: str, nlp: Language):
    msg = msg.strip()

    # Preliminary filter (discard rollback/merge commits, etc.)
    if not filter_message_pre(msg):
        return None

    # Clean commit message (remove whitespace, labels, ids, etc.)
    msg = clean_commit_message(msg)

    # Process commit message (split first sentence, NLP, etc.)
    tokens: List[Token] = parse_commit_message(msg, nlp)

    # Final filter (V-DO filter, etc.)
    if not filter_message_post(tokens, nlp):
        return None

    return tokens_to_string(tokens)


def process_diff(raw_diff: bytes):
    # Preliminary diff filter
    if not filter_diff_pre(raw_diff):
        return None, None

    diff, meta = clean_diff(raw_diff)

    if not diff:
        return None, meta

    # Parse diff / tokenize (and cutoff at specified point)
    tokens, meta = parse_diff(diff, meta)

    if not tokens:
        return None, meta

    # Glue tokens back together
    diff_str = ' '.join(tokens)

    return diff_str, meta


def process_dataset(dataset: dict):
    # Return if dataset is empty
    if not dataset:
        return

    p = pathlib.Path(constants.DATA_DIR, constants.DATASET)

    # Load spacy
    print("Loading SpaCy...")
    # using_gpu = spacy.prefer_gpu()
    using_gpu = False
    nlp = spacy.load(constants.SPACY_LANGUAGE_MODEL, disable=["ner", "textcat"])
    _add_special_tokenizer_cases(nlp)
    print("Using GPU: {}".format(using_gpu))

    # Open write handlers for result files
    fh_msg = p.joinpath(constants.DATASET + '.processed.msg').open('a', encoding=constants.OUTPUT_ENCODING)
    fh_diff = p.joinpath(constants.DATASET + '.processed.diff').open('a', encoding=constants.OUTPUT_ENCODING)
    fh_diff_meta = p.joinpath(constants.DATASET + '.diff.meta.jsonl').open('a', encoding=constants.OUTPUT_ENCODING)

    # Iterate over repositories (in commit messages folder)
    for repo, ids in dataset.items():

        print(f'Start processing repo "{repo}"')
        repo_time_start = time.time()

        msg_path = p.joinpath('msg', repo)
        diff_path = p.joinpath('diff', repo)

        for entry in ids:

            # Process commit
            with msg_path.joinpath(f'{entry}.msg').open('r', encoding=constants.OUTPUT_ENCODING) as f:
                msg = process_commit(f.read(), nlp)

            # Bail early if message cannot be processed
            if not msg:
                continue

            # Process diff
            with diff_path.joinpath(f'{entry}.diff').open('rb') as f:

                try:
                    data = f.read()
                    diff, meta = process_diff(data)

                except UnicodeDecodeError:
                    print(f)  # TODO: investigate, for now fixed by reading raw bytes ('rb' flags)
                    raise

            # Bail if diff cannot be processed
            if not diff:
                continue

            if meta:  # No metadata if commit filtered by preliminary filter
                # Dump metadata
                meta['ext_count'] = dict(meta['ext_count'])  # Can't serialize defaultdict
                fh_diff_meta.write(f'{str(orjson.dumps(meta), constants.OUTPUT_ENCODING)}\n')

            # Write results to file
            if constants.DEBUG:
                fh_msg.write(f'{entry}: {msg}\n')
                fh_diff.write(f'{entry}: {diff}\n')
            else:
                fh_msg.write(f'{msg}\n')
                fh_diff.write(f'{diff}\n')

        print(f'Finished processing "{repo}" in {time.time() - repo_time_start:.1f}s ({len(ids)} commits)')

    # Close result file handlers
    fh_msg.close()
    fh_diff.close()
    fh_diff_meta.close()


def _check_results_file(p: pathlib.Path, force=False) -> bool:
    msg_results = p.joinpath(constants.DATASET + '.processed.msg')
    diff_results = p.joinpath(constants.DATASET + '.processed.diff')
    diff_results_meta = p.joinpath(constants.DATASET + '.diff.meta.jsonl')

    if (msg_results.exists() and msg_results.stat().st_size > 0) \
            or (diff_results.exists() and diff_results.stat().st_size > 0) \
            or (diff_results_meta.exists() and diff_results_meta.stat().st_size > 0):

        if force:
            msg_results.unlink()
            diff_results.unlink()
            diff_results_meta.unlink()
            return True

        print("\nOne or more result files exist and are not empty.")
        choice = input("Clear these files and continue? [y/N] ")

        if choice.lower() == 'y':
            msg_results.unlink()
            diff_results.unlink()
            diff_results_meta.unlink()
            print("\nFiles removed.")
            return True
        else:
            print("\nNo changes have been made.")
            return False

    # Nothing on the hand
    return True


if __name__ == "__main__":
    """Preprocess commit + diff datasets."""

    print("Started preprocessing script. Calculating dataset size...")

    num_processes = mp.cpu_count() if not constants.DEBUG else 1

    # Read dataset structure
    ds_path = pathlib.Path(constants.DATA_DIR, constants.DATASET)
    ds, num_ids = read_dataset(ds_path, num_partitions=num_processes)

    print(f'Processing dataset "{constants.DATASET}" with {num_ids} commits')
    print(f'Firing up {num_processes} processes...')

    # Check if results file is empty
    if not _check_results_file(ds_path, force=constants.DEBUG):
        print("Exiting...")
        sys.exit(0)
    else:
        print("Continuing operation...\n")

    start_time = time.time()

    # Process dataset in parallel
    with mp.Pool(num_processes) as pool:
        pool.map(process_dataset, ds)

    print(f"\n\nPreprocessing finished in {time.time() - start_time:.1f} seconds!")
