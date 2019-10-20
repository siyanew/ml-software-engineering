import pathlib
import sys
import time
from multiprocessing import current_process
from typing import List
import orjson

import spacy
from spacy.language import Language
from spacy.tokens import Token

from preprocessing import constants
from preprocessing.utils import dataset
from preprocessing.utils.dataset import read_dataset, merge_output_files
from preprocessing.utils.filter import filter_message_pre, filter_message_post, filter_diff_pre
from preprocessing.utils.process import clean_commit_message, parse_commit_message, clean_diff, parse_diff
from preprocessing.utils.spacy import tokens_to_string, add_special_tokenizer_cases

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


def process_dataset(dataset: List[tuple]):
    # Return if dataset is empty
    if len(dataset) <= 0:
        return

    proc = current_process()
    proc_id = proc._identity[0]

    p = pathlib.Path(constants.DATA_DIR, constants.DATASET)

    # Load spacy
    print("Loading SpaCy...")
    using_gpu = spacy.prefer_gpu()
    nlp = spacy.load(constants.SPACY_LANGUAGE_MODEL, disable=["ner", "textcat"])
    nlp = add_special_tokenizer_cases(nlp)
    print("Using GPU: {}".format(using_gpu))

    # Open write handlers for result files
    fh_msg = p.joinpath(f'{constants.DATASET}.processed.msg.part{proc_id}').open('a', encoding=constants.OUTPUT_ENCODING, buffering=1)
    fh_diff = p.joinpath(f'{constants.DATASET}.processed.diff.part{proc_id}').open('a', encoding=constants.OUTPUT_ENCODING, buffering=1)
    fh_diff_meta = p.joinpath(f'{constants.DATASET}.diff.meta.jsonl.part{proc_id}').open('a', encoding=constants.OUTPUT_ENCODING, buffering=1)

    # Iterate over repositories (in commit messages folder)
    for repo, entry in dataset:

        msg_path = p.joinpath('msg', repo, f'{entry}.msg')
        diff_path = p.joinpath('diff', repo, f'{entry}.diff')

        try:

            # Process commit
            with msg_path.open('r', encoding=constants.OUTPUT_ENCODING, errors='ignore') as f:
                msg = process_commit(f.read(), nlp)

            # Bail early if message cannot be processed
            if not msg:
                continue

            # Process diff
            with diff_path.open('rb') as f:
                data = f.read(constants.PREPROCESS_DIFF_MAX_BYTES + 1024)  # Don't read more than necessary, +1KB so length check to discard diffs works correctly
                diff, meta = process_diff(data)

            # Bail if diff cannot be processed
            if not diff:
                continue

            if meta:  # Dump metadata if any (commit could be filtered by preliminary filter)
                meta['id'] = str(entry)
                meta['ext_count'] = dict(meta['ext_count'])  # Can't serialize defaultdict, so convert to regular dict
                fh_diff_meta.write(f'{str(orjson.dumps(meta), constants.OUTPUT_ENCODING)}\n')

            # Write results to file
            if constants.DEBUG:
                fh_msg.write(f'{entry}: {msg}\n')
                fh_diff.write(f'{entry}: {diff}\n')
            else:
                fh_msg.write(f'{msg}\n')
                fh_diff.write(f'{diff}\n')

        except FileNotFoundError:
            print(f'Cannot open diff or msg file for "{repo}/{entry}"')
            pass

    # Close result file handlers
    fh_msg.close()
    fh_diff.close()
    fh_diff_meta.close()


if __name__ == "__main__":
    """Preprocess commit + diff datasets."""

    print("Started preprocessing script. Calculating dataset size...")

    num_processes = mp.cpu_count()

    # Read dataset structure
    ds_path = pathlib.Path(constants.DATA_DIR, constants.DATASET)
    ds, num_ids = read_dataset(ds_path, num_partitions=num_processes)

    print(f'Processing dataset "{constants.DATASET}" with {num_ids} commits')

    # Check if results file is empty
    if not dataset.check_results_file(ds_path, force=constants.DEBUG):
        print("Exiting...")
        sys.exit(0)
    else:
        print("Continuing operation...\n")

    start_time = time.time()

    # Process dataset in parallel
    print(f'Firing up {num_processes} processes...')
    with mp.Pool(num_processes) as pool:
        pool.map_async(process_dataset, ds).get()

    # Merge output files
    merge_output_files(ds_path)

    print(f"\n\nPreprocessing finished in {time.time() - start_time:.1f} seconds!")
