import pathlib
import numpy as np
from typing import List

from preprocessing.constants import DATASET


def read_dataset(path_to_dataset: pathlib.Path, num_partitions=1) -> (List[tuple], int):
    """
    Parses the dataset file tree and constructs work map.
    Returns a specified number of lists with (repo,ID) tuples.

    Uses the message folder tree to get the IDs. Does not check for missing diffs.

    Required folder structure:
        dataset-name
            - msg
                - repo0
                    - 1.msg
                    - ....
                    - n.msg
                - ...
                - repoN
            - diff
                - repo0
                    - 1.diff
                    - ....
                    - n.diff
                - ...
                - repoN

    TODO: implement mechanism to skip already processed repos and commits
    """

    # Init return values
    data: List[tuple] = list()

    # Read repos from msg dir
    for repo in path_to_dataset.joinpath('msg').iterdir():

        repo_id = repo.name
        msg_path = path_to_dataset.joinpath('msg', repo_id)

        # Read commits, extract ID and store in data list
        for msg in msg_path.glob("*.msg"):
            data.append((repo_id, msg.stem))

    total_commits: int = len(data)

    # Partition data and return
    return chunker_list(data, num_partitions), total_commits


def read_dataset_thoroughly(path_to_dataset: pathlib.Path, num_partitions=1) -> (List[dict], int):
    """
    Parses the dataset file tree and constructs work map.
    Ignores (commit,diff) pairs with any missing files.

    Returns dictionary with {repo1: [id1 .. idN]} and total number of IDs in dataset.

    Required folder structure:
        dataset-name
            - msg
                - repo0
                    - 1.msg
                    - ....
                    - n.msg
                - ...
                - repoN
            - diff
                - repo0
                    - 1.diff
                    - ....
                    - n.diff
                - ...
                - repoN

    TODO: implement mechanism to skip already processed repos and commits
    TODO: current implementation very slow on large datasets
    """

    # Init return values
    partitions: List[int] = [0] * num_partitions
    data: List[dict] = [dict() for i in range(num_partitions)]
    total_commits: int = 0

    # Read repos from msg dir
    for folder in path_to_dataset.joinpath('msg').iterdir():

        # Skip files
        if not folder.is_dir(): continue

        ids = []

        repo = folder.name

        msg_path = path_to_dataset.joinpath('msg', repo)
        diff_path = path_to_dataset.joinpath('diff', repo)

        # Read commits, extract ID
        for msg in msg_path.glob("*.msg"):
            ID = msg.stem

            if diff_path.joinpath(ID + '.diff').exists():
                ids.append(ID)

        # Store repo in correct partition
        if ids:
            num_ids = len(ids)

            # Select partition to add this repo to
            part_idx = _select_partition(num_ids, partitions)
            data[part_idx][repo] = ids

            # Bookkeeping
            partitions[part_idx] += num_ids
            total_commits += num_ids

    return data, total_commits


def _select_partition(num: int, partitions: List[int]) -> int:
    # For now just greedy select partition with least items
    return np.argmin(partitions)


def check_results_file(p: pathlib.Path, force=False) -> bool:
    msg_results = p.joinpath(DATASET + '.processed.msg')
    diff_results = p.joinpath(DATASET + '.processed.diff')
    diff_results_meta = p.joinpath(DATASET + '.diff.meta.jsonl')

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


def merge_output_files(output_dir: pathlib.Path) -> None:
    _merge_and_delete_files(output_dir, DATASET + '.processed.msg')
    _merge_and_delete_files(output_dir, DATASET + '.processed.diff')
    _merge_and_delete_files(output_dir, DATASET + '.diff.meta.jsonl')


def _merge_and_delete_files(output_dir: pathlib.Path, output_file: str) -> None:
    # Get all parts and sort on name, so messages and diffs are concatenated in the same order
    paths = sorted(output_dir.glob(f'{output_file}.part*'))

    # Merge files
    with output_dir.joinpath(output_file).open('w') as outfile:
        for p in paths:

            # Pipe lines to other file
            with p.open() as infile:
                for line in infile:
                    outfile.write(line)

            # Delete file
            p.unlink()


def chunker_list(seq, size):
    """Source: https://stackoverflow.com/a/43922107"""
    return [seq[i::size] for i in range(size)]
