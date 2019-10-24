import pathlib
from typing import List

from preprocessing import constants


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


def check_results_file(p: pathlib.Path, force=False) -> bool:
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


def merge_output_files(output_dir: pathlib.Path) -> None:
    dataset = output_dir.name
    _merge_and_delete_files(output_dir, dataset + '.processed.msg')
    _merge_and_delete_files(output_dir, dataset + '.processed.diff')
    _merge_and_delete_files(output_dir, dataset + '.diff.meta.jsonl')


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
