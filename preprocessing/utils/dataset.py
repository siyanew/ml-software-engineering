import pathlib
import numpy as np
from typing import List


def read_dataset(path_to_dataset: pathlib.Path, num_partitions=1) -> (dict, int):
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
