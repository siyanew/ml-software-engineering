import pathlib


def load_dataset(path_to_dataset: pathlib.Path) -> (dict, int):
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
    """

    # Init return values
    structure = dict()
    num_commits = 0

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

        if ids:
            structure[repo] = ids
            num_commits += len(ids)

    return structure, num_commits
