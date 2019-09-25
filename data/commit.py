from typing import List


class Commit:
    def __init__(self, repo: str, hash_id: str, parent_hash_id: str, msg: str):
        self.repo = repo
        self.hash_id = hash_id
        self.parent_hash_id = parent_hash_id
        self.msg = msg

    def get_parent_code(self) -> List[str]:
        pass

    def get_code(self) -> List[str]:
        pass
