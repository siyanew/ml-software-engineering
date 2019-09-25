from typing import List

from data.commit import Commit


class Repo:
    def __init__(self, repo: str):
        self.repo = repo

    def get_commits(self) -> List[Commit]:
        return [
            Commit("group4/repo", "830f7a900c0f70194af0759bb1ceb55e495c7092",
                   "3b8668de2adbacf605c5e0bbb4dd44bb29996d27",
                   "Changed content of File1 and File2")
        ]
