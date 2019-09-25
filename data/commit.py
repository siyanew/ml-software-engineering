from typing import List


class Commit:
    def __init__(self, repo: str, hash_id: str, parent_hash_id: str, msg: str):
        self.repo = repo
        self.hash_id = hash_id
        self.parent_hash_id = parent_hash_id
        self.msg = msg

    def get_parent_code(self) -> List[str]:
        """Returns original code for each changed file"""
        return ["File1 content before", "File2 content before"]

    def get_code(self) -> List[str]:
        """Returns resulting code for each changed file"""
        return ["File1 content after", "File2 content after"]

    def get_diff(self) -> str:
        """Returns the git diff for this commit"""
        return """diff --git a/File1 b/File1
index 1d12e7f..c816f1c 100644
--- a/File1
+++ b/File1
@@ -1 +1 @@
-File1 content before
+File1 content afterdiff --git a/File2 b/File2
index fd16e7f..r896f1c 100644
--- a/File2
+++ b/File2
@@ -1 +1 @@
-File2 content before
+File2 content after
"""
