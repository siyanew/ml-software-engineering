import os
import queue
from typing import Iterable

from git import Repo, Commit
from github import Github

github_api_key = "b92c53e6e67a5b20380843b98dabca329a2f8389"
prefix = "https://github.com/"
suffix = ".git"


def parse_top_repos():
    repo_urls = get_top_java_projects(limit=1)
    for repo_url in repo_urls:
        repo = clone_repo(repo_url)
        parse_commits(repo)


def get_top_java_projects(limit=1000) -> Iterable[str]:
    client = Github(github_api_key)
    repos = client.search_repositories(query='language:java', sort='stars')
    i = 0
    for repo in repos:
        i += 1
        if i > limit:
            return
        yield repo.clone_url


def parse_url_to_path(url: str) -> str:
    return url[len(prefix):-len(suffix)]


def clone_repo(url: str) -> Repo:
    print(url)
    url_to_path = parse_url_to_path(url)
    path = "repos/" + url_to_path
    commits_path = "commits/" + url_to_path

    if not os.path.exists(commits_path):
        os.makedirs(commits_path)

    repo = Repo(path)
    if not repo.bare:
        return repo

    return Repo.clone_from(url, path)


def parse_commits(repo: Repo, limit=100):
    commits: queue.Queue[Commit] = queue.Queue()
    commits.put(repo.head.commit)

    count = 0
    while not commits.empty() and count < limit:
        c = commits.get()
        for p in c.parents:
            commits.put(p)

        if len(c.parents) is 1:
            count += 1
            parse_commit(c)


def parse_commit(commit: Commit):
    repo = commit.repo
    sha = commit.hexsha
    path = "commits/" + parse_url_to_path(repo.remotes.origin.url)
    file_name = f"{path}/{sha}"
    msg_file_name = f"{file_name}.msg"
    diff_file_name = f"{file_name}.diff"

    if os.path.exists(msg_file_name):
        print("Commit already parsed")
        return

    msg = commit.summary
    print(msg)
    with open(msg_file_name, "w+") as msg_file:
        msg_file.write(msg)

    diff = repo.git.diff(commit.parents[0], commit)
    with open(diff_file_name, "w+") as diff_file:
        diff_file.write(diff)


# run with `python git-helper/github_api.py`
if __name__ == '__main__':
    parse_top_repos()
