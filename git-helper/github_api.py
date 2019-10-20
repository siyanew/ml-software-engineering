import multiprocessing
import os
import shutil
import sys
from multiprocessing.pool import Pool
from typing import Iterable

from git import Repo, Commit
from github import Github

# language = 'java'
language = 'c#'
limit_amount_projects = 1000
limit_amount_commits = 10000

# processes = 4
processes = multiprocessing.cpu_count() - 1
github_api_key = "b92c53e6e67a5b20380843b98dabca329a2f8389"


def parse_top_repos():
    repo_urls = get_top_projects()
    print("Processes: %d" % processes)
    with Pool(processes) as pool:
        p = pool.map_async(parse_repo, repo_urls, chunksize=1)
        try:
            p.get(0xFFFF)
        except KeyboardInterrupt:
            print("Interrupted")


def get_top_projects() -> Iterable[str]:
    projects_file_path = os.path.join(output_path, "projects.txt")

    if not os.path.exists(projects_file_path):
        # get top projects from github
        os.makedirs(output_path)

        client = Github(github_api_key, per_page=100)
        repos = client.search_repositories(query=f'language:{language}', sort='stars', order='desc')

        # filter duplicates (happens for some reason) and call api until the project amount limit is reached.
        repo_urls = set()
        for repo in repos:
            repo_url = repo.clone_url
            repo_urls.add(repo_url)
            if len(repo_urls) >= limit_amount_projects:
                break

        # save list to file
        with open(projects_file_path, "w+", encoding="utf-8") as file:
            try:
                for repo_url in repo_urls:
                    file.write(f"{repo_url}\n")
            except Exception as e:
                cleanup_file(file)
                raise e

    # yield all lines (repo urls)
    with open(projects_file_path, encoding="utf-8") as file:
        for line in file:
            yield line.rstrip("\n")
    return


def parse_repo(url: str):
    repo_path = url_to_repo_path(url)
    msg_path = url_to_msg_path(url)
    diff_path = url_to_diff_path(url)

    try:
        if os.path.exists(msg_path) and os.path.exists(diff_path) and not os.path.exists(repo_path):
            print(url, "Skipping, already parsed")
            return  # already parsed
        cleanup_repo(url)
        os.makedirs(repo_path)
        os.makedirs(msg_path)
        os.makedirs(diff_path)

        repo = clone_repo(url)
        parse_repo_commits(repo)
        repo.close()
        shutil.rmtree(repo_path)  # delete cloned repo after parsing
    except KeyboardInterrupt:
        print("KeyboardInterrupt in parse_repo", url)
        try:
            repo.close()
        except Exception:
            pass
        cleanup_repo(url)
        return
    except Exception as e:
        print(url, "Exception\n", e)
        try:
            repo.close()
        except Exception:
            pass
        cleanup_repo(url)
        return


def clone_repo(url: str) -> Repo:
    print(url, "Cloning...")

    repo_path = url_to_repo_path(url)
    repo = Repo.clone_from(url, repo_path, multi_options=['--no-checkout'])

    print(url, "Cloning done.")
    return repo


def cleanup_repo(url):
    repo_path = url_to_repo_path(url)
    msg_path = url_to_msg_path(url)
    diff_path = url_to_diff_path(url)
    cleaned = False
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
        cleaned = True
    if os.path.exists(msg_path):
        shutil.rmtree(msg_path)
        cleaned = True
    if os.path.exists(diff_path):
        shutil.rmtree(diff_path)
        cleaned = True
    if cleaned:
        print(url, "CLEANED.")


def parse_repo_commits(repo: Repo):
    url = repo.remotes.origin.url
    print(url, "Parsing commits...")
    count = 0
    for c in repo.iter_commits():
        if len(c.parents) is 1:
            count += 1
            parse_commit(c)
            if count > limit_amount_commits:
                break
    print(url, "Parsing commits done.")


def parse_commit(commit: Commit):
    repo = commit.repo
    sha = commit.hexsha

    url = repo.remotes.origin.url
    msg_file_name = os.path.join(url_to_msg_path(url), f"{sha}.msg")
    diff_file_name = os.path.join(url_to_diff_path(url), f"{sha}.diff")

    if os.path.exists(msg_file_name) and os.path.exists(diff_file_name):
        # Commit already parsed
        return

    msg = commit.message
    write_to_file(msg, msg_file_name)

    diff = repo.git.diff(commit.parents[0], commit)
    write_to_file(diff, diff_file_name)


def write_to_file(text, file_path):
    with open(file_path, "w+", encoding="utf-8") as file:
        # replace CHINA CHINA CHINA
        text_encoded = text.encode('utf-8', 'replace')
        # ignore if bigger than 1MB
        if len(text_encoded) > 1024 * 1024:
            cleanup_file(file)
            return
        safe_text = text_encoded.decode('utf-8')
        file.write(safe_text)


def cleanup_file(file):
    file.flush()
    file.close()
    os.remove(file.name)


def url_to_repo_path(url):
    dir_name = url_to_dir_name(url)
    path = os.path.join(path_bare_repos, dir_name)
    return path


def url_to_msg_path(url):
    dir_name = url_to_dir_name(url)
    path = os.path.join(path_msg, dir_name)
    return path


def url_to_diff_path(url):
    dir_name = url_to_dir_name(url)
    path = os.path.join(path_diff, dir_name)
    return path


def url_to_dir_name(url: str) -> str:
    prefix = "https://github.com/"
    suffix = ".git"

    split = url[len(prefix):-len(suffix)].split("/")
    return f"{split[0]}_{split[1]}"


# run with `python github_api.py ~/path/to/ouput`
# run with `python github_api.py some\windows\path`
if __name__ == '__main__':
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "dataset_output"

    if output_path.endswith("/"):
        output_path = output_path.rstrip("/")
    if output_path.endswith("\\"):
        output_path = output_path.rstrip("\\")

    output_path = os.path.join(output_path, f"{language}-top-{limit_amount_projects}")
    path_bare_repos = os.path.join(output_path, "bare_repos")
    path_extracted_commits = os.path.join(output_path, "extracted_commits")
    path_msg = os.path.join(path_extracted_commits, "msg")
    path_diff = os.path.join(path_extracted_commits, "diff")

    parse_top_repos()
