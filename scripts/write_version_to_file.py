import os
from pathlib import Path

import git

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
VERSION_FILE = PROJECT_ROOT / "src/__about__.py"

repo = git.Repo(PROJECT_ROOT)
tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)

latest_tag = str(tags[-1])
commit_count_since_tag = repo.git.rev_list("--count", f"{latest_tag}..HEAD")
commit_count_since_tag = int(commit_count_since_tag)

if latest_tag[0] == "v":
    latest_tag = latest_tag[1:]

if commit_count_since_tag != 0:
    latest_tag += f"-dev{commit_count_since_tag+1}"

with open(VERSION_FILE, "w") as f:
    f.write(f'__version__ = "{latest_tag}"\n')

repo.index.add([VERSION_FILE])
