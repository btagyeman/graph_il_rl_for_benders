from pathlib import Path
from typing import Union


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(relative_path: Union[str, Path]) -> Path:
    return repo_root() / Path(relative_path)
