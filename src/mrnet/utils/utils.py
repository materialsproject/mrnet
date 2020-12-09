import os
from pathlib import Path
import yaml


def to_path(path):
    return Path(path).expanduser().resolve()


def create_directory(path, path_is_directory=False):
    p = to_path(path)
    if not path_is_directory:
        dirname = p.parent
    else:
        dirname = p
    if not dirname.exists():
        os.makedirs(dirname)
