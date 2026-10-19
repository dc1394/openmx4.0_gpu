#!/usr/bin/env python3
"""Publish a successful build atomically, also while the old binary runs."""
import argparse
import filecmp
import os
from pathlib import Path
import shutil
import tempfile


def copy_executable(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
        return
    fd, temporary = tempfile.mkstemp(prefix=".openmx-", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    copy_executable(args.source, args.destination)
