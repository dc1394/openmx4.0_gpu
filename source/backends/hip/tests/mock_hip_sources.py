"""Stage production sources beside HIP test doubles in a temporary folder."""
from pathlib import Path

def stage(root, directory, *sources):
    directory=Path(directory)
    for header in (root/'source').glob('*.h'):
        target=directory/header.name
        if not header.name.startswith(('hip','set_hip','set_openmp')) and not target.exists():
            target.write_bytes(header.read_bytes())
    for name in sources:
        (directory/name).write_bytes((root/'source'/name).read_bytes())
    return directory
