#!/usr/bin/env python3
"""Prepare an isolated native OpenMX build for the selected GPU backend.

The native makefiles retain their per-file optimization/offload choices. Source
files are linked into the build tree; mutable dependency sources are copied.
Nothing is generated inside the repository's source or dependency checkouts.
"""
import argparse
import json
from pathlib import Path
import shutil


SOURCE_SUFFIXES = {'.c', '.h', '.hpp', '.cu', '.cuh', '.cpp', '.f', '.f90', '.F90', '.mk'}
COPY_IGNORE = shutil.ignore_patterns(
    '.git', '.libs', '.deps', '__pycache__', '*.o', '*.a', '*.lo', '*.la', '*.so', '*.so.*',
    '*.mod', '*.d', '*.pyc', 'config.log', 'config.status', 'config.cache',
    'autom4te.cache', 'CMakeFiles', 'CMakeCache.txt')


def backend_name(value):
    names = {'CUDA': 'CUDA', 'NVIDIA': 'CUDA', 'HIP': 'HIP', 'AMD': 'HIP'}
    try:
        return names[value.upper()]
    except KeyError:
        raise ValueError('backend must be CUDA (NVIDIA) or HIP (AMD)')


def link_file(source, destination):
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise ValueError(f'refusing to replace an unmanaged build file: {destination}')
    destination.symlink_to(source, target_is_directory=source.is_dir())


def copy_sources(source, destination):
    """Incrementally copy source inputs without importing compiled artifacts."""
    destination.mkdir(parents=True, exist_ok=True)
    ignored = set(COPY_IGNORE(str(source), [p.name for p in source.iterdir()]))
    for entry in source.iterdir():
        if entry.name in ignored:
            continue
        target = destination / entry.name
        if entry.is_dir():
            copy_sources(entry, target)
        elif entry.is_file():
            stat = entry.stat()
            if not target.exists() or (target.stat().st_size, target.stat().st_mtime_ns) != (
                    stat.st_size, stat.st_mtime_ns):
                shutil.copy2(entry, target)


def prepare(source_root, build_root, backend):
    source_root = source_root.resolve(strict=True)
    build_root = build_root.resolve()
    source = source_root / 'source'
    if build_root == source_root or source_root in build_root.parents and (
            source in build_root.parents or build_root == source):
        raise ValueError('the build directory must be outside source/')
    if build_root in source_root.parents:
        raise ValueError('the build directory cannot contain the repository')
    marker = build_root / '.openmx-backend.json'
    identity = {'backend': backend, 'source_root': str(source_root)}
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError('this build directory belongs to another backend or repository; '
                         'use a separate build directory')
    build_root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    native = build_root / 'native'
    stage = native / 'source'
    stage.mkdir(parents=True, exist_ok=True)
    # C ABI forwarding headers keep direct Makefile builds compatible while
    # their definitions belong to the independently buildable numerical core.
    link_file(source_root / 'core', native / 'core')

    selected = {p.name: p for p in source.iterdir()
                if p.is_file() and p.suffix in SOURCE_SUFFIXES}
    selected['Makefile'] = source / 'Makefile'
    hip = source / 'backends' / 'hip'
    if backend == 'HIP':
        manifest = json.loads((hip / 'sources.json').read_text())
        # Imported shared sources and subsequent common integration code use
        # the same implementation in both GPU backends.
        selected = {name: source / name for name in manifest['shared'] + manifest.get('integrated_shared', [])
                    if len(Path(name).parts) == 1}
        overrides = {**manifest['overrides'], **manifest.get('integrated_overrides', {})}
        for name, relative in overrides.items():
            selected[name] = hip / relative
        selected['Makefile'] = hip / 'Makefile'
        selected['makefile'] = hip / 'Makefile'
    record = build_root / 'selected-sources.json'
    if record.exists():
        previous = json.loads(record.read_text())['sources']
        for name, relative in previous.items():
            destination = stage / name
            if name not in selected or selected[name].resolve() != (source_root / relative).resolve():
                # Update links previously created by this driver when backend
                # source selection changes. Leave all native build outputs alone.
                if destination.is_symlink() and destination.resolve() == (source_root / relative).resolve():
                    destination.unlink()
    for name, path in selected.items():
        link_file(path, stage / name)
    # Older projections linked the embedded ELPA tree independently of the
    # source manifest. Remove that managed link when updating an existing build.
    legacy_elpa = stage / 'elpa-2018.05.001'
    if legacy_elpa.is_symlink():
        legacy_elpa.unlink()
    link_file(source_root / 'DFT_DATA19', native / 'DFT_DATA19')

    third_party = source / 'third_party'
    private = stage / 'third_party'
    private.mkdir(exist_ok=True)
    link_file(third_party / 'dist', private / 'dist')
    # Both backends use the same pinned GEMMul8 sources, but compile different
    # adapter objects in that tree. Never share its lib/ or object files.
    gemmul8 = third_party / 'GEMMul8'
    if (gemmul8 / 'Makefile').exists():
        copy_sources(gemmul8, private / 'GEMMul8')
    if backend == 'CUDA':
        for path in third_party.iterdir():
            if path.is_file():
                link_file(path, private / path.name)
        fftw = third_party / 'fftw3'
        if (fftw / 'configure.ac').exists():
            copy_sources(fftw, private / 'fftw3')
    else:
        # The HIP makefile expects these paths relative to its own cwd.
        # MAGMA extracts and patches beside the archive, inside this private tree.
        for path in (third_party / 'hip').iterdir():
            if path.is_dir():
                copy_sources(path, private / path.name)
            elif path.is_file():
                link_file(path, private / path.name)
        for directory in ('tests', 'tools'):
            if (hip / directory).is_dir():
                copy_sources(hip / directory, native / directory)
        # Native extraction tests share the ownership-boundary SDK shim with
        # the root tests; keep one implementation in both source projections.
        for name in ('native_resource_test_bridge.py', 'force_gpu_fixture.py'):
            resource_bridge = source_root / 'tests' / name
            if resource_bridge.is_file():
                link_file(resource_bridge, native / 'tests' / resource_bridge.name)
        # Copied regression runners read their default templates from ../work.
        # Share only the immutable input suites; keep work/ itself private so
        # their generated inputs and results stay inside the selected build.
        for directory in ('input_example', 'large_example', 'large2_example', 'large3_example'):
            link_file(source_root / 'work' / directory, native / 'work' / directory)
    selection = {name: str(path.relative_to(source_root)) for name, path in selected.items()}
    content = json.dumps({'backend': backend, 'sources': selection}, indent=2, sort_keys=True) + '\n'
    if not record.exists() or record.read_text() != content:
        record.write_text(content)
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--build-root', required=True, type=Path)
    parser.add_argument('--backend', required=True)
    args = parser.parse_args()
    try:
        stage = prepare(args.source_root, args.build_root, backend_name(args.backend))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'OpenMX build preparation failed: {error}\n')
    print(stage)


if __name__ == '__main__':
    main()
