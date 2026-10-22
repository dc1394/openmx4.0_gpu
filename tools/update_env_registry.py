#!/usr/bin/env python3
"""Keep source/openmx_env_registry.h in step with the readers in source/.

The registry lists every environment name the code reads, as X() lines:

    X(id, "NAME", "kind", "scope", "readers", "description")

This tool scans source/*.c|cu|cpp|h for the readers (the openmx_env_*_at()
accessors, the name-based openmx_env_*() calls, and the ids passed to the
few local wrappers), then rewrites the registry: names the code reads but
the registry lacks are added, the kind is derived from the accessor each
name is read with, the readers column is regenerated, descriptions, scopes
and the names of the build tools are kept.  --check only reports whether
the file would change (exit status 1), for the test suite and CI.
"""
from pathlib import Path
import argparse
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
REGISTRY = SOURCE / "openmx_env_registry.h"

ROW = re.compile(r'^\s*X\((\w+), "([^"]+)", "(\w+)", "(\w+)", "([^"]*)", "((?:[^"\\]|\\.)*)"\)')
# what each accessor parses
ACCESSOR_KIND = {"flag": "flag", "long": "int", "ulong": "uint", "double": "double", "mib_bytes": "mib", "": "string"}
# the local wrappers that take an id: the kind they parse
WRAPPER_KIND = {"gs2_env_long": "int", "gs2_forced_mode": "flag", "DC_EnvU32": "uint", "env_u32": "uint",
                "positive_env": "int", "BandCol_HostCacheLimit": "double", "BandNonCol_HostCacheLimit": "double",
                "env_percent": "uint", "env_mib": "mib", "gemmul8_num_moduli": "uint", "gemmul8_disabled": "flag",
                "gemmul8_environment_override": "flag", "grid_trace": "flag"}
# the widest kind wins when a name is read in several ways
WIDTH = {"string": 6, "double": 5, "int": 4, "mib": 3, "uint": 2, "flag": 1}
LOCAL_PREFIXES = ("OMPI_", "PMI_", "PMIX_", "SLURM_", "MV2_", "CUDA_MPS_", "LIBOMPTARGET_", "HOSTNAME")

HEADER = '''/**********************************************************************
  openmx_env_registry.h

  The registry of the environment settings this build reads: one X() line
  per name, sorted by name.  openmx_env.h turns the first column into the
  ids OPENMX_ENV_<id>, openmx_env.c into the table behind openmx_env_at()
  and friends.  tools/update_env_registry.py rewrites this file from the
  readers in source/ (adds names, derives the kind from the accessor
  used, keeps the descriptions); tools/audit_global_usage.py fails its
  gate on a name read by the code that is missing here.

  Columns: X(id, name, kind, scope, readers, description)
    kind   flag | int | uint | mib | double | string | tool
           (what the readers parse; "tool" names are read by the build
           or the tools only and never by openmx itself)
    scope  global: one value for the run, taken from rank 0 after
                   MPI_Init (openmx_env_pack/unpack)
           local:  per process (launcher-provided ranks, host name,
                   re-exec markers, MPS paths); never synchronized
           tool:   see kind tool
***********************************************************************/
#ifndef OPENMX_ENV_REGISTRY_H
#define OPENMX_ENV_REGISTRY_H

#define OPENMX_ENV_LIST(X) \\
'''
FOOTER = '''

#endif /* OPENMX_ENV_REGISTRY_H */
'''


def ident(name):
    return name[len("OPENMX_"):] if name.startswith("OPENMX_") else "EXT_" + name


def parse_registry(text):
    entries = {}
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            id_, name, kind, scope, module, doc = m.groups()
            entries[name] = {"id": id_, "kind": kind, "scope": scope, "module": module, "doc": doc}
    return entries


def call_arguments(text, start):
    """the text between the parentheses of the call whose '(' is at start-1"""
    depth, i = 1, start
    while depth and i < len(text):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
    return text[start:i - 1]


def scan_sources(source):
    """name -> {"kinds": set, "files": set}"""
    usage = {}
    ids = {}

    def note(name, kind, file):
        u = usage.setdefault(name, {"kinds": set(), "files": set()})
        if kind:
            u["kinds"].add(kind)
        u["files"].add(file)

    files = sorted(p for p in source.iterdir() if p.suffix in (".c", ".cu", ".cpp", ".h") and not p.name.startswith("openmx_env"))
    registry = parse_registry(REGISTRY.read_text()) if REGISTRY.is_file() else {}
    for name, e in registry.items():
        ids[e["id"]] = name
    for path in files:
        text = path.read_text(errors="replace")
        for m in re.finditer(r'\bopenmx_env_(flag|long|ulong|double|mib_bytes)?_?at\s*\(', text):
            accessor = m.group(1) or ""
            arguments = call_arguments(text, m.end())
            # "is it set?" (== NULL / != NULL) says nothing about the kind
            presence = accessor == "" and re.match(r'\s*[!=]=\s*(NULL|nullptr)\b', text[m.end() + len(arguments) + 1:])
            for token in re.findall(r'\bOPENMX_ENV_(\w+)', arguments):
                if token in ids:
                    note(ids[token], None if presence else ACCESSOR_KIND[accessor], path.name)
        for m in re.finditer(r'\bopenmx_env_(get|flag|long|ulong|double)\s*\(\s*"([A-Z][A-Z0-9_]+)"\s*\)', text):
            presence = m.group(1) == "get" and re.match(r'\s*[!=]=\s*(NULL|nullptr)\b', text[m.end():])
            note(m.group(2), None if presence else ACCESSOR_KIND["" if m.group(1) == "get" else m.group(1)], path.name)
        for m in re.finditer(r'\bopenmx_env_(?:set|set_if_unset|unset)\s*\(\s*"([A-Z][A-Z0-9_]+)"', text):
            note(m.group(1), None, path.name)
        for wrapper, kind in WRAPPER_KIND.items():
            for m in re.finditer(r'\b' + wrapper + r'\s*\(', text):
                for token in re.findall(r'\bOPENMX_ENV_(\w+)', call_arguments(text, m.end())):
                    if token in ids:
                        note(ids[token], kind, path.name)
        for token in re.findall(r'\bOPENMX_ENV_([A-Z][A-Z0-9_]*)\b', text):
            if token in ids and token not in ("COUNT", "LIST", "ID_", "ENTRY_"):
                note(ids[token], None, path.name)
    return usage


def derive_kind(name, kinds, current):
    if not kinds:
        return current
    kind = max(kinds, key=lambda k: WIDTH[k])
    if kind in ("uint", "int") and name.endswith("_MB"):
        kind = "mib"     # a size in MiB; the readers reject negative values anyway
    return kind


def render(entries):
    rows = []
    for name in sorted(entries):
        e = entries[name]
        rows.append(f'    X({e["id"]}, "{name}", "{e["kind"]}", "{e["scope"]}", "{e["module"]}", "{e["doc"]}")')
    return HEADER + " \\\n".join(rows) + FOOTER


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report whether the registry would change; exit 1 when it would")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    old_text = REGISTRY.read_text()
    entries = parse_registry(old_text)
    usage = scan_sources(SOURCE)
    notes = []
    for name, u in sorted(usage.items()):
        if name not in entries:
            scope = "local" if name.startswith(LOCAL_PREFIXES) else "global"
            entries[name] = {"id": ident(name), "kind": "string", "scope": scope, "module": "", "doc": ""}
            notes.append(f"added {name} ({scope})")
        e = entries[name]
        kind = derive_kind(name, u["kinds"], e["kind"])
        if kind != e["kind"]:
            notes.append(f"{name}: kind {e['kind']} -> {kind}")
            e["kind"] = kind
        module = ",".join(sorted(u["files"]))
        if module != e["module"]:
            e["module"] = module
    for name, e in sorted(entries.items()):
        if name not in usage and e["kind"] != "tool":
            notes.append(f"{name}: no reader found (kept)")
    ids = [e["id"] for e in entries.values()]
    if len(set(ids)) != len(ids):
        print("duplicate ids", file=sys.stderr)
        return 2
    new_text = render(entries)
    changed = new_text != old_text
    if not args.quiet:
        for n in notes:
            print(n)
        print(f"{len(entries)} names; registry {'would change' if changed else 'is up to date'}")
    if args.check:
        return 1 if changed else 0
    if changed:
        REGISTRY.write_text(new_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
