#!/usr/bin/env python3
"""Keep source/openmx_env_registry.h in step with the readers in source/.

The registry lists every environment name the code reads, as X() lines:

    X(id, "NAME", "kind", "scope", "default", "readers", "description")

This tool scans source/*.c|cu|cpp|h for the readers (the openmx_env_*_at()
accessors, the name-based openmx_env_*() calls, and the ids passed to the
few local wrappers), then rewrites the registry: names the code reads but
the registry lacks are added, the kind is derived from the accessor each
name is read with, the default from doc/environment_variables.json (a
documented "defaults" entry) or else from the fallback argument of the
readers (a literal, a macro or constant that resolves to one, or the
reader of another name; -1 is "auto", INT_MAX "unlimited"; a sentinel
behind an "is it set" test, several different fallbacks or a computed one
leave it empty), the readers column is regenerated, descriptions, scopes
and the names of the build tools are
kept.  --check only reports whether the file would change (exit status 1),
for the test suite and CI.
"""
from pathlib import Path
import argparse
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
REGISTRY = SOURCE / "openmx_env_registry.h"

ROW7 = re.compile(r'^\s*X\((\w+), "([^"]+)", "(\w+)", "(\w+)", "([^"]*)", "([^"]*)", "((?:[^"\\]|\\.)*)"\)')
ROW6 = re.compile(r'^\s*X\((\w+), "([^"]+)", "(\w+)", "(\w+)", "([^"]*)", "((?:[^"\\]|\\.)*)"\)')
# what each accessor parses
ACCESSOR_KIND = {"flag": "flag", "long": "int", "ulong": "uint", "double": "double", "mib_bytes": "mib", "": "string"}
# the local wrappers that take an id: the kind they parse and the index of their fallback argument
WRAPPERS = {"gs2_env_long": ("int", 1), "gs2_forced_mode": ("flag", None), "DC_EnvU32": ("uint", 1), "env_u32": ("uint", 1),
            "positive_env": ("int", 1), "BandCol_HostCacheLimit": ("double", 1), "BandNonCol_HostCacheLimit": ("double", 1),
            "env_percent": ("uint", 2), "env_mib": ("mib", 2), "gemmul8_num_moduli": ("uint", None),
            "gemmul8_disabled": ("flag", None), "gemmul8_environment_override": ("flag", None), "grid_trace": ("flag", None)}
ACCESSORS = {"openmx_env_flag_at": 1, "openmx_env_long_at": 1, "openmx_env_ulong_at": 1, "openmx_env_double_at": 1,
             "openmx_env_mib_bytes_at": 1}
# the widest kind wins when a name is read in several ways
WIDTH = {"string": 6, "double": 5, "int": 4, "mib": 3, "uint": 2, "flag": 1}
LOCAL_PREFIXES = ("OMPI_", "PMI_", "PMIX_", "SLURM_", "MV2_", "CUDA_MPS_", "LIBOMPTARGET_", "HOSTNAME")

HEADER = '''/**********************************************************************
  openmx_env_registry.h

  The registry of the environment settings this build reads: one X() line
  per name, sorted by name.  openmx_env.h turns the first column into the
  ids OPENMX_ENV_<id>, openmx_env.c into the table behind openmx_env_at()
  and friends.  tools/update_env_registry.py rewrites this file from the
  readers in source/ (adds names, derives the kind and the default from
  the accessors, keeps the descriptions); tools/env_registry_doc.py turns
  it, with doc/environment_variables.json, into the tables of the
  implementation document; tools/audit_global_usage.py fails its gate on
  a name read by the code that is missing here.

  Columns: X(id, name, kind, scope, default, readers, description)
    kind     flag | int | uint | mib | double | string | tool
             (what the readers parse; "tool" names are read by the build
             or the tools only and never by openmx itself)
    scope    global: one value for the run, taken from rank 0 after
                     MPI_Init (openmx_env_pack/unpack)
             local:  per process (launcher-provided ranks, host name,
                     re-exec markers, MPS paths); never synchronized
             tool:   see kind tool
    default  the readers' fallback when it is one literal ("auto" for -1,
             "unlimited" for INT_MAX); "" when it is computed or differs
             between readers (the document then describes it)
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
    """name -> {id, kind, scope, default, module, doc}; accepts the earlier six-column rows too"""
    entries = {}
    for line in text.splitlines():
        m = ROW7.match(line)
        if m:
            id_, name, kind, scope, default, module, doc = m.groups()
        else:
            m = ROW6.match(line)
            if not m:
                continue
            id_, name, kind, scope, module, doc = m.groups()
            default = ""
        entries[name] = {"id": id_, "kind": kind, "scope": scope, "default": default, "module": module, "doc": doc}
    return entries


def call_arguments(text, start):
    """the text between the parentheses of the call whose '(' is at start-1"""
    depth, i = 1, start
    while depth and i < len(text):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
    return text[start:i - 1]


def split_arguments(args):
    out, depth, current = [], 0, ""
    for ch in args:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(current.strip())
            current = ""
        else:
            current += ch
    out.append(current.strip())
    return out


def source_files(source):
    return sorted(p for p in source.iterdir() if p.suffix in (".c", ".cu", ".cpp", ".h") and not p.name.startswith("openmx_env"))


def scan_sources(source):
    """name -> {"kinds": set, "files": set, "fallbacks": set of expression texts}"""
    usage = {}
    registry = parse_registry(REGISTRY.read_text()) if REGISTRY.is_file() else {}
    ids = {e["id"]: n for n, e in registry.items()}

    def note(name, kind, file, fallback=None, presence=False):
        u = usage.setdefault(name, {"kinds": set(), "files": set(), "fallbacks": set(), "presence": False})
        if kind:
            u["kinds"].add(kind)
        u["files"].add(file)
        if fallback is not None:
            u["fallbacks"].add(fallback)
        if presence:
            u["presence"] = True   # read only when set: the fallback is a sentinel, not a default

    def names_in(argument):
        return [ids[t] for t in re.findall(r'\bOPENMX_ENV_(\w+)', argument) if t in ids]

    for path in source_files(source):
        text = path.read_text(errors="replace")
        for m in re.finditer(r'\bopenmx_env_(flag|long|ulong|double|mib_bytes)?_?at\s*\(', text):
            accessor = m.group(1) or ""
            arguments = call_arguments(text, m.end())
            parts = split_arguments(arguments)
            # "is it set?" (== NULL / != NULL) says nothing about the kind
            presence = accessor == "" and re.match(r'\s*[!=]=\s*(NULL|nullptr)\b', text[m.end() + len(arguments) + 1:])
            fallback = parts[1] if accessor and len(parts) > 1 else None
            for name in names_in(parts[0] if parts else ""):
                note(name, None if presence else ACCESSOR_KIND[accessor], path.name, fallback, bool(presence))
        for m in re.finditer(r'\bopenmx_env_(get|flag|long|ulong|double)\s*\(\s*"([A-Z][A-Z0-9_]+)"\s*(?:\)|,\s*)', text):
            presence = m.group(1) == "get" and re.match(r'\s*[!=]=\s*(NULL|nullptr)\b', text[m.end():])
            note(m.group(2), None if presence else ACCESSOR_KIND["" if m.group(1) == "get" else m.group(1)], path.name,
                 None, bool(presence))
        for m in re.finditer(r'\bopenmx_env_(?:set|set_if_unset|unset)\s*\(\s*"([A-Z][A-Z0-9_]+)"', text):
            note(m.group(1), None, path.name)
        for wrapper, (kind, index) in WRAPPERS.items():
            for m in re.finditer(r'\b' + wrapper + r'\s*\(', text):
                parts = split_arguments(call_arguments(text, m.end()))
                fallback = parts[index] if index is not None and index < len(parts) else None
                for position, part in enumerate(parts):
                    if index is not None and position >= index:
                        break
                    for name in names_in(part):
                        note(name, kind, path.name, fallback)
        for token in re.findall(r'\bOPENMX_ENV_([A-Z][A-Z0-9_]*)\b', text):
            if token in ids and token not in ("COUNT", "LIST", "ID_", "ENTRY_"):
                note(ids[token], None, path.name)
    return usage


def scan_constants(source):
    """macro and constant names with a one-token value, for resolving fallback arguments"""
    constants = {}
    for path in source_files(source):
        text = path.read_text(errors="replace")
        for m in re.finditer(r'^\s*#\s*define\s+(\w+)\s+([^\s/]+)\s*(?:/[*/].*)?$', text, re.M):
            constants.setdefault(m.group(1), m.group(2))
        for m in re.finditer(r'\b(?:constexpr|static const)\s+[\w:<> ]+?\b(\w+)\s*=\s*([^;,]+);', text):
            constants.setdefault(m.group(1), m.group(2).strip())
    return constants


def resolve_default(expression, constants, depth=0):
    """the documented form of a fallback expression, or None when it is computed"""
    if depth > 8 or expression is None:
        return None
    e = expression.strip()
    e = re.sub(r'^static_cast<[^>]+>\((.*)\)$', r'\1', e, flags=re.S)
    e = re.sub(r'^\((?:long|int|unsigned|size_t|double|float|long long|unsigned long long|unsigned long)\)\s*', '', e)
    while e.startswith("(") and e.endswith(")"):
        e = e[1:-1].strip()
    if e in ("false", "0ULL", "0u", "0U", "0L", "0UL"):
        return "0"
    if e == "true":
        return "1"
    if e == "INT_MAX":
        return "unlimited"
    m = re.fullmatch(r'(-?\d+)(?:\.0*)?(?:[uU]?[lL]{0,2})', e)
    if m:
        return "auto" if m.group(1) == "-1" else str(int(m.group(1)))
    m = re.fullmatch(r'(-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)[fFlL]?', e)
    if m:
        return m.group(1)
    m = re.match(r'openmx_env_(?:flag|long|ulong|double|mib_bytes)_at\s*\((.*)\)$', e, re.S)
    if m:
        parts = split_arguments(m.group(1))
        return resolve_default(parts[-1], constants, depth + 1) if len(parts) > 1 else None
    if re.fullmatch(r'[A-Za-z_]\w*', e) and e in constants:
        return resolve_default(constants[e], constants, depth + 1)
    return None


def documented_defaults(path=ROOT / "doc" / "environment_variables.json"):
    """name -> the document's default text (English), for names whose readers compute theirs"""
    if not path.is_file():
        return {}
    import json
    data = json.loads(path.read_text())
    return {name: text["en"] for row in data.get("rows", []) for name, text in row.get("defaults", {}).items()}


def derive_default(fallbacks, constants):
    if not fallbacks:
        return ""
    resolved = {resolve_default(f, constants) for f in fallbacks}
    if None in resolved or len(resolved) != 1:
        return ""
    return resolved.pop()


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
        rows.append(f'    X({e["id"]}, "{name}", "{e["kind"]}", "{e["scope"]}", "{e["default"]}", "{e["module"]}", "{e["doc"]}")')
    return HEADER + " \\\n".join(rows) + FOOTER


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report whether the registry would change; exit 1 when it would")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    old_text = REGISTRY.read_text()
    entries = parse_registry(old_text)
    usage = scan_sources(SOURCE)
    constants = scan_constants(SOURCE)
    documented = documented_defaults()
    notes = []
    for name, u in sorted(usage.items()):
        if name not in entries:
            scope = "local" if name.startswith(LOCAL_PREFIXES) else "global"
            entries[name] = {"id": ident(name), "kind": "string", "scope": scope, "default": "", "module": "", "doc": ""}
            notes.append(f"added {name} ({scope})")
        e = entries[name]
        kind = derive_kind(name, u["kinds"], e["kind"])
        if kind != e["kind"]:
            notes.append(f"{name}: kind {e['kind']} -> {kind}")
            e["kind"] = kind
        if name in documented:
            # a documented default replaces the readers' fallback; a sentence-long one stays in the document
            default = documented[name] if len(documented[name]) <= 40 else ""
        else:
            default = "" if u["presence"] else derive_default(u["fallbacks"], constants)
        if default != e["default"]:
            notes.append(f"{name}: default {e['default']!r} -> {default!r}")
            e["default"] = default
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
