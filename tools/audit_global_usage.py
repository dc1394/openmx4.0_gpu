#!/usr/bin/env python3
"""Measure how the OpenMX translation units USE global state, function by function.

The header audit (audit_common_globals.py) counts what openmx_common.h declares.
This tool measures the other side: which functions read, write or hand out every
file-scope object, where the file-scope and function-local statics are, and
which translation units read OPENMX_* environment variables.  It compiles each
unit to LLVM IR with clang (-O0, so every access is an explicit load/store) and
walks the IR: a direct load/store of a global, a store through a pointer that
was loaded from a global ("write-through"), and a derived pointer passed to a
call ("escape") are recorded per function.  A flow-insensitive, intra-procedural
provenance of SSA values and -O0 allocas is enough for that; it never claims a
callee does not write what it receives.

Translation units default to the C sources of the Makefile's object list; the
CUDA/HIP .cu/.cpp units and the Fortran units are not covered.  The compiler
flags default to the Makefile's include directories plus the few NVHPC headers
(omp.h, openacc.h, accel.h) the sources include; pass --nvhpc-include to point
at another SDK, or extra flags after --.

  python3 tools/audit_global_usage.py --output build/globals-usage.json \
      --markdown docs/GLOBAL_STATE_INVENTORY.md
  python3 tools/audit_global_usage.py --output build/globals-usage.json \
      --baseline docs/global-usage-baseline.json --check-no-new
"""
import argparse
from collections import Counter, defaultdict
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
GLOBALS_TU = "openmx_globals.c"
NVHPC_HEADERS = ("omp.h", "openacc.h", "nvhpc_acc_device_type.h", "accel.h")
ENV_PREFIXES = ("OPENMX_", "GEMMUL8_", "CUDA_MPS_", "NVCOMPILER_", "ACC_")

NAME = r'@("[^"]+"|[\w.$]+)'
RE_GLOBAL_DEF = re.compile(r"^" + NAME + r" = (.*?)\b(global|constant)\b\s+(.*)$")
RE_DEFINE = re.compile(r"^define\s+(.*?)" + NAME + r"\(")
RE_ASSIGN = re.compile(r"^\s+(%[\w.$]+|%\"[^\"]+\") = (\w+)\s+(.*)$")
RE_INSTR = re.compile(r"^\s+(\w+)\s+(.*)$")
RE_GLOBAL_REF = re.compile(NAME)
RE_SSA = re.compile(r"%[\w.$]+|%\"[^\"]+\"")
RE_STR = re.compile(r'c"((?:[^"\\]|\\.)*)"')
RE_GEP_BASE = re.compile(r"^(?:inbounds\s+|nuw\s+|nusw\s+|inrange\([^)]*\)\s+)*[^,]+,\s*(ptr\s+(?:@(?:\"[^\"]+\"|[\w.$]+)|%[\w.$]+|%\"[^\"]+\"))")


def unquote(name):
    return name[1:-1] if name.startswith('"') else name


def c_string(payload):
    """Decode an LLVM c"..." payload (\\XX escapes) up to the first NUL."""
    out = bytearray()
    i = 0
    while i < len(payload):
        ch = payload[i]
        if ch == "\\" and i + 2 < len(payload) + 1:
            esc = payload[i + 1:i + 3]
            if esc == "\\\\":
                out.append(92)
                i += 2
                continue
            try:
                out.append(int(esc, 16))
                i += 3
                continue
            except ValueError:
                pass
        out.append(ord(ch))
        i += 1
    text = out.split(b"\0", 1)[0]
    try:
        return text.decode("utf-8")
    except UnicodeDecodeError:
        return text.decode("latin-1")


def global_operands(text):
    """Global names appearing as operands in an instruction, in order."""
    return [unquote(m.group(1)) for m in RE_GLOBAL_REF.finditer(text)]


class Unit:
    """The global-state usage of one translation unit, read from its IR."""

    def __init__(self, source):
        self.source = source
        self.defined = {}       # external-linkage definitions: name -> type text
        self.declared = set()   # external references
        self.file_statics = {}  # name -> {"type", "const"}
        self.function_statics = {}  # name -> {"function", "variable", "type"}
        self.strings = {}       # constant name -> text
        self.functions = []     # dicts
        self.error = None

    def parse(self, ir):
        lines = ir.splitlines()
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            if line.startswith("@"):
                self._module_line(line)
            elif line.startswith("define "):
                j = i + 1
                while j < n and lines[j] != "}":
                    j += 1
                self._function(line, lines[i + 1:j])
                i = j
            i += 1

    def _module_line(self, line):
        m = RE_GLOBAL_DEF.match(line)
        if not m:
            return
        name, linkage, kind, rest = unquote(m.group(1)), m.group(2), m.group(3), m.group(4)
        if "private" in linkage or name.startswith((".", "__const.", "__func__.", "__PRETTY_FUNCTION__.", "llvm.")):
            sm = RE_STR.search(rest)
            if sm:
                self.strings[name] = c_string(sm.group(1))
            return
        if name.startswith(("switch.table.", "__profn", "__profc", "__llvm_")):
            return
        type_text = rest.split(",")[0].strip()
        if "external" in linkage.split():
            self.declared.add(name)
        elif "internal" in linkage.split():
            if "." in name:
                function, variable = name.split(".", 1)
                variable = re.sub(r"\.\d+$", "", variable)
                self.function_statics[name] = {"function": function, "variable": variable,
                                               "type": type_text, "const": kind == "constant"}
            else:
                self.file_statics[name] = {"type": type_text, "const": kind == "constant"}
        else:
            self.defined[name] = type_text

    def _function(self, header, body):
        m = RE_DEFINE.match(header)
        if not m:
            return
        attrs, name = m.group(1), unquote(m.group(2))
        reads, writes, writes_through, escapes, address_taken, env = (set(), set(), set(), set(), set(), set())
        self.current_address_calls = defaultdict(set)
        derived = defaultdict(set)   # SSA value or alloca -> globals it derives from
        allocas = set()

        def origins(operand_text):
            found = set()
            for value in RE_SSA.findall(operand_text):
                found |= derived.get(value, set())
            return found

        def note_direct(globs, target):
            for g in globs:
                if g in self.strings:
                    if self.strings[g].startswith(ENV_PREFIXES):
                        env.add(self.strings[g])
                elif g.startswith(("llvm.", ".")):
                    continue
                else:
                    target.add(g)

        def operands(text):
            """Globals named in an instruction: string constants are kept (env names), intrinsics dropped."""
            return [g for g in global_operands(text) if g in self.strings or not g.startswith(("llvm.", "."))]

        for line in body:
            a = RE_ASSIGN.match(line)
            if a:
                value, op, rest = a.group(1), a.group(2), a.group(3)
                if op == "alloca":
                    allocas.add(value)
                    continue
                globs = operands(rest)
                src = origins(rest)
                if op == "load":
                    # load of a global's value, or through a derived pointer; only a
                    # loaded POINTER carries provenance on (an integer index does not)
                    direct = [g for g in globs if g not in self.strings]
                    note_direct(direct, reads)
                    if rest.startswith("ptr,") or rest.startswith("ptr "):
                        derived[value] = set(direct) | src
                elif op == "getelementptr":
                    # provenance of the base pointer only, never of the indices
                    base = RE_GEP_BASE.match(rest)
                    base_text = base.group(1) if base else rest
                    base_globs = [g for g in global_operands(base_text) if g not in self.strings]
                    note_direct([g for g in globs if g in self.strings], reads)
                    derived[value] = set(base_globs) | origins(base_text)
                elif op in ("bitcast", "addrspacecast"):
                    derived[value] = set(g for g in globs if g not in self.strings) | src
                elif op in ("phi", "select"):
                    if rest.startswith("ptr") or rest.startswith("i1 "):
                        derived[value] = set(g for g in globs if g not in self.strings) | src
                elif op == "call" or op == "invoke":
                    self._call(rest, globs, src, escapes, address_taken, env)
                    if rest.startswith("ptr "):
                        derived[value] = src   # a returned pointer may alias what was passed
                continue
            m2 = RE_INSTR.match(line)
            if not m2:
                continue
            op, rest = m2.group(1), m2.group(2)
            globs = operands(rest)
            if op == "store":
                # store VALUE, ptr TARGET
                parts = rest.rsplit(", ptr ", 1)
                if len(parts) == 2:
                    value_text, target_text = parts
                    target_globs = [g for g in global_operands(target_text) if g not in self.strings]
                    note_direct(target_globs, writes)
                    for g in origins(target_text):
                        writes_through.add(g)
                    # the stored value: a global address or derived pointer kept in a local
                    value_globs = [g for g in global_operands(value_text) if g not in self.strings]
                    note_direct(value_globs, address_taken)
                    if value_text.startswith("ptr "):
                        for t in RE_SSA.findall(target_text):
                            if t in allocas:
                                derived[t] |= set(value_globs) | origins(value_text)
                    note_direct([g for g in global_operands(value_text) if g in self.strings], reads)
                else:
                    note_direct(globs, writes)
            elif op in ("call", "invoke"):
                self._call(rest, globs, origins(rest), escapes, address_taken, env)
            elif op in ("br", "switch", "ret", "icmp", "fcmp"):
                note_direct([g for g in globs if g not in self.strings], reads)
        self.functions.append({
            "name": name, "static": "internal" in attrs.split(),
            "reads": sorted(reads), "writes": sorted(writes), "writes_through": sorted(writes_through),
            "escapes": sorted(escapes), "address_taken": sorted(address_taken), "env": sorted(env),
            "address_calls": {g: sorted(c) for g, c in sorted(self.current_address_calls.items())}})

    def _call(self, rest, globs, src, escapes, address_taken, env):
        callee = None
        m = re.search(r"\)\s*" + NAME + r"\(", rest) or re.search(NAME + r"\(", rest)
        if m:
            callee = unquote(m.group(1))
        for g in globs:
            if g == callee:
                continue
            if g in self.strings:
                if self.strings[g].startswith(ENV_PREFIXES):
                    env.add(self.strings[g])
            else:
                address_taken.add(g)
                self.current_address_calls[g].add(callee or "?")
        for g in src:
            escapes.add(g)


def compile_unit(args):
    clang, flags, source = args
    unit = Unit(str(source.relative_to(ROOT)))
    result = subprocess.run([clang, *flags, "-O0", "-g0", "-S", "-emit-llvm", "-o", "-", str(source)],
                            capture_output=True, text=True)
    if result.returncode or not result.stdout:
        errors = [l for l in result.stderr.splitlines() if "error:" in l]
        unit.error = errors[0] if errors else result.stderr.strip().splitlines()[-1:] or "no output"
        return unit
    unit.parse(result.stdout)
    return unit


def make_variable(name, make_args):
    out = subprocess.run(["make", "-s", "-C", str(SOURCE), *make_args,
                          "--eval=print-%: ; @echo \"$($*)\"", f"print-{name}"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def default_units(make_args):
    objs = make_variable("OBJS", make_args).split()
    units = []
    for obj in objs:
        if obj.endswith(".o") and (SOURCE / (obj[:-2] + ".c")).is_file():
            units.append(SOURCE / (obj[:-2] + ".c"))
    return sorted(set(units))


def default_flags(make_args, nvhpc_include, tempdir):
    flags = ["-x", "c", "-std=c11", "-fopenmp", "-w", "-D_DEFAULT_SOURCE",
             "-Wno-error=implicit-function-declaration",
             "-I", str(SOURCE), "-I", str(ROOT / "core" / "include")]
    flags += make_variable("COMMON_INCLUDES", make_args).split()
    elpa_include = make_variable("GPUSOLVER2_ELPA_INCLUDE", make_args)
    if elpa_include:
        flags += ["-I", elpa_include]
    mpi_bin = make_variable("NVHPC_MPI_BIN", make_args)
    if mpi_bin:
        inc = subprocess.run([str(Path(mpi_bin) / "mpicc"), "--showme:incdirs"], capture_output=True, text=True)
        for d in inc.stdout.split():
            flags += ["-I", d]
    if nvhpc_include is None:
        bin_dir = make_variable("NVHPC_COMPILER_BIN", make_args)
        if bin_dir:
            nvhpc_include = Path(bin_dir).parent / "include"
    if nvhpc_include and Path(nvhpc_include).is_dir():
        # only the headers the sources need; the SDK's intrinsics headers do not build with clang
        private = Path(tempdir) / "nvhpc-include"
        private.mkdir(exist_ok=True)
        for header in NVHPC_HEADERS:
            src = Path(nvhpc_include) / header
            if src.is_file():
                shutil.copy(src, private / header)
        flags += ["-I", str(private)]
    return flags


def analyse(units):
    common = {}
    for u in units:
        if Path(u.source).name == GLOBALS_TU:
            common = dict(u.defined)
    globals_index = {}   # name -> record

    def record(name):
        if name not in globals_index:
            globals_index[name] = {"name": name, "readers": set(), "writers": set(), "writers_through": set(),
                                   "escapes": set(), "address_taken": set(), "broadcast": set(), "units": set(),
                                   "kind": "common_header" if name in common else "other"}
        return globals_index[name]

    for u in units:
        if u.error:
            continue
        for name, type_text in u.defined.items():
            r = record(name)
            r["defined_in"] = u.source
            r["type"] = type_text
        for f in u.functions:
            fid = f"{u.source}:{f['name']}"
            for key, field in (("reads", "readers"), ("writes", "writers"), ("writes_through", "writers_through"),
                               ("escapes", "escapes"), ("address_taken", "address_taken")):
                for g in f[key]:
                    if g in u.file_statics or g in u.function_statics:
                        continue
                    r = record(g)
                    if field == "address_taken" and set(f["address_calls"].get(g, [])) <= BROADCAST_CALLS:
                        r["broadcast"].add(fid)   # replicated by MPI_Bcast, not a new value
                    else:
                        r[field].add(fid)
                    r["units"].add(u.source)
    for r in globals_index.values():
        for k in ("readers", "writers", "writers_through", "escapes", "address_taken", "broadcast", "units"):
            r[k] = sorted(r[k])
        r["writer_units"] = sorted({x.split(":")[0] for x in r["writers"] + r["writers_through"] + r["address_taken"] + r["escapes"]})
    return common, globals_index


INPUT_UNITS = {"source/Input_std.c", "source/Inputtools.c"}
BROADCAST_CALLS = {"MPI_Bcast"}
ALLOCATOR_UNITS = {"source/Allocate_Arrays.c", "source/Free_Arrays.c", "source/truncation.c",
                   "source/Set_Allocate_Atom2CPU.c", "source/Allocate_DM_for_LPQ.c"}


def classify(record, common):
    """A coarse ownership class from who writes the object.

    Direct stores and the object's own address passed to a call are "strict"
    writes of the object; stores through a pointer loaded from it write its
    contents.  Derived pointers passed to calls (escapes) are not counted as
    writes here: for value handles such as MPI communicators they are reads.
    """
    strict = {x.split(":")[0] for x in record["writers"] + record["address_taken"]}
    through = {x.split(":")[0] for x in record["writers_through"]}
    if not record["units"]:
        return "unused"
    if not strict and not through:
        return "read-only"
    if strict <= INPUT_UNITS and through <= INPUT_UNITS:
        return "input-configuration"
    if strict and strict <= ALLOCATOR_UNITS | INPUT_UNITS:
        if through <= ALLOCATOR_UNITS | INPUT_UNITS:
            return "allocated-array/contents-fixed"
        return "allocated-array/contents-mutable"
    if len(strict | through) == 1:
        return "single-writer:" + Path(next(iter(strict | through))).name
    return "shared-mutable"


def markdown(report):
    s = report["summary"]
    lines = ["# Global state inventory (usage)", "",
             "Generated by `tools/audit_global_usage.py` from the LLVM IR of each C translation unit "
             f"(revision `{report.get('revision', '?')}`). Counts are per translation unit and per function; "
             "`write` is a direct store to the object, `write-through` a store through a pointer loaded from it, "
             "`escape` a derived pointer passed to a call (the callee may write), `address` the object's own address passed to a call.", ""]
    lines += ["## Summary", "", "| metric | value |", "|---|---:|"]
    for k, v in s.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Ownership classes of the common-header objects", "",
              "Who writes an object decides how it can leave the common header: an `input-configuration` object "
              "is only written by the input parser (an immutable configuration struct); a `single-writer` object "
              "belongs to one module; `shared-mutable` objects need a shared state object; `read-only` objects are "
              "written by no analysed unit (set in C++/CUDA code or constants); `unused` objects are referenced by no function.", "",
              "| class | objects |", "|---|---:|"]
    for k, v in sorted(report["classes"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    outside = report.get("unused_common_referenced_elsewhere", {})
    safe = [n for n in report["unused_common"] if n not in outside]
    lines += ["", "### Common-header objects referenced by no analysed C function", "",
              "Referenced by no other source either (deletion candidates): " + (", ".join(f"`{n}`" for n in safe) or "(none)"), ""]
    if outside:
        lines += ["Referenced only outside the analysed C units (CUDA/C++/Fortran/HIP sources or other headers):", ""]
        for n, hits in outside.items():
            lines.append(f"- `{n}`: " + ", ".join(f"`{h}`" for h in hits[:6]) + (" ..." if len(hits) > 6 else ""))
        lines.append("")
    lines += ["## Common-header objects by number of touching functions (top 40)", "",
              "| object | type | class | functions | readers | writers | write-through | address | broadcast | escapes | units |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in report["top_globals"]:
        lines.append(f"| `{g['name']}` | `{g['type']}` | {g['class']} | {g['functions']} | {g['readers']} | {g['writers']} | {g['writers_through']} | {g['address_taken']} | {g['broadcast']} | {g['escapes']} | {g['units']} |")
    lines += ["", "## Functions by number of distinct common-header objects touched (top 40)", "",
              "| function | unit | objects | read | written |", "|---|---|---:|---:|---:|"]
    for f in report["top_functions"]:
        lines.append(f"| `{f['name']}` | `{f['unit']}` | {f['objects']} | {f['read']} | {f['written']} |")
    lines += ["", "## Translation units", "",
              "| unit | functions | objects touched | objects written | file statics | function statics | env vars |", "|---|---:|---:|---:|---:|---:|---:|"]
    for u in report["units"]:
        if u.get("error"):
            lines.append(f"| `{u['source']}` | (not analysed: {u['error']}) | | | | | |")
        else:
            lines.append(f"| `{u['source']}` | {u['functions']} | {u['objects']} | {u['written']} | {u['file_statics']} | {u['function_statics']} | {u['env']} |")
    lines += ["", "## Objects defined outside openmx_globals.c (module-level globals with external linkage)", "",
              "| object | defined in | type | units referencing |", "|---|---|---|---:|"]
    for g in report["other_defined"]:
        lines.append(f"| `{g['name']}` | `{g['defined_in']}` | `{g['type']}` | {g['units']} |")
    lines += ["", "## File-scope statics", "", "| unit | name | type |", "|---|---|---|"]
    for st in report["file_statics"]:
        lines.append(f"| `{st['unit']}` | `{st['name']}` | `{st['type']}`{' (const)' if st['const'] else ''} |")
    lines += ["", "## Function-local statics", "", "| unit | function | variable | type |", "|---|---|---|---|"]
    for st in report["function_statics"]:
        lines.append(f"| `{st['unit']}` | `{st['function']}` | `{st['variable']}` | `{st['type']}`{' (const)' if st['const'] else ''} |")
    lines += ["", "## Environment variables read (by name literal)", "", "| variable | units |", "|---|---|"]
    for name, units in report["env"].items():
        lines.append(f"| `{name}` | {', '.join(f'`{u}`' for u in units)} |")
    lines.append("")
    return "\n".join(lines)


def build_report(units, common, globals_index, revision):
    classes = Counter()
    for name in common:
        rec = globals_index.get(name)
        if rec is None:
            classes["unused"] += 1
        else:
            rec["class"] = classify(rec, common)
            classes[rec["class"]] += 1
    unused_common = sorted(n for n in common if n not in globals_index or not globals_index[n]["units"])
    top_globals = []
    for name, r in globals_index.items():
        if r["kind"] != "common_header":
            continue
        funcs = set(r["readers"]) | set(r["writers"]) | set(r["writers_through"]) | set(r["escapes"]) | set(r["address_taken"]) | set(r["broadcast"])
        top_globals.append({"name": name, "type": r.get("type", common.get(name, "?")), "class": r.get("class", "?"),
                            "functions": len(funcs), "readers": len(r["readers"]), "writers": len(r["writers"]),
                            "writers_through": len(r["writers_through"]), "address_taken": len(r["address_taken"]),
                            "broadcast": len(r["broadcast"]), "escapes": len(r["escapes"]), "units": len(r["units"])})
    top_globals.sort(key=lambda g: (-g["functions"], g["name"]))
    # names referenced by no analysed C unit, cross-checked against the sources this tool does not compile
    outside = {}
    analysed_sources = {str(ROOT / u.source) for u in units}
    skip_parts = {"third_party", "build", "__pycache__", "gpusolver2-install", "fftw3-install", "magma-install"}
    others = [p for p in ROOT.glob("source/**/*") if p.suffix in (".c", ".cu", ".cuh", ".cpp", ".hpp", ".h", ".f90", ".F90", ".f")
              and p.is_file() and not (skip_parts & set(p.parts)) and p.name != "openmx_common.h"
              and p.name != GLOBALS_TU and str(p) not in analysed_sources]
    others += [p for p in ROOT.glob("core/**/*") if p.suffix in (".cpp", ".hpp", ".h", ".cu") and p.is_file()
               and not (skip_parts & set(p.parts))]
    texts = {}
    for p in others:
        try:
            texts[p] = p.read_text(errors="replace")
        except OSError:
            continue
    for name in unused_common:
        pat = re.compile(r"\b" + re.escape(name) + r"\b")
        hits = sorted(str(p.relative_to(ROOT)) for p, txt in texts.items() if pat.search(txt)
                      and not (len(name) <= 2 and p.suffix in (".cpp", ".hpp")))
        if hits:
            outside[name] = hits
    functions = []
    unit_rows = []
    file_statics, function_statics, env = [], [], defaultdict(set)
    for u in units:
        if u.error:
            unit_rows.append({"source": u.source, "error": u.error})
            continue
        touched, written = set(), set()
        for f in u.functions:
            objs = {g for key in ("reads", "writes", "writes_through", "escapes", "address_taken") for g in f[key] if g in common}
            wr = {g for key in ("writes", "writes_through") for g in f[key] if g in common}
            rd = {g for g in f["reads"] if g in common}
            touched |= objs
            written |= wr
            functions.append({"name": f["name"], "unit": u.source, "objects": len(objs), "read": len(rd), "written": len(wr)})
            for e in f["env"]:
                env[e].add(u.source)
        for name, st in sorted(u.file_statics.items()):
            file_statics.append({"unit": u.source, "name": name, **st})
        for name, st in sorted(u.function_statics.items()):
            function_statics.append({"unit": u.source, **st})
        unit_rows.append({"source": u.source, "functions": len(u.functions), "objects": len(touched), "written": len(written),
                          "file_statics": len(u.file_statics), "function_statics": len(u.function_statics),
                          "env": len({e for f in u.functions for e in f["env"]})})
    functions.sort(key=lambda f: (-f["objects"], f["unit"], f["name"]))
    unit_rows.sort(key=lambda r: (-(r.get("objects", -1)), r["source"]))
    other_defined = [{"name": n, "defined_in": r.get("defined_in"), "type": r.get("type"), "units": len(r["units"])}
                     for n, r in sorted(globals_index.items()) if r["kind"] != "common_header" and r.get("defined_in")]
    analysed = [u for u in units if not u.error]
    summary = {
        "translation units analysed": len(analysed),
        "translation units not analysed": len(units) - len(analysed),
        "functions": sum(len(u.functions) for u in analysed),
        "common-header objects (defined in openmx_globals.c)": len(common),
        "common-header objects referenced by some function": sum(1 for n in common if n in globals_index and globals_index[n]["units"]),
        "common-header objects written by some function (direct or through)": sum(1 for n in common if n in globals_index and (globals_index[n]["writers"] or globals_index[n]["writers_through"])),
        "function-object edges (common header)": sum(g["functions"] for g in top_globals),
        "objects with external linkage defined outside openmx_globals.c": len(other_defined),
        "file-scope statics": len(file_statics),
        "function-local statics": len(function_statics),
        "environment variables (name literals)": len(env),
    }
    return {
        "scope": "global-state usage of the C translation units of the NVIDIA build",
        "revision": revision,
        "summary": summary,
        "classes": dict(classes),
        "unused_common": unused_common,
        "unused_common_referenced_elsewhere": outside,
        "top_globals": top_globals[:40],
        "all_globals": {n: {k: v for k, v in r.items()} for n, r in sorted(globals_index.items())},
        "top_functions": functions[:40],
        "functions": [{**f, "detail": None} for f in functions] if False else functions,
        "units": unit_rows,
        "other_defined": other_defined,
        "file_statics": file_statics,
        "function_statics": function_statics,
        "env": {k: sorted(v) for k, v in sorted(env.items())},
        "edges": sorted({f"{f['name']}@{u.source}->{g}" for u in analysed for f in u.functions
                         for key in ("reads", "writes", "writes_through", "escapes", "address_taken") for g in f[key] if g in common}),
        "per_function": [{"unit": u.source, **f} for u in analysed for f in u.functions],
    }


def gate(report, baseline):
    """Lists of what the report adds beyond the baseline."""
    added = {}
    before_edges = set(baseline.get("edges", []))
    added["edges"] = sorted(set(report["edges"]) - before_edges)
    key = lambda st: f"{st['unit']}:{st['name']}"
    added["file_statics"] = sorted({key(s) for s in report["file_statics"]} - {key(s) for s in baseline.get("file_statics", [])})
    fkey = lambda st: f"{st['unit']}:{st['function']}.{st['variable']}"
    added["function_statics"] = sorted({fkey(s) for s in report["function_statics"]} - {fkey(s) for s in baseline.get("function_statics", [])})
    added["env"] = sorted(set(report["env"]) - set(baseline.get("env", {})))
    added["other_defined"] = sorted({g["name"] for g in report["other_defined"]} - {g["name"] for g in baseline.get("other_defined", [])})
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--output", type=Path, required=True, help="JSON report")
    parser.add_argument("--markdown", type=Path, help="also write a Markdown summary")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-output", type=Path, help="also write the gate baseline (edges, statics, env, module globals) without the per-function detail")
    parser.add_argument("--check-no-new", action="store_true", help="fail when the report adds edges, statics or env variables to the baseline")
    parser.add_argument("--units", type=Path, help="file with one source path per line (default: the Makefile's C objects)")
    parser.add_argument("--make-arg", action="append", default=["GEMMUL8_GPU_ARCH=120"], help="make variable for reading the Makefile (repeatable)")
    parser.add_argument("--nvhpc-include", type=Path, help="NVHPC compilers/include (default: from the Makefile)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("flags", nargs=argparse.REMAINDER, help="extra compiler flags after --")
    args = parser.parse_args()
    extra = args.flags[1:] if args.flags[:1] == ["--"] else args.flags
    with tempfile.TemporaryDirectory(prefix="openmx-global-usage-") as tempdir:
        flags = default_flags(args.make_arg, args.nvhpc_include, tempdir) + extra
        if args.units:
            units = [ROOT / line.strip() if not line.startswith("/") else Path(line.strip())
                     for line in args.units.read_text().splitlines() if line.strip()]
        else:
            units = default_units(args.make_arg)
        if not units:
            parser.exit(1, "no translation units found; pass --units\n")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(compile_unit, [(args.clang, flags, u) for u in units]))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "?"
    common, globals_index = analyse(results)
    if not common:
        parser.exit(1, f"{GLOBALS_TU} was not analysed; cannot classify the common-header objects\n")
    report = build_report(results, common, globals_index, revision)
    failed = [u for u in results if u.error]
    for u in failed:
        print(f"not analysed: {u.source}: {u.error}", file=sys.stderr)
    status = 0
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        report["added"] = gate(report, baseline)
        for k, v in report["added"].items():
            if v:
                print(f"new {k} ({len(v)}): " + ", ".join(v[:20]) + (" ..." if len(v) > 20 else ""))
        if args.check_no_new and any(report["added"].values()):
            status = 1
    elif args.check_no_new:
        parser.error("--check-no-new requires --baseline")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n")
    if args.baseline_output:
        lean = {k: report[k] for k in ("scope", "revision", "summary", "classes", "edges", "file_statics",
                                       "function_statics", "env", "other_defined")}
        args.baseline_output.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_output.write_text(json.dumps(lean, indent=0, sort_keys=False) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report))
    for k, v in report["summary"].items():
        print(f"{k}: {v}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
