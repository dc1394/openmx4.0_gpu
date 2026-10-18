#!/usr/bin/env python3
"""Inventory file-scope objects in openmx_common.h using Clang's C AST.

Pass the same include directories/defines as the selected C build, after --.
This deliberately measures the common header, not all application globals:
translation-unit statics, transport headers and third-party state are separate
migration work. The compiler must parse successfully before a report is written.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile


def inventory(header, compiler, flags):
    result = subprocess.run(
        [compiler, "-x", "c", "-std=c11", "-Xclang", "-ast-dump=json",
         "-fsyntax-only", *flags, str(header)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    tree = json.loads(result.stdout)
    objects = []
    for node in tree["inner"]:
        location = node.get("loc", {})
        # The requested header is the main translation unit. Definitions from
        # its included SDK/MPI headers carry includedFrom and are excluded.
        if node.get("kind") != "VarDecl" or "includedFrom" in location:
            continue
        spelling = node["type"]["qualType"]
        objects.append({"name": node["name"], "type": spelling,
                        "pointer_depth": spelling.count("*"),
                        "storage": node.get("storageClass", "external")})
    objects.sort(key=lambda item: item["name"])
    if not objects:
        raise RuntimeError("No common-header objects found; verify header/compiler flags")
    return {"scope": "file-scope objects declared in source/openmx_common.h",
            "count": len(objects),
            "pointer_depth_counts": dict(sorted(Counter(item["pointer_depth"] for item in objects).items())),
            "objects": objects}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--header", type=Path, default=Path(__file__).resolve().parents[1]/"source/openmx_common.h")
    parser.add_argument("--git-ref", help="inventory the header at this revision instead of the working tree")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--check-no-new", action="store_true")
    parser.add_argument("flags", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    flags = args.flags[1:] if args.flags[:1] == ["--"] else args.flags
    header = args.header.resolve()
    flags = ["-I", str(header.parent), *flags]
    try:
        if args.git_ref:
            root = Path(__file__).resolve().parents[1]
            relative = header.relative_to(root)
            content = subprocess.check_output(["git", "show", f"{args.git_ref}:{relative}"], cwd=root)
            with tempfile.TemporaryDirectory(prefix="openmx-global-audit-") as directory:
                snapshot = Path(directory)/header.name
                snapshot.write_bytes(content)
                report = inventory(snapshot, args.clang, flags)
            report["revision"] = args.git_ref
        else:
            report = inventory(header, args.clang, flags)
        if args.baseline:
            baseline = json.loads(args.baseline.read_text())
            before = {item["name"] for item in baseline["objects"]}
            after = {item["name"] for item in report["objects"]}
            report["removed"] = sorted(before-after)
            report["added"] = sorted(after-before)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(f"Common-header objects: {report['count']}; report: {args.output}")
        if args.baseline:
            print(f"Removed: {report['removed']}; added: {report['added']}")
            if args.check_no_new and report["added"]:
                return 1
        elif args.check_no_new:
            parser.error("--check-no-new requires --baseline")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
