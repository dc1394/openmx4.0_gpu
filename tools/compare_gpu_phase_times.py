#!/usr/bin/env python3
"""Compare completed cases using the final OpenMX MPI timing summaries.

Example:
  python3 tools/compare_gpu_phase_times.py \
    work/codex_20260930/before_S_repeat work/codex_20260930/after_S

The phase values are Max_Time over MPI ranks; different phases can have
different slowest ranks, so phase sums do not exactly partition elapsed time.
Use --json for machine-readable output. Paired cases only are aggregated.
"""
import argparse
import json
import re
from pathlib import Path

ROW = re.compile(
    r"^\s*(.+?)\s+=\s+(\d+)\s+([\d.eE+-]+)\s+(\d+)\s+([\d.eE+-]+)\s*$",
    re.MULTILINE,
)
OUTER = {"Total Computational Time", "DFT", "readfile", "truncation", "MD_pac", "OutData", "HWF"}


def read_run(directory):
    cases = {}
    for log in sorted(directory.glob("*/run.log")):
        text = log.read_text(errors="replace")
        if "Computational Time (second)" not in text:
            continue
        tail = text.rsplit("Computational Time (second)", 1)[-1]
        if "The calculation was normally finished." not in tail:
            continue
        phases = {row[0].strip(): float(row[4]) for row in ROW.findall(tail)}
        if "Total Computational Time" not in phases or "DFT" not in phases:
            continue
        cases[log.parent.name] = {
            "max_times": phases,
            "scf_steps": len(set(re.findall(r"\*+ MD=\s*(\d+)\s+SCF=\s*(\d+) \*+", text))),
            "global_eigensolver_cpu_fallback": "global dense eigensolver paths use a CPU fallback" in text,
        }
    return cases


def delta(before, after):
    return {"before_s": before, "after_s": after, "saved_s": before - after,
            "speedup": before / after if after else None}


def compare(before_dir, after_dir):
    for directory in (before_dir, after_dir):
        if not directory.is_dir():
            raise ValueError(f"Run directory does not exist: {directory}")
    suites = []
    for directory in (before_dir, after_dir):
        metadata = directory / "command.json"
        command = json.loads(metadata.read_text()) if metadata.exists() else {}
        if command.get("together"):
            raise ValueError(f"Per-case logs are required; this run combines the suite in one job: {directory}")
        suites.append(command.get("suite"))
    if all(suites) and suites[0] != suites[1]:
        raise ValueError(f"Cannot compare different suites: {suites[0]} and {suites[1]}")
    before, after = read_run(before_dir), read_run(after_dir)
    common = sorted(before.keys() & after.keys())
    if not common:
        raise ValueError("No paired completed case logs")
    phases = set.intersection(*(set(run[c]["max_times"]) for run in (before, after) for c in common))
    aggregate = {
        phase: delta(*(sum(run[c]["max_times"][phase] for c in common) for run in (before, after)))
        for phase in phases
    }
    cases = []
    for case in common:
        b, a = before[case], after[case]
        changes = {p: delta(b["max_times"][p], a["max_times"][p]) for p in phases}
        largest = sorted((p for p in phases if p not in OUTER), key=lambda p: -changes[p]["saved_s"])[:3]
        cases.append({"case": case, "scf_before": b["scf_steps"], "scf_after": a["scf_steps"],
                      "cpu_fallback_before": b["global_eigensolver_cpu_fallback"],
                      "cpu_fallback_after": a["global_eigensolver_cpu_fallback"],
                      "phases": changes, "largest_dft_savings": largest})
    return {"before": str(before_dir), "after": str(after_dir), "paired_cases": len(common),
            "before_only_completed": sorted(before.keys() - after.keys()),
            "after_only_completed": sorted(after.keys() - before.keys()),
            "aggregate_max_times": aggregate, "cases": cases,
            "note": "Paired completed cases only. Phase Max_Time values can refer to different MPI ranks; they are not additive."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path, help="baseline run directory containing case/run.log files")
    parser.add_argument("after", type=Path, help="candidate run directory containing case/run.log files")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of the timing table")
    args = parser.parse_args()
    try:
        result = compare(args.before, args.after)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"Paired completed cases: {result['paired_cases']}")
    for side in ("before", "after"):
        if result[f"{side}_only_completed"]:
            print(f"Excluded {side}-only completed cases: {', '.join(result[f'{side}_only_completed'])}")
    print(f"{'Phase (MPI Max_Time)':28s} {'Before':>10s} {'After':>10s} {'Saved':>10s} {'Ratio':>9s}")
    for phase, item in sorted(result["aggregate_max_times"].items(), key=lambda entry: -entry[1]["saved_s"]):
        ratio = f"{item['speedup']:.3f}x" if item["speedup"] is not None else "-"
        print(f"{phase:28s} {item['before_s']:10.3f} {item['after_s']:10.3f} {item['saved_s']:10.3f} {ratio:>9s}")
    print("\nCase: total saved; DFT saved; SCF steps before/after; largest DFT phase savings")
    for item in result["cases"]:
        phases = item["phases"]
        largest = ", ".join(f"{p} {phases[p]['saved_s']:+.3f}s" for p in item["largest_dft_savings"])
        print(f"{item['case']:12s} {phases['Total Computational Time']['saved_s']:+8.3f}s; "
              f"{phases['DFT']['saved_s']:+8.3f}s; {item['scf_before']}/{item['scf_after']}; {largest}")
    print("\n" + result["note"])


if __name__ == "__main__":
    main()
