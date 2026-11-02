#!/usr/bin/env python3
"""Re-aggregate the individual outputs of the measurement archive into the CSV
files the paper names (Appendix "Provenance of the measurement series"):

  production_runs.csv        one row per run of the main performance series
                             (H100 sip_*, RTX 5080 s5p_* and the np16/np18
                             A/B series s5x_*), timings from the .out Max_Time
                             table, counters from the manifest, memory peaks
  production_summary.csv     mean +- sample SD per (machine, system, config,
                             ranks) over the retained runs, and the speedups
                             the tables and figures quote
  amdahl_sites.csv           the GEMM call sites of the Amdahl-type model:
                             shape, call counts (manifests), stand-alone times
                             (gemm_micro_*.csv) and the walls, both GPUs
  accuracy_runs.csv          one row per run of the accuracy series (displaced
                             Si216 and the auxiliary cases): SCF count,
                             convergence verdict, printed Utot, |F|max
  accuracy_comparison.csv    the decimal differences between configurations
                             (C-O, G-O, C-G and the run-to-run pairs)
  strong_scaling_runs.csv    the dedicated H100 band strong-scaling series sibs_*
  distributed_solver_runs.csv  the cnc384 single-binary series s443nc* (16 and
                             48 ranks per node, the 36-rank comparison)
  supplementary_series_runs.csv  every other run directory with an output
                             (exploratory cluster sizes, DC-LNO node scaling,
                             the preflight probes), so that no run directory
                             of the archive is left unlisted
  reaggregate_sha256.txt     SHA-256 of every file this script read
  reaggregate_check.txt      cross-checks against the archive's own
                             aggregation tables and counts

Only files of the archive are read; nothing is recomputed from scratch
beyond parsing.  "retained" marks the runs that the archive's thesis
aggregation tables list (slow-node discards, superseded preliminary
repetitions and aborted series stay in the archive but are flagged 0).

Usage: reaggregate.py <archive root containing work/> [--out DIR]
       [--gemm-micro DIR]   (default <root>/work/gemm_micro)
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import re
import statistics
import sys
from decimal import Decimal
from pathlib import Path

PAPER_FINAL_NP = {"bcol216": 18, "ccol216": 18, "cnc216": 18}   # RTX 5080, s5prod_thesis_result_v2.txt
RTX_NP18_C18_SYSTEMS = {"bcol216", "bnc216", "ccol216", "cnc216", "bcol384"}
RTX_NP18_GPU_SYSTEMS = {"bcol216", "ccol216", "cnc216"}

SHA = {}


def read_text(path):
    data = Path(path).read_bytes()
    SHA[str(path)] = hashlib.sha256(data).hexdigest()
    return data.decode("utf-8", errors="replace")


def read_json(path):
    return json.loads(read_text(path))


# ---------------------------------------------------------------- parsers
MAXTIME_KEYS = {"Total Computational Time": "total_s", "DFT": "dft_s",
                "Diagonalization": "diag_s", "Set_Hamiltonian": "set_hamiltonian_s"}


def parse_out(path):
    """timing table (Max_Time column), SCF history, printed Utot, forces"""
    r = {"total_s": "", "dft_s": "", "diag_s": "", "set_hamiltonian_s": ""}
    txt = read_text(path)
    for line in txt.splitlines():
        m = re.match(r"\s+(Total Computational Time|DFT|Diagonalization|Set_Hamiltonian)\s+=\s+\d+\s+([\d.]+)\s+\d+\s+([\d.]+)", line)
        if m and MAXTIME_KEYS[m.group(1)] in r and r[MAXTIME_KEYS[m.group(1)]] == "":
            r[MAXTIME_KEYS[m.group(1)]] = m.group(3)
    scf = re.findall(r"SCF=\s*(\d+)\s+NormRD=\s*([\d.]+)\s+Uele=\s*(-?[\d.]+)", txt)
    r["scf_lines"] = len(scf)
    r["last_normrd"] = scf[-1][1] if scf else ""
    r["last_scf"] = scf[-1][0] if scf else ""
    u = re.findall(r"^\s*Utot\.\s+(-?\d+\.\d+)", txt, re.M)
    r["utot"] = u[-1] if u else ""
    m = re.search(r"<coordinates\.forces\n(.*?)coordinates\.forces>", txt, re.S)
    forces = []
    if m:
        for line in m.group(1).splitlines():
            f = line.split()
            if len(f) >= 8:
                forces.append(tuple(f[5:8]))
    r["forces"] = forces
    r["fmax"] = max((abs(Decimal(c)) for fx in forces for c in fx), default=None)
    mc = re.search(r"^scf\.criterion\s+([\d.eE+-]+)", txt, re.M)
    mi = re.search(r"^scf\.maxIter\s+(\d+)", txt, re.M)
    r["criterion"] = mc.group(1) if mc else ""
    r["maxiter"] = mi.group(1) if mi else ""
    return r


def parse_smi(path):
    """nvidia-smi / host-memory sampler: maxima of used VRAM (MiB) and host memory (GiB)"""
    vram = []
    host = []
    for line in read_text(path).splitlines():
        m = re.match(r"\s*\d+ %, (\d+) MiB", line)
        if m:
            vram.append(int(m.group(1)))
        m = re.search(r"host mem used/total \(GiB\): (\d+)/(\d+)", line)
        if m:
            host.append(int(m.group(1)))
    return (max(vram) if vram else ""), (max(host) if host else "")


def parse_env(path):
    txt = read_text(path)
    nodes = []
    grab = False
    for line in txt.splitlines():
        if line.startswith("=== nodes ==="):
            grab = True
            continue
        if grab:
            if line.startswith("=== "):
                break
            if line.strip():
                nodes.append(line.strip())
    cp = re.search(r"control_pipe=(yes|no)", txt)
    return {"nodes": nodes, "control_pipe": cp.group(1) if cp else ""}


def parse_joblog_timing(path):
    for line in read_text(path).splitlines():
        if line.startswith("TIMING "):
            m = re.match(r"TIMING (\S+) total=(\S+) dft=(\S+) diag=(\S+) set_hamiltonian=(\S+)", line)
            if m:
                return m.groups()
            return (line,)
    return None


def run_table_names(path):
    """the run names of the per-run table(s) of an aggregation file"""
    names = set()
    for line in read_text(path).splitlines():
        m = re.match(r"^\s*((?:sip|s5p|s5x|s5a8|s5a|siacc8|siacc64|siacc|sibs|s443nc48|s443nc|s444nc|s444cl|s333nc|dia64dc|cghelgeo)_\S+)\s+[\d-]", line)
        if m:
            names.add(m.group(1))
    return names


# ---------------------------------------------------------------- classification
PROD = re.compile(r"^(?P<prefix>sip|s5p)_(?P<sys>[a-z]+\d+)_(?P<cfg>c48|c16|c18|om18|gm18|o18|g18|om|gm|o|g)(?P<rep>\d+)$")
AB = re.compile(r"^s5x_(?P<sys>[a-z]+\d+)_(?P<cfg>c16|c18h|c18|o18)_(?P<rep>\d+)$")
ACC = re.compile(r"^(?P<fam>siacc8|siacc|s5a8|s5a)_(?P<case>[a-z0-9]+)_(?P<cfg>c1611|c48|c|o|g)(?P<rep>\d+)$")
ACC64 = re.compile(r"^(?P<fam>siacc64)_(?P<cfg>c|o|g)(?P<rep>\d+)$")
SCAL = re.compile(r"^sibs_(?P<sys>col300|nc216)_(?P<nodes>\d)n_(?P<cfg>o|g)(?P<rep>\d+)$")
DIST = re.compile(r"^(?P<fam>s443nc48|s443nc)_(?P<nodes>\d)n(?P<rpn>\d*)_(?P<cfg>[a-z]+)(?P<rep>\d*)$")

CFG_GROUP = {"c": "cpu", "c16": "cpu", "c18": "cpu", "c48": "cpu", "c1611": "cpu", "c18h": "cpu",
             "o": "cublas", "o18": "cublas", "om": "cublas", "om18": "cublas",
             "g": "gemmul8", "g18": "gemmul8", "gm": "gemmul8", "gm18": "gemmul8",
             "d": "gpusolver2", "ds": "gpusolver2", "dv": "gpusolver2", "e": "other"}


def machine_of(prefix):
    return "H100" if prefix in ("sip", "siacc", "siacc8", "siacc64", "sibs", "s443nc", "s443nc48") else "RTX5080"


def sys_parts(sysname):
    m = re.match(r"^([bc])(col|nc)(\d+)$", sysname)
    if not m:
        return "", "", ""
    return ("band" if m.group(1) == "b" else "cluster"), m.group(2), m.group(3)


# ---------------------------------------------------------------- row builders
def base_row(work, name):
    d = work / name
    out = d / f"{name}.out"
    if not out.is_file():
        return None
    o = parse_out(out)
    row = {"run": name, "dir": f"work/{name}", "out_file": f"work/{name}/{name}.out"}
    row.update({k: o[k] for k in ("total_s", "dft_s", "diag_s", "set_hamiltonian_s")})
    row["scf_iters_out"] = o["last_scf"]
    row["last_normrd"] = o["last_normrd"]
    row["utot_printed"] = o["utot"]
    row["fmax_printed"] = str(o["fmax"]) if o["fmax"] is not None else ""
    row["criterion"] = o["criterion"]
    row["maxiter"] = o["maxiter"]
    row["_forces"] = o["forces"]
    mp = d / f"{name}.manifest.json"
    row["manifest_file"] = f"work/{name}/{name}.manifest.json" if mp.is_file() else ""
    m = read_json(mp) if mp.is_file() else {}
    lay = m.get("layout", {})
    inp = m.get("input", {})
    g8 = m.get("gemmul8", {})
    ds = m.get("dense_solver", {})
    mem = m.get("memory", {})
    wall = m.get("wall", {})
    row.update({
        "ranks": lay.get("ranks", ""), "nodes": lay.get("nodes", ""), "threads": lay.get("threads", ""),
        "mps_detected": int(bool(lay.get("mps", {}).get("detected"))) if lay else "",
        "atoms": inp.get("atoms", ""), "solver_input": inp.get("solver", ""), "spin_input": inp.get("spin", ""),
        "kgrid": "x".join(str(k) for k in inp.get("kgrid", [])) if inp else "",
        "eigen_lib_input": inp.get("eigen_lib", ""), "gemmul8_input": inp.get("gemmul8_enable", ""),
        "scf_maxiter_input": inp.get("scf_maxiter", ""), "criterion_input": inp.get("criterion", ""),
        "input_sha256": inp.get("sha256", ""),
        "scf_iters_manifest": wall.get("scf_iters", ""), "scf_converged_manifest": wall.get("scf_converged", ""),
        "dense_path": ds.get("path", ""), "eigen_lib_effective": ds.get("eigen_lib_effective", ""),
        "gpu_solves": ds.get("gpu_solves", ""), "cpu_solves": ds.get("cpu_solves", ""), "gs2_solves": ds.get("gs2_solves", ""),
        "gemmul8_enabled": g8.get("enabled", ""), "d_calls": g8.get("d_calls", ""), "z_calls": g8.get("z_calls", ""),
        "d_fallbacks": g8.get("d_fallbacks", ""), "z_fallbacks": g8.get("z_fallbacks", ""),
        "g8_workspace_percent": g8.get("workspace_percent", ""), "g8_min_free_after_mib": g8.get("min_free_after_mib", ""),
        "peak_device_vram_used_mb": mem.get("peak_device_vram_used_mb", ""),
        "vram_min_free_mb": mem.get("vram_min_free_mb", ""),
        "peak_node0_rss_sum_kb": mem.get("peak_node0_rss_sum_kb", ""),
        "peak_rank_vmhwm_kb_max": mem.get("peak_rank_vmhwm_kb_max", ""),
        "binary_md5": m.get("build", {}).get("md5", ""), "release_tag": m.get("release_tag", ""),
        "source_commit": m.get("source_commit", ""), "gemmul8_commit": m.get("gemmul8_commit", ""),
        "exit_status": m.get("exit_status", ""), "written_at": m.get("written_at", ""),
        "wall_total_manifest_s": wall.get("total_s", ""), "wall_diag_manifest_s": wall.get("diag_s", ""),
    })
    smi = d / f"{name}.smi"
    if smi.is_file():
        row["smi_vram_used_max_mib"], row["smi_host_mem_used_max_gib"] = parse_smi(smi)
    else:
        row["smi_vram_used_max_mib"] = row["smi_host_mem_used_max_gib"] = ""
    env = d / f"{name}.env"
    if env.is_file():
        e = parse_env(env)
        row["env_nodes"] = " ".join(e["nodes"])
        row["env_node_count"] = len(e["nodes"])
        row["mps_control_pipe"] = e["control_pipe"]
    else:
        row["env_nodes"] = row["env_node_count"] = row["mps_control_pipe"] = ""
    jl = d / f"{name}.joblog"
    t = parse_joblog_timing(jl) if jl.is_file() else None
    row["joblog_timing"] = " ".join(t) if t else ""
    return row


def check_timing(row, problems):
    """the joblog TIMING line and the manifest walls must agree with the .out table"""
    t = row.get("joblog_timing", "").split()
    if len(t) == 5 and t[1] != "INCOMPLETE":
        for k, v in zip(("total_s", "dft_s", "diag_s", "set_hamiltonian_s"), t[1:]):
            if row[k] and v != "-" and abs(float(row[k]) - float(v)) > 0.0015:
                problems.append(f"{row['run']}: {k} out={row[k]} joblog={v}")
    for k, mk in (("total_s", "wall_total_manifest_s"), ("diag_s", "wall_diag_manifest_s")):
        if row[k] != "" and row[mk] != "" and abs(float(row[k]) - float(row[mk])) > 0.011:
            problems.append(f"{row['run']}: {k} out={row[k]} manifest={row[mk]}")


def mean_sd(vals):
    vals = [float(v) for v in vals if v != ""]
    if not vals:
        return "", "", 0
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.fmean(vals):.3f}", f"{sd:.3f}", len(vals)


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


RUN_COLS = ["machine", "series", "run", "system", "solver", "spin", "atoms", "config", "config_group", "mps",
            "ranks", "nodes", "threads", "rep", "retained", "retained_in", "paper_final",
            "total_s", "dft_s", "diag_s", "set_hamiltonian_s", "scf_iters_out", "scf_iters_manifest", "last_normrd",
            "utot_printed", "criterion", "maxiter", "dense_path", "eigen_lib_effective", "gpu_solves", "cpu_solves",
            "gs2_solves", "gemmul8_enabled", "d_calls", "z_calls", "d_fallbacks", "z_fallbacks",
            "g8_workspace_percent", "g8_min_free_after_mib", "peak_device_vram_used_mb", "vram_min_free_mb",
            "smi_vram_used_max_mib", "peak_node0_rss_sum_kb", "smi_host_mem_used_max_gib", "peak_rank_vmhwm_kb_max",
            "binary_md5", "release_tag", "source_commit", "gemmul8_commit", "exit_status", "written_at",
            "env_node_count", "env_nodes", "mps_control_pipe", "input_sha256", "out_file", "manifest_file", "dir"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gemm-micro", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    work = root / "work"
    out = Path(a.out) if a.out else Path(__file__).resolve().parent
    gm = Path(a.gemm_micro) if a.gemm_micro else work / "gemm_micro"
    out.mkdir(parents=True, exist_ok=True)
    problems = []

    # retained sets from the archive's own aggregation tables
    tables = {f: run_table_names(work / f) for f in (
        "siprod_thesis_result.txt", "s5prod_thesis_result.txt", "s5prod_bcol384_c16_result.txt",
        "siacc_thesis_result.txt", "s5acc_thesis_result.txt", "siacc8_thesis_result.txt", "s5a8_thesis_result.txt",
        "siband_scaling_thesis_result.txt", "s443nc16_reps_result.txt", "s443nc_reps_result.txt",
        "s444nc_reps_result.txt", "s444cl_reps_result.txt", "s333nc_reps_result.txt", "dia64dc_nodes_result.txt",
        "cghelgeo_v4_result.txt", "cghelgeo_v2_result.txt", "cghelgeo_reps_result.txt") if (work / f).is_file()}

    def retained_in(name):
        return [f for f, names in tables.items() if name in names]

    prod, acc, scal, dist, supp = [], [], [], [], []
    for d in sorted(p for p in work.iterdir() if p.is_dir()):
        name = d.name
        row = base_row(work, name)
        if row is None:
            continue
        check_timing(row, problems)
        row["rep"] = ""
        m = PROD.match(name)
        if m:
            row["machine"] = machine_of(m["prefix"])
            row["series"] = "main" if (row["machine"] == "H100" or not m["cfg"].endswith("18")) else "np18"
            row["system"], row["config"], row["rep"] = m["sys"], m["cfg"], m["rep"]
            row["solver"], row["spin"], _ = sys_parts(m["sys"])
            row["config_group"] = CFG_GROUP[m["cfg"]]
            row["mps"] = "na" if row["config_group"] == "cpu" else ("off" if m["cfg"].startswith(("om", "gm")) else "on")
            ri = retained_in(name)
            if row["machine"] == "RTX5080" and m["cfg"].endswith("18") and (
                    (m["cfg"] == "c18" and m["sys"] in RTX_NP18_C18_SYSTEMS) or
                    (m["cfg"] != "c18" and m["sys"] in RTX_NP18_GPU_SYSTEMS)):
                ri.append("s5prod_thesis_result_v2.txt (np18 policy: " + ("s5prod_c18_result.txt" if m["cfg"] == "c18" else "s5prod_gpu18_result.txt") + ")")
            row["retained"] = int(bool(ri))
            row["retained_in"] = "; ".join(ri)
            final_np = PAPER_FINAL_NP.get(m["sys"], 16) if row["machine"] == "RTX5080" else 48
            row["paper_final"] = int(bool(ri) and int(row["ranks"] or 0) == final_np)
            prod.append(row)
            continue
        m = AB.match(name)
        if m:
            row["machine"] = "RTX5080"
            row["series"] = "np_ab"
            row["system"], row["config"], row["rep"] = m["sys"], m["cfg"], m["rep"]
            row["solver"], row["spin"], _ = sys_parts(m["sys"])
            row["config_group"] = CFG_GROUP.get(m["cfg"], "")
            row["mps"] = "na" if row["config_group"] == "cpu" else "on"
            row["retained"] = 1
            row["retained_in"] = "s5x_np18_ab_result.txt"
            row["paper_final"] = 0
            prod.append(row)
            continue
        m = ACC.match(name) or ACC64.match(name)
        if m:
            fam = m["fam"]
            row["machine"] = machine_of(fam)
            row["series"] = fam
            row["case"] = m.groupdict().get("case", "si64")
            row["config"], row["rep"] = m["cfg"], m["rep"]
            row["config_group"] = CFG_GROUP[m["cfg"]]
            row["config_letter"] = {"cpu": "C", "cublas": "O", "gemmul8": "G"}[row["config_group"]]
            ri = retained_in(name)
            row["retained"] = int(bool(ri))
            row["retained_in"] = "; ".join(ri)
            # the program's own verdict, wall.scf_converged of the manifest:
            # the stopping test passed, possibly at the scf.maxIter-th
            # iteration itself (s5a_bcol_c1611 converges at 60 of 60).  The
            # same flag marks the runs of the paper's accuracy table.  Empty
            # for the preliminary runs that have no manifest.
            sc = row["scf_converged_manifest"]
            row["converged_by_criterion"] = "" if sc in ("", None) else int(bool(sc))
            acc.append(row)
            continue
        m = SCAL.match(name)
        if m:
            row["machine"] = "H100"
            row["series"] = "sibs"
            row["system"] = {"col300": "bcol300", "nc216": "bnc216"}[m["sys"]]
            row["nodes_label"], row["config"], row["rep"] = m["nodes"], m["cfg"], m["rep"]
            row["config_group"] = CFG_GROUP[m["cfg"]]
            ri = retained_in(name)
            row["retained"] = int(bool(ri))
            row["retained_in"] = "; ".join(ri)
            scal.append(row)
            continue
        m = DIST.match(name)
        if m:
            row["machine"] = "H100"
            row["series"] = m["fam"]
            row["nodes_label"], row["ranks_per_node_label"], row["config"], row["rep"] = m["nodes"], m["rpn"] or "48", m["cfg"], m["rep"]
            row["config_group"] = CFG_GROUP.get(m["cfg"], m["cfg"])
            ri = retained_in(name)
            row["retained"] = int(bool(ri))
            row["retained_in"] = "; ".join(ri)
            dist.append(row)
            continue
        row["machine"] = "H100" if name.startswith(("sib_", "s444", "s333", "dia64", "cghelgeo")) else "RTX5080"
        row["series"] = name.split("_")[0]
        ri = retained_in(name)
        row["retained"] = int(bool(ri))
        row["retained_in"] = "; ".join(ri)
        supp.append(row)

    # ---------------- production_runs.csv / production_summary.csv
    prod.sort(key=lambda r: (r["machine"], r["series"], r["system"], r["config"], int(r["rep"])))
    write_csv(out / "production_runs.csv", prod, RUN_COLS)
    groups = {}
    for r in prod:
        if not r["retained"] or r["series"] == "np_ab":
            continue
        groups.setdefault((r["machine"], r["system"], r["config"], str(r["ranks"])), []).append(r)
    summ = []
    for (mach, sysn, cfg, ranks), rs in sorted(groups.items()):
        s = {"machine": mach, "system": sysn, "config": cfg, "config_group": rs[0]["config_group"], "mps": rs[0]["mps"],
             "ranks": ranks, "n_runs": len(rs), "paper_final": rs[0]["paper_final"], "runs": " ".join(r["run"] for r in rs)}
        for k in ("total_s", "dft_s", "diag_s", "set_hamiltonian_s"):
            s[k + "_mean"], s[k + "_sd"], _ = mean_sd([r[k] for r in rs])
        s["peak_device_vram_used_mb_max"] = max((int(r["peak_device_vram_used_mb"]) for r in rs if r["peak_device_vram_used_mb"] != ""), default="")
        s["peak_node0_rss_sum_kb_max"] = max((int(r["peak_node0_rss_sum_kb"]) for r in rs if r["peak_node0_rss_sum_kb"] != ""), default="")
        s["smi_host_mem_used_max_gib_max"] = max((int(r["smi_host_mem_used_max_gib"]) for r in rs if r["smi_host_mem_used_max_gib"] != ""), default="")
        s["z_fallbacks_total"] = sum(int(r["z_fallbacks"] or 0) for r in rs)
        s["d_fallbacks_total"] = sum(int(r["d_fallbacks"] or 0) for r in rs)
        summ.append(s)
    by = {(s["machine"], s["system"], s["ranks"], s["config"]): s for s in summ}

    def ratio(num, den, key):
        if num and den and num[key] and den[key] and float(den[key]) > 0:
            return f"{float(num[key]) / float(den[key]):.4f}"
        return ""
    for s in summ:
        key = (s["machine"], s["system"], s["ranks"])
        cpu = by.get(key + ("c48",)) or by.get(key + ("c18",)) or by.get(key + ("c16",))
        cub = by.get(key + ("o18",)) or by.get(key + ("o",))
        if s["config_group"] in ("cublas", "gemmul8") and s["mps"] == "on":
            s["speedup_vs_cpu_total"] = ratio(cpu, s, "total_s_mean")
            s["speedup_vs_cpu_diag"] = ratio(cpu, s, "diag_s_mean")
        if s["config_group"] == "gemmul8" and s["mps"] == "on":
            s["speedup_vs_cublas_total"] = ratio(cub, s, "total_s_mean")
            s["speedup_vs_cublas_diag"] = ratio(cub, s, "diag_s_mean")
        if s["mps"] == "off":
            on = by.get(key + (s["config"].replace("m", ""),))
            s["mps_off_over_on_total"] = ratio(s, on, "total_s_mean")
            s["mps_off_over_on_diag"] = ratio(s, on, "diag_s_mean")
    SUM_COLS = ["machine", "system", "config", "config_group", "mps", "ranks", "n_runs", "paper_final",
                "total_s_mean", "total_s_sd", "dft_s_mean", "dft_s_sd", "diag_s_mean", "diag_s_sd",
                "set_hamiltonian_s_mean", "set_hamiltonian_s_sd", "speedup_vs_cpu_total", "speedup_vs_cpu_diag",
                "speedup_vs_cublas_total", "speedup_vs_cublas_diag", "mps_off_over_on_total", "mps_off_over_on_diag",
                "d_fallbacks_total", "z_fallbacks_total", "peak_device_vram_used_mb_max", "peak_node0_rss_sum_kb_max",
                "smi_host_mem_used_max_gib_max", "runs"]
    write_csv(out / "production_summary.csv", summ, SUM_COLS)

    # ---------------- amdahl_sites.csv
    def dicts_of(path):
        src = read_text(path)
        wall = ast.literal_eval(re.search(r"^WALL = (\{.*?^\})", src, re.S | re.M).group(1))
        sites = ast.literal_eval(re.search(r"^SITES = (\{.*?^\})", src, re.S | re.M).group(1))
        return wall, sites
    sites_rows = []
    for mach, script, csvname in (("RTX5080", "analysis.py", "gemm_micro_5080.csv"), ("H100", "analysis_h100.py", "gemm_micro_h100.csv")):
        wall, sites = dicts_of(gm / script)
        micro = {r["label"]: r for r in csv.DictReader(read_text(gm / csvname).splitlines())}
        for sysn, lst in sites.items():
            np_, total_o, diag_o, total_g, diag_g = wall[sysn]
            for label, calls, g8 in lst:
                r = micro[label]
                sites_rows.append({"machine": mach, "system": sysn, "ranks": np_, "site": label, "type": r["type"],
                                   "opA": r["opA"], "opB": r["opB"], "m": r["m"], "n": r["n"], "k": r["k"],
                                   "calls": calls, "g8_calls": g8, "cublas_fallback_calls": calls - g8,
                                   "cublas_ms_med": r["cublas_ms_med"], "g8_ms_med": r["g8_ms_med"],
                                   "s_gemm_med": r["s_gemm_med"], "relerr": r["relerr"], "stand_alone_csv": csvname,
                                   "total_cublas_s": total_o, "diag_cublas_s": diag_o, "total_g8_s": total_g, "diag_g8_s": diag_g,
                                   "counts_source": "run manifests gemmul8.d_calls/z_calls and the per-solve call structure (shapes_campaign.txt / shapes_h100.txt)"})
    write_csv(out / "amdahl_sites.csv", sites_rows, list(sites_rows[0].keys()))

    # ---------------- accuracy_runs.csv / accuracy_comparison.csv
    acc.sort(key=lambda r: (r["machine"], r["series"], r["case"], r["config_group"], int(r["rep"])))
    ACC_COLS = ["machine", "series", "run", "case", "config", "config_letter", "config_group", "rep", "retained", "retained_in",
                "ranks", "atoms", "scf_iters_out", "maxiter", "criterion", "converged_by_criterion", "last_normrd",
                "utot_printed", "fmax_printed", "d_calls", "z_calls", "d_fallbacks", "z_fallbacks", "dense_path", "binary_md5",
                "release_tag", "total_s", "diag_s", "out_file", "manifest_file"]
    write_csv(out / "accuracy_runs.csv", acc, ACC_COLS)
    cmp_rows = []
    by_case = {}
    for r in acc:
        by_case.setdefault((r["machine"], r["series"], r["case"]), []).append(r)

    def diff_rows(a, b, label):
        def sci(v):
            return "0.000e+00" if v == 0 else f"{v:.3e}"
        de = ""
        if a["utot_printed"] and b["utot_printed"]:
            de = sci(Decimal(a['utot_printed']) - Decimal(b['utot_printed']))
        df = ""
        if a["_forces"] and b["_forces"] and len(a["_forces"]) == len(b["_forces"]):
            df = sci(max(abs(Decimal(x) - Decimal(y)) for fa, fb in zip(a['_forces'], b['_forces']) for x, y in zip(fa, fb)))
        return {"machine": a["machine"], "series": a["series"], "case": a["case"], "pair": label,
                "run_a": a["run"], "run_b": b["run"], "dUtot_a_minus_b_Ha": de, "dF_max_Ha_per_Bohr": df,
                "scf_a": a["scf_iters_out"], "scf_b": b["scf_iters_out"],
                "converged_a": a["converged_by_criterion"], "converged_b": b["converged_by_criterion"],
                "criterion": a["criterion"], "maxiter": a["maxiter"],
                "z_fallbacks_a": a["z_fallbacks"], "z_fallbacks_b": b["z_fallbacks"],
                "retained_a": a["retained"], "retained_b": b["retained"]}
    for key, rs in sorted(by_case.items()):
        ret = [r for r in rs if r["retained"]]
        first = {}
        for r in sorted(ret, key=lambda r: int(r["rep"])):
            first.setdefault(r["config_letter"], r)
        for la, lb in (("C", "O"), ("G", "O"), ("C", "G")):
            if la in first and lb in first:
                cmp_rows.append(diff_rows(first[la], first[lb], f"{la}-{lb}"))
        for letter in ("O", "G", "C"):
            same = sorted([r for r in rs if r["config_letter"] == letter], key=lambda r: int(r["rep"]))
            for i in range(1, len(same)):
                cmp_rows.append(diff_rows(same[0], same[i], f"{letter}{same[0]['rep']}-{letter}{same[i]['rep']}"))
    write_csv(out / "accuracy_comparison.csv", cmp_rows, list(cmp_rows[0].keys()))

    # ---------------- strong_scaling_runs.csv / distributed_solver_runs.csv / supplementary
    SCAL_COLS = ["machine", "series", "run", "system", "nodes_label", "config", "config_group", "rep", "retained", "retained_in",
                 "ranks", "nodes", "threads", "env_node_count", "env_nodes", "total_s", "dft_s", "diag_s", "set_hamiltonian_s",
                 "scf_iters_out", "utot_printed", "d_calls", "z_calls", "d_fallbacks", "z_fallbacks", "dense_path",
                 "binary_md5", "release_tag", "source_commit", "exit_status", "out_file", "manifest_file"]
    scal.sort(key=lambda r: (r["system"], int(r["nodes_label"]), r["config"], int(r["rep"])))
    write_csv(out / "strong_scaling_runs.csv", scal, SCAL_COLS)
    DIST_COLS = ["machine", "series", "run", "nodes_label", "ranks_per_node_label", "config", "config_group", "rep", "retained",
                 "retained_in", "ranks", "nodes", "threads", "env_node_count", "env_nodes", "total_s", "dft_s", "diag_s",
                 "set_hamiltonian_s", "scf_iters_out", "utot_printed", "d_calls", "z_calls", "d_fallbacks", "z_fallbacks",
                 "dense_path", "eigen_lib_effective", "gs2_solves", "binary_md5", "release_tag", "source_commit",
                 "exit_status", "out_file", "manifest_file"]
    dist.sort(key=lambda r: (r["series"], int(r["nodes_label"]), r["ranks_per_node_label"], r["config"], r["rep"]))
    write_csv(out / "distributed_solver_runs.csv", dist, DIST_COLS)
    SUPP_COLS = ["machine", "series", "run", "retained", "retained_in", "ranks", "nodes", "threads", "atoms", "solver_input",
                 "spin_input", "kgrid", "eigen_lib_input", "total_s", "dft_s", "diag_s", "set_hamiltonian_s",
                 "scf_iters_out", "utot_printed", "d_calls", "z_calls", "d_fallbacks", "z_fallbacks", "dense_path",
                 "binary_md5", "release_tag", "source_commit", "exit_status", "out_file", "manifest_file"]
    supp.sort(key=lambda r: r["run"])
    write_csv(out / "supplementary_series_runs.csv", supp, SUPP_COLS)

    # ---------------- sha256 record and checks
    with open(out / "reaggregate_sha256.txt", "w") as f:
        for p in sorted(SHA):
            f.write(f"{SHA[p]}  {os.path.relpath(p, root)}\n")
    with open(out / "reaggregate_check.txt", "w") as f:
        f.write(f"archive root: {root.name}\n")
        f.write(f"runs: production {len(prod)} (retained {sum(r['retained'] for r in prod)}, paper_final {sum(r['paper_final'] for r in prod)}), "
                f"accuracy {len(acc)} (retained {sum(r['retained'] for r in acc)}), strong-scaling {len(scal)} (retained {sum(r['retained'] for r in scal)}), "
                f"distributed {len(dist)} (retained {sum(r['retained'] for r in dist)}), supplementary {len(supp)}\n")
        f.write(f"amdahl sites: {len(sites_rows)}; accuracy pairs: {len(cmp_rows)}; files hashed: {len(SHA)}\n")
        f.write(f"timing cross-check (.out Max_Time vs joblog TIMING vs manifest walls): {len(problems)} discrepancies\n")
        for p in problems:
            f.write("  " + p + "\n")
        f.write("\nproduction_summary rows with paper_final=1:\n")
        for s in summ:
            if s["paper_final"]:
                f.write(f"  {s['machine']:8s} {s['system']:8s} {s['config']:5s} np{s['ranks']:>3s} n={s['n_runs']} total {s['total_s_mean']}+-{s['total_s_sd']} "
                        f"diag {s['diag_s_mean']}+-{s['diag_s_sd']} S_cpu(tot/diag) {s.get('speedup_vs_cpu_total','')}/{s.get('speedup_vs_cpu_diag','')} "
                        f"S_cuB(tot/diag) {s.get('speedup_vs_cublas_total','')}/{s.get('speedup_vs_cublas_diag','')} "
                        f"MPSoff/on {s.get('mps_off_over_on_total','')}/{s.get('mps_off_over_on_diag','')}\n")
    print(open(out / "reaggregate_check.txt").read())


if __name__ == "__main__":
    main()
