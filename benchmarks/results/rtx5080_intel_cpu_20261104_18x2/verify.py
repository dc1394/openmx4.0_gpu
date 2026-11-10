#!/usr/bin/env python3
"""Check the archived Intel CPU measurements without running OpenMX."""
import argparse
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import re

A = Path(__file__).resolve().parent
CASES = {"sidia333_nc_cluster": 216, "cghelgeo": 650}
TOLERANCE = Decimal("1e-7")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def physical_input(text):
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
            and line.split()[0].lower() not in {"data.path", "scf.eigen.lib"}]


def output_values(text, count):
    energy = Decimal(re.search(r"^\s*Utot\.\s+(\S+)", text, re.M)[1])
    marker = re.search(r"<coordinates.forces\s*\n\s*(\d+)\s*\n", text)
    assert int(marker[1]) == count
    rows = [line.split() for line in text[marker.end():].splitlines()[:count]]
    atoms = [(int(row[0]), row[1]) for row in rows]
    coordinates = [Decimal(value) for row in rows for value in row[2:5]]
    forces = [Decimal(value) for row in rows for value in row[5:8]]
    assert len(forces) == 3 * count
    assert all(value.is_finite() for value in [energy] + coordinates + forces)
    return energy, atoms, coordinates, forces


def quantum(value):
    return Decimal(1).scaleb(value.as_tuple().exponent)


def verify():
    manifest = json.loads((A / "manifest.json").read_text())
    for entry in manifest["files"]:
        data = (A / entry["path"]).read_bytes()
        assert len(data) == entry["bytes"], entry["path"]
        assert digest(data) == entry["sha256"], entry["path"]
        if "uncompressed_sha256" in entry:
            raw = gzip.decompress(data)
            assert digest(raw) == entry["uncompressed_sha256"], entry["path"]
            assert len(raw) == entry["uncompressed_bytes"], entry["path"]
    historical = json.loads((A / "historical-manifest.json").read_text())
    for name, entry in historical["files"].items():
        data = (A / name).read_bytes()
        assert digest(data) == entry["sha256"] and len(data) == entry["bytes"], name
    assert historical["source"]["source_commit"] == manifest["source_commit"]
    assert historical["protocol"]["MPI_processes"] == 18
    assert historical["protocol"]["OMP_threads_per_process"] == 2
    assert historical["protocol"]["MKL_threads"] == 2
    assert historical["protocol"]["repetitions"] == 1
    audit = json.loads((A / "input-data-audit.json").read_text())
    for entry in audit["inputs"]:
        case = entry["case"]
        assert digest((A / "inputs" / (case + ".dat")).read_bytes()) == entry["sha256"]
    exact, results, summaries = {}, {}, []
    for case, count in CASES.items():
        for variant in ("intel", "upstream"):
            folder = A / "runs" / case / variant
            result = json.loads((folder / "result.json").read_text())
            results[case, variant] = result
            assert result["status"] == "completed" and result["returncode"] == 0
            assert result["executable_sha256"] == historical["binary_sha256"][variant]
            input_data = (folder / "input.dat").read_bytes()
            assert digest(input_data) == result["input_sha256"]
            assert physical_input(input_data.decode()) == physical_input((A / "inputs" / (case + ".dat")).read_text())
            output = gzip.decompress((folder / (case + ".out.gz")).read_bytes())
            assert digest(output) == result["output_sha256"]
            energy, atoms, coordinates, forces = output_values(output.decode(), count)
            exact[case, variant] = energy, atoms, coordinates, forces
            assert float(energy) == result["energy_hartree"]
            assert [float(value) for value in forces] == result["forces"]
            assert len(result["atoms"]) == count
            log = gzip.decompress((folder / "run.log.gz").read_bytes()).decode()
            assert log.count("The calculation was normally finished.") == 1
            assert log.count("The SCF was achieved at MD=") == result["scf_converged_md"]
            assert log.count("The SCF was not fully achieved at MD=") == result["scf_unconverged_md"]
            steps = len(re.findall(r"MD=\s*1\s+SCF=\s*\d+", log))
            assert steps == result["scf_steps"]
            phases = {name.strip(): Decimal(high) for name, low, high in re.findall(
                r"^\s*([^=\n]+?)\s*=\s+\d+\s+([\d.]+)\s+\d+\s+([\d.]+)\s*$", log, re.M)}
            assert phases["Total Computational Time"] == Decimal(str(result["total_seconds"]))
            controls = result["controls"]
            assert controls["OMP_NUM_THREADS"] == controls["MKL_NUM_THREADS"] == "2"
            assert controls["OMP_DYNAMIC"] == controls["MKL_DYNAMIC"] == "FALSE"
            assert controls["OPENMX_GPU"] == "0"
            assert controls["MKL_THREADING_LAYER"] == "INTEL"
            assert result["command"][1:3] == ["-np", "18"]
            assert result["command"][-2:] == ["-nt", "2"]
            maps = gzip.decompress((folder / "rank-maps.txt.gz").read_bytes()).decode()
            for name in ("libmkl_intel_thread", "libiomp5.so", "/intel/oneapi/mpi/2021.18/"):
                assert name in maps, (case, variant, name)
            assert "libnvomp" not in maps and "libgomp" not in maps
            if case == "sidia333_nc_cluster":
                assert steps == 25 and result["scf_unconverged_md"] == 1
            else:
                assert result["scf_converged_md"] == 1 and result["scf_unconverged_md"] == 0
            summaries.append(dict(case=case, variant=variant, total_seconds=result["total_seconds"],
                                  wall_seconds=result["wall_seconds"], SCF_steps=steps,
                                  SCF_converged=result["scf_converged_md"] == 1))
    checks = []
    for case in CASES:
        upstream, intel = results[case, "upstream"], results[case, "intel"]
        assert upstream["input_sha256"] == intel["input_sha256"]
        assert upstream["grids"] == intel["grids"]
        ea, aa, ca, fa = exact[case, "upstream"]
        eb, ab, cb, fb = exact[case, "intel"]
        assert aa == ab
        coordinate = max(abs(x - y) for x, y in zip(ca, cb))
        de = abs(ea - eb)
        differences = [abs(x - y) for x, y in zip(fa, fb)]
        df = max(differences)
        eu = de + (quantum(ea) + quantum(eb)) / 2
        fu = max(d + (quantum(x) + quantum(y)) / 2 for d, x, y in zip(differences, fa, fb))
        assert eu <= TOLERANCE and fu <= TOLERANCE and coordinate <= Decimal("0.00001000001")
        checks.append(dict(case=case, passed=True, energy_difference_hartree=float(de),
                           max_force_component_difference_hartree_per_bohr=float(df),
                           energy_difference_rounding_upper_bound=float(eu),
                           force_difference_rounding_upper_bound=float(fu),
                           max_coordinate_difference=float(coordinate),
                           force_components=len(fa)))
    return dict(passed=True, measurements_reexecuted=False, source_commit=manifest["source_commit"],
                archived_files_checked=len(manifest["files"]), measurements=summaries,
                raw_output_comparisons=checks,
                tolerance=dict(energy_hartree=1e-7, force_component_hartree_per_bohr=1e-7),
                printed_rounding_included=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = verify()
    if args.write_report:
        (A / "archive-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
