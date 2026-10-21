#!/usr/bin/env python3
"""Test the shared production CPU grid quadrature without a GPU or MPI.

The core tests compare float/double local, halo and contracted orbital views,
all spin modes, SIMD caps, odd shapes, tails, repeated grid points and varying
worker teams against independent long-double references. Set --build to reuse
an existing configured build (for example build/cuda or a sanitizer build).
"""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/grid-assembly-cpu")
    args = parser.parse_args()
    build = args.build.resolve()
    if not (build / "CMakeCache.txt").exists():
        subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build),
                        "-DOPENMX_BUILD_APPLICATION=OFF", "-DOPENMX_CORE_BACKEND=SERIAL",
                        "-DBUILD_TESTING=ON", "-DCMAKE_BUILD_TYPE=Release"], check=True)
    else:
        subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build)], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "openmx_grid_assembly_tests", "-j", "4"], check=True)
    subprocess.run(["ctest", "--test-dir", str(build), "-R", "^core-grid-assembly-",
                    "--output-on-failure"], check=True)


if __name__ == "__main__":
    main()
