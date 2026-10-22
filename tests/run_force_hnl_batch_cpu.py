#!/usr/bin/env python3
"""Check native CUDA/HIP HNL batch preparation and teardown with real C++ owners.

The snapshot reference was recorded from ea8d1fad's native functions with this
same deterministic archive/density/geometry fixture. SDK operations are modeled;
device launch pragmas execute as host loops. Real device validation is separate.
"""
from pathlib import Path
import os
import resource
import shlex
import subprocess
import sys
import tempfile

from run_force4b_stream_cpu import extract_function

ROOT = Path(__file__).resolve().parents[1]


def program(hip, reference_source=None):
    adapter = (ROOT / "source/openmx_solver_workspace.h").read_text()
    begin = adapter.index("static inline OpenMXNativeResources *OpenMX_Solver_Resources(")
    end = adapter.index("\n}", begin) + 2
    harness = (ROOT / "tests/force_hnl_gpu_harness.cpp").read_text().replace(
        "/* OPENMX_GPU_RESOURCE_ADAPTER */", adapter[begin:end])
    harness = "static void batch_transfer();\n" + harness.replace("int main(){", "int archive_regression_unused_main(){")
    harness = harness.replace("std::memcpy(d,s,n);", "batch_transfer();std::memcpy(d,s,n);")
    extra = (ROOT / "tests/force_hnl_batch_harness.cpp").read_text()
    oracle = (ROOT / "tests/fixtures/hnl_before_capacity.c").read_text()
    contractions = "static void reference1(int nitems_c,const Force4BGpuItem *items,const size_t *krowA,const size_t *krowB,const int *krow_halo,const int *krow_ene,const int *krow_nlp,const double *cdm_pref,const double *halo,const double *ene,const double *flat,size_t flat_stride,double *item_f){\n"
    contractions += oracle.split("/* CASE 1 */\n")[1].split("/* CASE 2 */")[0] + "}\n"
    contractions += "static void reference2(int nitems_c,const Force4BGpuItem *items,const int *item_h0,const int *item_nlp,const int *item_ene,const size_t *item_a,const size_t *item_b,const double *cdm_pref,const double *c2,const double *ene,const double *flat,size_t c2_stride,double *item_f){\n"
    contractions += oracle.split("/* CASE 2 */\n")[1] + "}\n"
    source = reference_source if reference_source is not None else (ROOT / ("source/backends/hip/src/Force.c" if hip else "source/Force.c")).read_text()
    native = extract_function(source, "Force_gpu_arena_off")
    if reference_source is None:
        native += '#include "openmx_force_hnl_batch.h"\n'
        for kind in (1, 2):
            native += extract_function(source, f"Force_HNL_GpuCase{kind}Dispatch")
    else:
        for kind in (1, 2):
            native += extract_function(source, f"Force_HNL_GpuCase{kind}Run_Part")
    return harness + extra.replace("/* FROZEN_CONTRACTIONS */", contractions).replace("/* NATIVE_BATCH_FUNCTIONS */", native)


def bounded_stack():
    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, hard))


def main():
    library = Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="openmx-force-hnl-batch-") as directory:
        folder = Path(directory)
        for hip in (0, 1):
            source = folder / f"check{hip}.cpp"
            source.write_text(program(hip))
            exe = folder / f"check{hip}"
            command = shlex.split(os.environ.get("CXX", "c++")) + shlex.split(os.environ.get("CXXFLAGS", "")) + [
                "-std=c++20", "-O1", "-g", "-Wno-unknown-pragmas", "-fno-pie", "-no-pie",
                "-fsanitize=address,undefined", "-fno-sanitize-recover=all", f"-DOPENMX_GPU_HIP={hip}",
                "-I" + str(ROOT / "core/tests"), "-I" + str(ROOT / "core/tests/fixtures/native_resources"),
                "-I" + str(ROOT / "source"), "-I" + str(ROOT / "core/include"), "-I" + str(ROOT / "tests"),
                str(source), str(ROOT / "source/openmx_env.c"), str(ROOT / "core/tests/allocation_faults.cpp"),
                str(library), "-Wl,-rpath," + str(library.parent), "-lgomp", "-o", str(exe)]
            subprocess.run(command, check=True)
            subprocess.run([str(exe)], check=True, timeout=120, preexec_fn=bounded_stack)


if __name__ == "__main__":
    main()
