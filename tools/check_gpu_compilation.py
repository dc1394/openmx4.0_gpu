#!/usr/bin/env python3
"""Compile review-sensitive native units with a real SDK, without GPU hardware.

C files are compiled at -O0 so unreachable references cannot hide behind
optimization. HIP C uses host OpenMP here; CUDA/HIP .cu kernels are compiled
for the explicit architecture. This is an ABI/compile check, not a link or
device execution test.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("CUDA", "HIP"), required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--cc", required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--mpicc", default="mpicc")
    parser.add_argument("--arch", required=True, help="CUDA integer (e.g. 80) or HIP gfx942")
    parser.add_argument("--include", type=Path, action="append", default=[])
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build.resolve()
    subprocess.run([sys.executable, str(root/"tools/prepare_build.py"),
        "--source-root", str(root), "--build-root", str(build), "--backend", args.backend], check=True)
    source = build/"native/source"
    output = build/"compile-check"
    output.mkdir(exist_ok=True)
    mpi = shlex.split(subprocess.check_output([args.mpicc, "--showme:compile"], text=True))
    include = ["-I"+str(p) for p in [source, args.sdk.resolve()/"include",
        root/"source/third_party/fftw3/api", *args.include]]
    cc = shlex.split(args.cc) + ["-std=gnu11", "-O0", *mpi, *include]
    cxx = shlex.split(args.cxx) + ["-std=c++17", "-O0", *include]
    if args.backend == "HIP":
        cc += ["-D__HIP_PLATFORM_AMD__", "-fopenmp", "-Werror=implicit-function-declaration"]
        cxx += ["-x", "hip", "-D__HIP_PLATFORM_AMD__", "--rocm-path="+str(args.sdk.resolve()),
                "--offload-arch="+args.arch]
    else:
        cc += ["-acc", "-gpu=cc"+args.arch, "--diag_error=implicit_func_decl"]
        cxx += ["-arch=sm_"+args.arch]
    jobs = [(cc, n) for n in (
        "openmx_gpu_error.c", "openmx_gpu_workers.c", "openmx_globals.c", "Band_DFT_Col.c",
        "Divide_Conquer.c", "Divide_Conquer_LNO.c", "Mixing_H.c", "DIIS_Mixing_Rhok.c",
        "Kerker_Mixing_Rhok.c", "DIIS_Mixing_DM.c", "ADIIS_Mixing_DM.c",
        "GR_Pulay_DM.c", "Mixing_DM.c", "Simple_Mixing_DM.c", "Mixing_V.c", "Set_Vpot.c",
        "Set_Hamiltonian.c", "DFT.c", "Force.c", "Krylov.c", "Input_std.c", "outputfile1.c")]
    jobs += [(cxx, n) for n in (
        "openmx_gpu_dm.cu", "openmx_gpu_transform.cu", "openmx_mgpu_gemm.cu",
        "openmx_mgpu_runtime.cu")]

    def compile_unit(job):
        command, name = job
        flags = ["-fno-common"] if name == "openmx_globals.c" and args.backend=="HIP" else []
        result = subprocess.run(command+flags+["-c", str(source/name), "-o", str(output/(name+".o"))],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output/(name+".log")).write_text(result.stdout)
        if result.returncode:
            print(result.stdout, file=sys.stderr)
        return name, result.returncode

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(compile_unit, jobs))
    if any(code for _, code in results):
        print("FAILED:", ", ".join(name for name, code in results if code), file=sys.stderr)
        return 1
    for name in ("Force.c", "Divide_Conquer.c"):
        undefined = subprocess.check_output(["nm", "-u", str(output/(name+".o"))], text=True)
        for removed in ("Force_openacc_reduce_terms", "Eigen_lapack_openmp", "Eigen_lapack_openacc"):
            if removed in undefined:
                raise RuntimeError(name+": unexpected reference to "+removed)
    defined = subprocess.check_output(["nm", "-g", "--defined-only",
        str(output/"openmx_mgpu_runtime.cu.o")], text=True)
    for name in ("openmx_mgpu_solve_device_try", "openmx_mgpu_dispatch_dgemm",
                 "openmx_mgpu_dispatch_zgemm"):
        if " T "+name+"\n" not in defined:
            raise RuntimeError("runtime symbol must be strong: "+name)
    # Including the application header must no longer define COMMON/BSS data.
    probe = output/"declarations.c"
    probe.write_text('#include "openmx_common.h"\n')
    common_flag = ["-fno-common"] if args.backend=="HIP" else []
    subprocess.run(cc+common_flag+["-c", str(probe), "-o", str(output/"declarations.o")], check=True)
    declared = subprocess.check_output(["nm", "-g", "--defined-only",
        str(output/"declarations.o")], text=True)
    if any(len(parts:=line.split())==3 and parts[1] in "BCDGSR" for line in declared.splitlines()):
        raise RuntimeError("openmx_common.h still defines externally visible data")
    storage = subprocess.check_output(["nm", "-g", "--defined-only",
        str(output/"openmx_globals.c.o")], text=True)
    if " C " in storage or " B atomnum\n" not in storage:
        raise RuntimeError("application globals must have a single strong definition")
    print(f"PASS: {args.backend}, {len(jobs)} native units, O0 references and strong runtime symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
