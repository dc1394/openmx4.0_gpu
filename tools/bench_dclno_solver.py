#!/usr/bin/env python3
"""Prepare a bounded MPI benchmark of the production collinear DC-LNO solvers.

By default only generate the C harness and its Makefile. --build compiles it
against the existing OpenMX cuSOLVER/GEMMul8 objects and the source Makefile's
BLAS/toolkit settings; it never rebuilds OpenMX or launches MPI/GPU work.
Toolchain overrides must match the toolkit used to build those existing objects.
Run the generated executable separately with the same mpirun, CPU binding,
thread limits, CUDA visibility, and MPS environment as the OpenMX benchmark.

The matrices are deterministic synthetic SPD overlaps and symmetric dense
Hamiltonians, different on each rank. They are NOT production matrix dumps.
The measured functions, including matrix transfers and solver/GEMMul8 policy,
are extracted verbatim from Divide_Conquer_LNO.c. Synthetic crossover results
only motivate full OpenMX A/B experiments; they cannot select a new default.
GPU nonconvergence fails this timing harness rather than counting a CPU retry
as a GPU timing. The full OpenMX caller handles that exceptional retry.

Example (after the main benchmark queue is idle):
  python3 tools/bench_dclno_solver.py --build-dir /tmp/dclno-bench --build
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20m mpirun -np 8 \\
      /tmp/dclno-bench/bench_dclno_solver
  # Repeat with -np 18, using the benchmark's existing MPS/binding options.
  # Executable options: --n 600 --pairs half --warmup 2 --repeats 3
"""

import argparse
import hashlib
from pathlib import Path
import re
import shlex
import subprocess


def function(source, name):
    match = re.search(r"^static [^;{}]*\b" + re.escape(name) + r"\([^;]*?\)\s*\{",
                      source, re.MULTILINE)
    if match is None:
        raise ValueError(f"Cannot find production function {name}")
    brace = source.index("{", match.start())
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


PREFIX = r'''
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "openmx_common.h"
#include "lapack_prototypes.h"

/* Bound errors in the harness instead of retrying permanently failed calls.
   Successful production calls and their numerical work are unchanged. */
#undef wait_cudafunc
#define wait_cudafunc(call) do { int status_ = (int)(call); if (status_ != 0) { \
    fprintf(stderr, "DC-LNO benchmark: %s failed (%d) at line %d\n", \
            #call, status_, __LINE__); MPI_Abort(MPI_COMM_WORLD, 2); \
} } while (0)

static MPI_Comm DCLNO_gpu_group_comm = MPI_COMM_NULL;
static int DCLNO_gpu_group_rank, DCLNO_gpu_group_size, DCLNO_gpu_id;
static void DCLNO_AbortWithMessage(const char *message)
{
    fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 2);
}
'''


HARNESS = r'''
static double random_pair(int i, int j, int rank, uint64_t salt)
{
    uint64_t x = ((uint64_t)(i + 1) << 32) ^ (uint64_t)(j + 1);
    x ^= ((uint64_t)(rank + 1) * UINT64_C(0x9e3779b97f4a7c15)) ^ salt;
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return (double)(x >> 11) * 0x1.0p-53 - 0.5;
}

static void matrices(int n, int rank, double *S, double *H)
{
    /* Strict diagonal dominance gives SPD without an untimed O(n^3) setup. */
    memset(S, 0, sizeof(double) * (size_t)n * n);
    const double scale = 1.0 / sqrt((double)n);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < j; i++) {
            double s = 0.2 * scale * random_pair(i, j, rank, 17);
            S[(size_t)j*n+i] = S[(size_t)i*n+j] = s;
            S[(size_t)i*n+i] += fabs(s);
            S[(size_t)j*n+j] += fabs(s);
            H[(size_t)j*n+i] = H[(size_t)i*n+j] =
                2.0 * scale * random_pair(i, j, rank, 101);
        }
        H[(size_t)j*n+j] = -2.0 + 4.0 * (j + 0.5) / n;
    }
    for (int j = 0; j < n; j++) S[(size_t)j*n+j] += 1.0 + (double)j/n;
}

static double validate(int n, int k, const double *S, const double *H,
                       const double *w, double **C, const double *packed)
{
    const int selected[3] = {0, k/2, k-1};
    double hnorm = 0.0, snorm = 0.0, error = 0.0;
    double *sv = DCLNO_MallocArray((size_t)3*n, sizeof(double), "validation S*v");
    for (size_t i = 0; i < (size_t)n*n; i++) {
        hnorm += H[i]*H[i]; snorm += S[i]*S[i];
    }
    hnorm = sqrt(hnorm); snorm = sqrt(snorm);
    for (int q = 0; q < 3; q++) {
        int col = selected[q];
        const double *v = packed ? packed + (size_t)col*n : C[col+1]+1;
        double residual = 0.0, vnorm = 0.0;
        for (int i = 0; i < n; i++) {
            double hv = 0.0, sum = 0.0;
            for (int j = 0; j < n; j++) {
                hv += H[(size_t)j*n+i]*v[j];
                sum += S[(size_t)j*n+i]*v[j];
            }
            sv[(size_t)q*n+i] = sum;
            double r = hv - w[col+1]*sum;
            residual += r*r; vnorm += v[i]*v[i];
        }
        double scale = (hnorm + fabs(w[col+1])*snorm)*sqrt(vnorm);
        /* The valid scalar fixture H=[0] has both scale and residual zero.
           A nonzero residual with zero scale must still fail validation. */
        double rel = scale > 0.0 ? sqrt(residual) / scale
                                : (residual == 0.0 ? 0.0 : INFINITY);
        if (!isfinite(rel)) DCLNO_AbortWithMessage("Non-finite eigenpair residual.");
        if (rel > error) error = rel;
        for (int p = 0; p <= q; p++) {
            const double *u = packed ? packed + (size_t)selected[p]*n : C[selected[p]+1]+1;
            double product = 0.0;
            for (int i = 0; i < n; i++) product += u[i]*sv[(size_t)q*n+i];
            double defect = fabs(product - (selected[p] == col ? 1.0 : 0.0));
            if (!isfinite(defect)) DCLNO_AbortWithMessage("Non-finite eigenvector normalization.");
            if (defect > error) error = defect;
        }
    }
    free(sv);
    return error;
}

static int positive(const char *text)
{
    char *end;
    long value = strtol(text, &end, 10);
    if (!*text || *end || value < 1 || value > 8192)
        DCLNO_AbortWithMessage("Expected an integer in 1..8192.");
    return (int)value;
}

int main(int argc, char **argv)
{
    int rank, ranks, n_only = 0, pair_only = -1, warmup = 2, repeats = 3;
    const int dimensions[] = {400, 600, 800, 1024, 1536, 2100};
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--help")) {
            if (!rank) puts("Synthetic DC-LNO CPU/GPU benchmark: [--n N] [--pairs full|half] [--warmup N] [--repeats N]\n"
                            "Default dimensions 400,600,800,1024,1536,2100; full and half eigenpairs; 2 warmups, 3 measured solves.\n"
                            "Launch with the OpenMX benchmark's MPI binding, one BLAS thread, and MPS settings.\n"
                            "CPU reference, selected residuals/orthogonality, and production GPU memory gate are checked.\n"
                            "Use an external timeout; no production calculation or threshold is changed.");
            MPI_Finalize(); return 0;
        }
        if (a + 1 >= argc) DCLNO_AbortWithMessage("Missing option value.");
        if (!strcmp(argv[a], "--n")) n_only = positive(argv[++a]);
        else if (!strcmp(argv[a], "--warmup")) warmup = positive(argv[++a]);
        else if (!strcmp(argv[a], "--repeats")) repeats = positive(argv[++a]);
        else if (!strcmp(argv[a], "--pairs")) {
            a++;
            if (!strcmp(argv[a], "full")) pair_only = 0;
            else if (!strcmp(argv[a], "half")) pair_only = 1;
            else DCLNO_AbortWithMessage("--pairs requires full or half.");
        } else DCLNO_AbortWithMessage("Unknown benchmark option.");
    }

    int local_rank, devices;
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local_rank);
    wait_cudafunc(cudaGetDeviceCount(&devices));
    if (devices <= 0) DCLNO_AbortWithMessage("No visible CUDA device.");
    DCLNO_gpu_id = local_rank % devices;
    wait_cudafunc(cudaSetDevice(DCLNO_gpu_id));
    MPI_Comm_split(node, DCLNO_gpu_id, local_rank, &DCLNO_gpu_group_comm);
    MPI_Comm_rank(DCLNO_gpu_group_comm, &DCLNO_gpu_group_rank);
    MPI_Comm_size(DCLNO_gpu_group_comm, &DCLNO_gpu_group_size);
    if (!rank) {
        puts("# SYNTHETIC matrices; production CPU/GPU solve functions; not a production threshold calibration");
        printf("# ranks=%d visible_devices=%d warmup=%d repeats=%d\n", ranks, devices, warmup, repeats);
        const char *names[] = {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_MPS_PIPE_DIRECTORY",
                              "OPENMX_CUSOLVER_ALGORITHM", "OPENMX_GEMMUL8_DISABLE_D"};
        for (int i = 0; i < 5; i++) printf("# %s=%s\n", names[i], getenv(names[i]) ? getenv(names[i]) : "(unset)");
        puts("backend,n,requested,ranks,mean_rank_s,mean_slowest_rank_s,min_rank_s,max_rank_s,max_validation_error");
        fflush(stdout);
    }

    int failed = 0;
    for (int d = 0; d < (n_only ? 1 : 6); d++) {
        int n = n_only ? n_only : dimensions[d];
        size_t count = (size_t)n*n, bytes = count*sizeof(double);
        double *S0 = DCLNO_MallocArray(count, sizeof(double), "original S");
        double *H0 = DCLNO_MallocArray(count, sizeof(double), "original H");
        double *S = DCLNO_MallocArray(count, sizeof(double), "S");
        double *H = DCLNO_MallocArray(count, sizeof(double), "H");
        double *tmp = DCLNO_MallocArray(count, sizeof(double), "CPU temporary");
        double *eig = DCLNO_MallocArray(count, sizeof(double), "CPU eigenvectors");
        double *work = DCLNO_MallocArray(1u+6u*n+2u*count, sizeof(double), "CPU eig workspace");
        INTEGER *iwork = DCLNO_MallocArray(3u+5u*n, sizeof(INTEGER), "CPU integer workspace");
        double *w = DCLNO_MallocArray(n+1u, sizeof(double), "eigenvalues");
        double *ref = DCLNO_MallocArray(n+1u, sizeof(double), "reference eigenvalues");
        double **C = DCLNO_MallocArray(n+1u, sizeof(double*), "CPU vector pointers");
        double *vectors = DCLNO_MallocArray((size_t)(n+1)*(n+1), sizeof(double), "CPU vector values");
        for (int i = 0; i <= n; i++) C[i] = vectors + (size_t)i*(n+1);
        matrices(n, rank, S0, H0);
        for (int partial = 0; partial < 2; partial++) {
            if (pair_only >= 0 && pair_only != partial) continue;
            int k = partial && n > 1 ? n/2 : n;
            for (int gpu = 0; gpu < 2; gpu++) {
                if (gpu) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    int fits = DCLNO_GpuGroupMemoryFits(n, 0), all_fit;
                    MPI_Allreduce(&fits, &all_fit, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                    if (!all_fit) {
                        if (!rank) printf("# SKIP GPU n=%d requested=%d: production memory budget\n", n, k);
                        continue;
                    }
                }
                double sum = 0.0, slow_sum = 0.0, smallest = 1e300, largest = 0.0;
                for (int rep = -warmup; rep < repeats; rep++) {
                    memcpy(S, S0, bytes); memcpy(H, H0, bytes);
                    MPI_Barrier(MPI_COMM_WORLD);
                    double begin = MPI_Wtime();
                    if (gpu) {
                        int info = DCLNO_Solve_Col_GpuSolver(n, k, S, H, w);
                        if (info != 0)
                            DCLNO_AbortWithMessage("GPU solve did not converge; CPU retries are excluded from GPU timings.");
                    }
                    else DCLNO_Solve_Col_Local(n, k, S, H, C, w, tmp, eig, work, iwork);
                    double seconds = MPI_Wtime() - begin, slow;
                    MPI_Allreduce(&seconds, &slow, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                    if (rep >= 0) {
                        sum += seconds; slow_sum += slow;
                        if (seconds < smallest) smallest = seconds;
                        if (seconds > largest) largest = seconds;
                    }
                }
                double error = validate(n, k, S0, H0, w, C, gpu ? H : NULL);
                if (!gpu) memcpy(ref+1, w+1, sizeof(double)*k);
                else for (int i = 1; i <= k; i++) {
                    double e = fabs(w[i]-ref[i]) / fmax(1.0, fabs(ref[i]));
                    if (!isfinite(e)) DCLNO_AbortWithMessage("Non-finite eigenvalue comparison.");
                    if (e > error) error = e;
                }
                double total, minimum, maximum, max_error;
                MPI_Allreduce(&sum, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&smallest, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&largest, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                if (!rank) {
                    printf("%s,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.3e\n", gpu ? "GPU" : "CPU", n, k, ranks,
                           total/(ranks*repeats), slow_sum/repeats, minimum, maximum, max_error);
                    fflush(stdout);
                }
                if (max_error > 1e-9) failed = 1;
                if (gpu) {
                    openmx_gemmul8ReleaseWorkspaces();
                    DCLNO_GpuSolver_Destroy();
                }
            }
        }
        free(S0); free(H0); free(S); free(H); free(tmp); free(eig); free(work);
        free(iwork); free(w); free(ref); free(C); free(vectors);
    }
    MPI_Comm_free(&DCLNO_gpu_group_comm);
    MPI_Comm_free(&node);
    MPI_Finalize();
    return failed;
}
'''


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=root / "work/dclno_solver_bench")
    parser.add_argument("--build", action="store_true", help="compile using existing OpenMX objects; do not run")
    parser.add_argument("--make-option", action="append", default=[], metavar="NAME=VALUE",
                        help="source Makefile override; must match the existing objects' toolchain")
    args = parser.parse_args()
    if any(not re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", item) for item in args.make_option):
        parser.error("--make-option requires NAME=VALUE")
    source = (root / "source/Divide_Conquer_LNO.c").read_text()
    output = args.build_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    context_end = source.index("} DCLNO_GpuSolverCtx;") + len("} DCLNO_GpuSolverCtx;")
    context_start = source.rfind("typedef struct {", 0, context_end)
    before = ["DCLNO_CheckedArrayBytes", "DCLNO_CheckedMulCount", "DCLNO_MallocArray",
              "DCLNO_Eigen_lapack_d_reuse"]
    after = ["DCLNO_GpuSolver_Destroy", "DCLNO_GpuSolver_Init", "DCLNO_GpuSolver_EnsureMatrixCapacity",
             "DCLNO_GpuSolver_EnsureWorkspace", "DCLNO_GpuSolver_Eigen", "DCLNO_Solve_Col_GpuSolver",
             "DCLNO_Solve_Col_Local", "DCLNO_GpuGroupMemoryFits"]
    extracted = "\n\n".join(function(source, name) for name in before)
    extracted += "\n\n" + source[context_start:context_end]
    extracted += "\nstatic DCLNO_GpuSolverCtx DCLNO_gpusolver_ctx = {0};\n"
    extracted += "\n\n".join(function(source, name) for name in after)
    digest = hashlib.sha256(extracted.encode()).hexdigest()
    c_path = output / "bench_dclno_solver.c"
    c_path.write_text(f"/* Extracted production functions SHA256: {digest} */\n" + PREFIX + extracted + HARNESS)
    binary = output / "bench_dclno_solver"
    # This recipe intentionally has no object prerequisites: a benchmark build
    # must not trigger a rebuild of the production executable or its libraries.
    makefile = output / "bench.mk"
    makefile.write_text(".PHONY: dclno-crossover-bench\n"
                       "dclno-crossover-bench:\n"
                       "\t$(CC_NOACC_O3) -I$(CURDIR) " + shlex.quote(str(c_path)) +
                       " openmx_cusolver_compat.o gemmul8_bridge.o $(GEMMUL8_LIB) "
                       "$(NVPL_LIBS) $(MPI_FORTRAN_LIBS) -pgf90libs -mp -lpthread -lm -ldl "
                       "$(CUDA_LIBS) -lcublasLt -Wl,--no-as-needed -lcusparse -Wl,--as-needed "
                       "-lnvidia-ml -lstdc++ -latomic -o " + shlex.quote(str(binary)) + "\n")
    command = ["make", "-C", str(root / "source"), "-f", "Makefile", "-f", str(makefile),
               *args.make_option, "dclno-crossover-bench"]
    print(f"Prepared synthetic benchmark; production functions SHA256 {digest}")
    if args.build:
        for relative in ("source/openmx_cusolver_compat.o", "source/gemmul8_bridge.o"):
            if not (root / relative).is_file():
                parser.error(f"Existing production object required: {relative}; build OpenMX first")
        subprocess.run(command, check=True)
        print(f"Built {binary}; no benchmark was run.")
    else:
        print("No compilation or GPU run. Build command:\n" + shlex.join(command))
    print("Run separately with the established MPI/MPS/CPU-binding options, e.g.:\n"
          "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20m mpirun -np 8 " + shlex.quote(str(binary)))
    print("Repeat with -np 18. Use --n 600 --pairs half to isolate one case.")


if __name__ == "__main__":
    main()
