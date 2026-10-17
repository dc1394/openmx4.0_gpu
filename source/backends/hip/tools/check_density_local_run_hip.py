#!/usr/bin/env python3
"""End-to-end check of the density's local mode on the GPU: prepare -> run.

source/Set_Density_Grid_GPU.c's SDG_local_prepare, SDG_local_build,
Set_Density_Grid_GPU_Local_Run, Set_Density_Grid_GPU_EndCycle and their
helpers are extracted verbatim and built with the OpenMP offload flags of
the program; source/set_hamiltonian_hip.cu (the double-float CSR kernel,
Set_Density_Grid_Hip_LocalDensity) is compiled with the HIP flags of the
makefile and linked in.  Synthetic Set_Hamiltonian tables in the layout of
the matrix-element tables (the central orbitals over each atom's whole
sphere, the neighbour orbitals per overlap point, spin-major DM blocks in
pair order) are put on the device the way Set_Hamiltonian makes them
resident (hipMalloc + omp_target_associate_ptr).  Only the eigensolver, the
table probe and the packing of CDM are stubbed.

Every case runs once with OPENMX_GRID_PRECISION=df and once with fp64, each
in its own process, over four SCF-like rounds: a fresh CSR and upload, the
resident buffers again (the double-float kernel's in-place split of the
device DM must be overwritten by the next upload), after
Set_Density_Grid_GPU_EndCycle (staged again from the host CSR), and once
more.  Checked: the density against a long double evaluation (double-float
5e-14, FP64 1e-15 of the per-point sum of the magnitudes), the same bits in
every round, and which kernel ran (the resident device DM is split only by
the double-float kernel).  With OPENMX_DENSITY_GRID_GPU_LOCAL_CACHE_MB=0
the buffers are staged per call instead.

Run from the repository root (one GPU process at a time, < 100 MiB of
device memory, a few seconds):
  python3 tools/check_density_local_run_hip.py [--hipcc HIPCC] [--clang CLANG] [--arch gfx1200]
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ROCM = Path("/opt/rocm-7.2.0")


def function(source, name):
    match = re.search(r"^[^\n;]*\b" + re.escape(name) + r"\([^;]*?\)\n\{", source, re.M)
    if not match:
        raise RuntimeError(f"function not found: {name}")
    pos = match.end()
    depth = 1
    while depth:
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
        pos += 1
    return source[match.start():pos]


def typedef(source, name):
    end = source.index("} " + name + ";") + len("} " + name + ";")
    return source[source.rfind("typedef struct {", 0, end):end]


PREFIX = r'''
#define _POSIX_C_SOURCE 200809L
#include <hip/hip_runtime_api.h>
#include <omp.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define GPUSOLVER 1
typedef float Type_Orbs_Grid;
static int SpinP_switch=0, Solver=3, Cnt_switch=0, scf_eigen_lib_flag=GPUSOLVER;
static int Matomnum=0, *M2G=NULL, *GridN_Atom=NULL;
/* eigen_refine_gpu.cu: an RDNA ratio (the precision comes from OPENMX_GRID_PRECISION) */
int openmx_gpu_fp32_fp64_ratio(void) { return 16; }
static int openmx_gpu_is_apu(void) { return 0; }
static int gpu_rank_device_usable(void) { return 1; }
static int OpenMX_GpuMemGetInfo(size_t *f,size_t *t) { return (int)hipMemGetInfo(f,t); }
static int Set_Density_Grid_OTF_Possible(int k) { (void)k; return 0; }
static int Set_Density_Grid_OTF_Run(int k,double *****C,double ***T) { (void)k; (void)C; (void)T; return 0; }
'''

STUBS = r'''
static SetHamiltonianMETables g_tables;
static double *g_dm;
static int Set_Hamiltonian_MatrixElementsTables_Ready(int k) { (void)k; return 1; }
static int Set_Hamiltonian_GetMatrixElementsTables(int k,SetHamiltonianMETables *t) { (void)k; *t=g_tables; return 1; }
static int SDG_pack_local_cdm(double *****CDM, double *buffer, size_t expected, int spin_count) {
  (void)CDM; (void)spin_count; memcpy(buffer,g_dm,sizeof(double)*expected); return 1;
}
'''

NAMES = ["SDG_env_bool", "SDG_env_mib", "SDG_gpu_default_enabled", "SDG_add_bytes", "SDG_malloc",
         "SDG_solver_supported", "SDG_local_delete_device", "SDG_local_upload", "SDG_local_free",
         "SDG_local_build", "SDG_local_call_bytes", "SDG_local_prepare", "Set_Density_Grid_GPU_EndCycle",
         "Set_Density_Grid_GPU_Local_Run"]

MAIN = r'''
static unsigned long long rng_state = 88172645463325252ULL;
static double urand(void) {   /* [-1, 1) */
  rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
  return (double)(rng_state >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}
/* resident as Set_Hamiltonian_ME_Associate makes a table */
static void associate(void *host, size_t bytes) {
  void *d = NULL;
  if (hipMalloc(&d, bytes) != hipSuccess || hipMemcpy(d, host, bytes, hipMemcpyHostToDevice) != hipSuccess ||
      omp_target_associate_ptr(host, d, bytes, 0, omp_get_default_device()) != 0) {
    fprintf(stderr, "cannot make a table resident\n"); exit(2);
  }
}
#define MAXP 4096
int main(int argc, char **argv) {
  const int S = atoi(argv[1]), atoms = atoi(argv[2]), nbr = atoi(argv[3]);
  const int species[3] = { atoi(argv[4]), 5, 19 };
  static int pair_Mc_AN[MAXP], pair_h_AN[MAXP], pair_NO0[MAXP], pair_NO1[MAXP], pair_NOLG[MAXP];
  static size_t hoff[MAXP], noff[MAXP], o0off[MAXP], o1off[MAXP];
  int *nolg_Nc = NULL, *nolg_MN = NULL, *no_atom, a, h, p = 0, s, fails = 0, df;
  float *orbs0 = NULL, *orbs1 = NULL;
  double *dm = NULL, ***Tmp, *ref, *mag, *first;
  size_t n_nc = 0, n_o0 = 0, n_o1 = 0, n_h = 0, out_total = 0;

  if (atoms * (nbr + 1) > MAXP) return 2;
  SpinP_switch = (S == 1) ? 0 : (S == 2 ? 1 : 3);
  Matomnum = atoms;
  M2G = calloc(atoms + 2, sizeof(int)); GridN_Atom = calloc(atoms + 2, sizeof(int)); no_atom = calloc(atoms + 2, sizeof(int));
  for (a = 1; a <= atoms; a++) {
    M2G[a] = a; GridN_Atom[a] = 800 + (int)(400 * (0.5 + 0.5 * urand())); no_atom[a] = species[(a - 1) % 3];
  }
  for (a = 1; a <= atoms; a++) {
    const int n = GridN_Atom[a], no = no_atom[a];
    const size_t o0base = n_o0;
    orbs0 = realloc(orbs0, sizeof(float) * (n_o0 + (size_t)n * no));
    for (int g = 0; g < n; g++) for (int i = 0; i < no; i++) orbs0[n_o0++] = (float)(urand() * exp(-2.0 * g / n));
    for (h = 0; h <= nbr; h++, p++) {
      const int no1 = (h == 0) ? no : species[(a + h) % 3];
      int cnt = 0;
      pair_Mc_AN[p] = a; pair_h_AN[p] = h; pair_NO0[p] = no; pair_NO1[p] = no1;
      hoff[p] = n_h; noff[p] = n_nc; o0off[p] = o0base; o1off[p] = n_o1;
      for (int g = 0; g < n; g++) {
        if (h != 0 && urand() < 0.3) continue;
        nolg_Nc = realloc(nolg_Nc, sizeof(int) * (n_nc + 1)); nolg_MN = realloc(nolg_MN, sizeof(int) * (n_nc + 1));
        nolg_Nc[n_nc] = g; nolg_MN[n_nc] = g; n_nc++; cnt++;
        orbs1 = realloc(orbs1, sizeof(float) * (n_o1 + no1));
        for (int j = 0; j < no1; j++) orbs1[n_o1++] = (float)urand();
      }
      pair_NOLG[p] = cnt;
      dm = realloc(dm, sizeof(double) * (n_h + (size_t)S * no * no1));
      for (s = 0; s < S; s++) for (int i = 0; i < no; i++) for (int j = 0; j < no1; j++)
        dm[n_h++] = 0.5 * urand() + ((h == 0 && i == j && s < 2) ? 1.0 : 0.0);
    }
  }
  g_dm = dm;
  memset(&g_tables, 0, sizeof(g_tables));
  g_tables.pair_count = p; g_tables.total_h = n_h; g_tables.total_nolg = n_nc;
  g_tables.total_orbs0 = n_o0; g_tables.total_orbs1 = n_o1;
  g_tables.pair_Mc_AN = pair_Mc_AN; g_tables.pair_h_AN = pair_h_AN; g_tables.pair_NO0 = pair_NO0;
  g_tables.pair_NO1 = pair_NO1; g_tables.pair_NOLG = pair_NOLG; g_tables.nolg_MN = nolg_MN; g_tables.nolg_Nc = nolg_Nc;
  g_tables.pair_h_offset = hoff; g_tables.pair_nolg_offset = noff;
  g_tables.pair_orbs0_offset = o0off; g_tables.pair_orbs1_offset = o1off;
  g_tables.orbs0buf = orbs0; g_tables.orbs1buf = orbs1;
  g_tables.orbs0_resident = g_tables.orbs1_resident = g_tables.nolg_resident = g_tables.meta_resident = 1;
  associate(pair_NO0, sizeof(int) * p); associate(pair_NO1, sizeof(int) * p);
  associate(hoff, sizeof(size_t) * p); associate(noff, sizeof(size_t) * p);
  associate(o0off, sizeof(size_t) * p); associate(o1off, sizeof(size_t) * p);
  associate(nolg_Nc, sizeof(int) * n_nc); associate(orbs0, sizeof(float) * n_o0); associate(orbs1, sizeof(float) * n_o1);

  /* the reference in long double, and per point the sum of the magnitudes */
  for (a = 1; a <= atoms; a++) out_total += (size_t)GridN_Atom[a];
  ref = calloc((size_t)S * out_total, sizeof(double)); mag = calloc((size_t)S * out_total, sizeof(double));
  first = calloc((size_t)S * out_total, sizeof(double));
  {
    size_t *base = calloc(atoms + 2, sizeof(size_t));
    long double *r = calloc((size_t)S * out_total, sizeof(long double));
    for (a = 2; a <= atoms + 1; a++) base[a] = base[a - 1] + (size_t)GridN_Atom[a - 1];
    for (int q = 0; q < p; q++) {
      const int n0 = pair_NO0[q], n1 = pair_NO1[q];
      for (int k = 0; k < pair_NOLG[q]; k++) {
        const size_t pt = noff[q] + (size_t)k, o = base[pair_Mc_AN[q]] + (size_t)nolg_Nc[pt];
        const float *phi0 = orbs0 + o0off[q] + (size_t)nolg_Nc[pt] * n0, *phi1 = orbs1 + o1off[q] + (size_t)k * n1;
        for (s = 0; s < S; s++) {
          const double *d = dm + hoff[q] + (size_t)s * n0 * n1;
          long double e = 0;
          double m = 0;
          for (int i = 0; i < n0; i++) {
            long double t = 0;
            for (int j = 0; j < n1; j++) {
              t += (long double)d[i * n1 + j] * phi1[j];
              m += fabs(d[i * n1 + j] * phi1[j] * phi0[i]);
            }
            e += (long double)phi0[i] * t;
          }
          r[(size_t)s * out_total + o] += e;
          mag[(size_t)s * out_total + o] += m;
        }
      }
    }
    for (size_t k = 0; k < (size_t)S * out_total; k++) ref[k] = (double)r[k];
    free(r); free(base);
  }
  Tmp = malloc(sizeof(double **) * S);
  for (s = 0; s < S; s++) {
    Tmp[s] = malloc(sizeof(double *) * (atoms + 1));
    for (a = 1; a <= atoms; a++) Tmp[s][a] = calloc(GridN_Atom[a], sizeof(double));
  }

  df = openmx_grid_precision_df();
  printf("S=%d atoms=%d pairs=%d points=%zu terms=%zu %s:", S, atoms, p, out_total, n_nc, df ? "df" : "fp64");
  for (int round = 0; round < 4; round++) {
    double maxe = 0.0;
    int split = -1;
    if (round == 2) Set_Density_Grid_GPU_EndCycle();
    if (!SDG_local_prepare(0, 0, 1)) { printf(" FAIL: prepare in round %d\n", round); return 1; }
    for (s = 0; s < S; s++) for (a = 1; a <= atoms; a++) memset(Tmp[s][a], 0, sizeof(double) * GridN_Atom[a]);
    if (!Set_Density_Grid_GPU_Local_Run(NULL, Tmp)) { printf(" FAIL: run in round %d\n", round); return 1; }
    for (s = 0; s < S; s++) {
      size_t off = (size_t)s * out_total;
      for (a = 1; a <= atoms; a++) for (int g = 0; g < GridN_Atom[a]; g++, off++) {
        const double x = Tmp[s][a][g], e = fabs(x - ref[off]) / (mag[off] > 0 ? mag[off] : 1.0);
        if (e > maxe) maxe = e;
        if (round == 0) first[off] = x;
        else if (memcmp(&first[off], &x, sizeof(double)) != 0 && fails++ < 3)
          printf(" FAIL: round %d differs from round 0", round);
      }
    }
    /* the double-float kernel leaves the resident device DM split */
    if (SDG_local.d_dm != NULL) {
      double *back = malloc(sizeof(double) * n_h);
      if (hipMemcpy(back, SDG_local.d_dm, sizeof(double) * n_h, hipMemcpyDeviceToHost) != hipSuccess) return 2;
      split = memcmp(back, dm, sizeof(double) * n_h) != 0;
      free(back);
    }
    printf(" [%d] %.1e%s", round, maxe, split < 0 ? " staged" : (split ? " split" : ""));
    if (split >= 0 && split != df) { printf(" FAIL: the %s kernel did not run", df ? "double-float" : "FP64"); fails++; }
    if (maxe > (df ? 5e-14 : 1e-15)) { printf(" FAIL: accuracy"); fails++; }
  }
  Set_Density_Grid_GPU_EndCycle();
  SDG_local_free();
  printf("\n");
  return fails ? 1 : 0;
}
'''

# (S, atoms, neighbours, orbitals of the first species): the register rows
# (16, 32, 64) and the memory variant (beyond 64 orbitals)
CASES = [(1, 6, 8, 13), (2, 5, 8, 13), (4, 4, 6, 19), (4, 3, 4, 48), (2, 3, 3, 80)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hipcc", default=str(ROCM / "bin/hipcc"))
    parser.add_argument("--clang", default=str(ROCM / "llvm/bin/clang"))
    parser.add_argument("--arch", default="gfx1200")
    args = parser.parse_args()

    source = (ROOT / "source/Set_Density_Grid_GPU.c").read_text()
    header = (ROOT / "source/openmx_common.h").read_text().replace("\r", "")
    decl_start = source.index("/* set_hamiltonian_hip.cu */")
    decl_end = source.index("typedef struct {", decl_start)
    parts = [PREFIX, typedef(header, "SetHamiltonianMETables"), source[decl_start:decl_end],
             typedef(source, "SDGLocalContext"), "static SDGLocalContext SDG_local = {0};", STUBS]
    parts += [function(source, name) for name in NAMES]
    parts.append(MAIN)

    with tempfile.TemporaryDirectory(prefix="openmx-density-local-") as tmp:
        tmp = Path(tmp)
        (tmp / "check.c").write_text("\n\n".join(parts))
        subprocess.run([args.hipcc, "-std=c++20", "-O3", "-fPIC", f"--offload-arch={args.arch}",
                        "-D__HIP_PLATFORM_AMD__", "-ffp-contract=off", "-DOCML_BASIC_ROUNDED_OPERATIONS",
                        "-Wno-unused-result", "-Wno-unused-value", "-I" + str(ROOT / "source"),
                        "-c", str(ROOT / "source/set_hamiltonian_hip.cu"), "-o", str(tmp / "kernels.o")], check=True)
        subprocess.run([args.clang, "-O3", "-std=c11", "-w", "-D__HIP_PLATFORM_AMD__", "-I" + str(ROCM / "include"),
                        "-fopenmp", f"--offload-arch={args.arch}", str(tmp / "check.c"), str(tmp / "kernels.o"),
                        "-L" + str(ROCM / "lib"), "-Wl,-rpath," + str(ROCM / "lib"), "-lamdhip64", "-lstdc++", "-lm",
                        "-o", str(tmp / "check")], check=True)
        failures = 0
        for cache in ("32", "0"):
            for mode in ("df", "fp64"):
                for case in CASES:
                    env = dict(os.environ, OPENMX_GRID_PRECISION=mode, OPENMX_DENSITY_GRID_GPU_LOCAL_CACHE_MB=cache)
                    run = subprocess.run([str(tmp / "check")] + [str(v) for v in case], env=env, timeout=120)
                    failures += run.returncode != 0
        if failures:
            raise SystemExit(f"FAILED: {failures} run(s)")
    print(f"PASS: {2 * 2 * len(CASES)} local-mode runs (double-float and FP64, resident and staged buffers)")


if __name__ == "__main__":
    main()
