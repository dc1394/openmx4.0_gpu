#!/usr/bin/env python3
"""Check the cached batch plans of the on-the-fly grid integrals on the CPU.

The plan code of source/Set_Hamiltonian.c (SetH_OTF_GetPlan and the
functions it uses, SetH_OTF_PackBlocks, the releases) is cut out of the
production file and compiled against host stubs of the HIP calls and a small
synthetic system; no device code runs.  The oracle rebuilds every batch from
the system arrays and checks: all atoms and pairs covered in order, the
batch arenas within the budget (or one atom), the descriptors (atoms, pairs,
point counts, the cumulative offsets inside a batch and across batches), the
overlap-point lists, the packed H/DM blocks, a second lookup with the same
key allocating nothing, a new key rebuilding, and the releases freeing every
host buffer (page-locked or pageable).  Both kinds (0 matrix elements, 1
density), preflight exact-fit/shortage/reserve/query-failure/ineligible/empty-rank
conditions, spin counts 1, 2, 4, and OPENMX_SETHAM_ONTHEFLY_PINNED_MB=0
(pageable only) and its default are run.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "source/Set_Hamiltonian.c").read_text()


def function(name):
    match = re.search(r"^[^\n;]*\b" + re.escape(name) + r"\([^;{]*?\)\n\{", SOURCE, re.M)
    if not match:
        raise RuntimeError("function not found: " + name)
    pos, depth = match.end(), 1
    while depth:
        if SOURCE[pos] == "{":
            depth += 1
        elif SOURCE[pos] == "}":
            depth -= 1
        pos += 1
    return SOURCE[match.start():pos]


def typedef(name):
    end = SOURCE.index("} " + name + ";") + len("} " + name + ";")
    return SOURCE[SOURCE.rfind("typedef struct {", 0, end):end]


def static_struct(name):
    end = SOURCE.index("} " + name + ";") + len("} " + name + ";")
    return SOURCE[SOURCE.rfind("static struct {", 0, end):end]


PREFIX = r'''
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int wan; int no; int pad0, pad1; double gx, gy, gz; double ax, ay, az; } OTF_Atom;

static void check(int ok, const char *msg) { if (!ok) { fprintf(stderr, "FAIL: %s\n", msg); exit(1); } }

/* host allocations, counted */
static long live_host = 0, live_pinned = 0, host_allocs = 0, pinned_calls = 0;
static void *t_malloc(size_t n) { void *p = malloc(n ? n : 1); check(p != NULL, "malloc"); live_host++; host_allocs++; return p; }
static void *t_calloc(size_t a, size_t b) { void *p = calloc(a ? a : 1, b ? b : 1); check(p != NULL, "calloc"); live_host++; host_allocs++; return p; }
static void t_free(void *p) { if (p != NULL) { live_host--; free(p); } }
#define hipSuccess 0
typedef int hipError_t;
static hipError_t hipHostMalloc(void **p, size_t n, unsigned flags) {
  (void)flags; pinned_calls++; *p = malloc(n ? n : 1); check(*p != NULL, "hipHostMalloc"); live_pinned++; host_allocs++; return hipSuccess; }
static hipError_t hipHostFree(void *p) { if (p != NULL) { live_pinned--; free(p); } return hipSuccess; }
static hipError_t hipFree(void *p) { (void)p; return hipSuccess; }
static hipError_t hipGetLastError(void) { return hipSuccess; }
#define malloc(n) t_malloc(n)
#define calloc(a, b) t_calloc(a, b)
#define free(p) t_free(p)

/* the synthetic system: 7 local atoms of 3 species, 2..4 neighbours each */
#define NA 7
static int Matomnum = NA, SpinP_switch=0, otf_possible=1, memory_query_ok=1;
static size_t free_device_bytes;
static int Set_Density_Grid_OTF_Possible(int k) { (void)k; return otf_possible; }
static int OpenMX_GpuMemGetInfo(size_t *f,size_t *t) { *f=*t=free_device_bytes; return memory_query_ok ? hipSuccess : 1; }
static int M2G[NA + 1], WhatSpecies[NA + 1], GridN_Atom[NA + 1], FNAN[NA + 1];
static int Spe_Total_NO[3] = {4, 9, 13};
static int natn[NA + 1][5], ncn[NA + 1][5], NumOLG[NA + 1][5];
static int glist[NA + 1][5][80], *GListTAtoms1[NA + 1][5];
static double Gxyz[NA + 1][4], atv[6][4];
static size_t test_budget[2];
static size_t Set_Hamiltonian_OTF_Budget(int kind) { return test_budget[kind == 1 ? 1 : 0]; }
#define SETH_OTF_ALIGN(x) (((x) + (size_t)255) & ~(size_t)255)
'''

MAIN = r'''
static double *****make_blocks(int spins)
{
  double *****m = (double *****)t_malloc(sizeof(double ****) * (size_t)spins);
  for (int s = 0; s < spins; s++) {
    m[s] = (double ****)t_malloc(sizeof(double ***) * (NA + 1));
    for (int a = 1; a <= NA; a++) {
      const int no0 = Spe_Total_NO[WhatSpecies[M2G[a]]];
      m[s][a] = (double ***)t_malloc(sizeof(double **) * (size_t)(FNAN[M2G[a]] + 1));
      for (int h = 0; h <= FNAN[M2G[a]]; h++) {
        const int no1 = Spe_Total_NO[WhatSpecies[natn[M2G[a]][h]]];
        m[s][a][h] = (double **)t_malloc(sizeof(double *) * (size_t)no0);
        for (int i = 0; i < no0; i++) {
          m[s][a][h][i] = (double *)t_malloc(sizeof(double) * (size_t)no1);
          for (int j = 0; j < no1; j++) m[s][a][h][i][j] = 1.0e6 * s + 1.0e4 * a + 1.0e3 * h + 31.0 * i + j + 0.5;
        }
      }
    }
  }
  return m;
}

static void free_blocks(double *****m, int spins)
{
  for (int s = 0; s < spins; s++) {
    for (int a = 1; a <= NA; a++) {
      const int no0 = Spe_Total_NO[WhatSpecies[M2G[a]]];
      for (int h = 0; h <= FNAN[M2G[a]]; h++) {
        for (int i = 0; i < no0; i++) t_free(m[s][a][h][i]);
        t_free(m[s][a][h]);
      }
      t_free(m[s][a]);
    }
    t_free(m[s]);
  }
  t_free(m);
}

static void check_plan(SetHOTFPlan *pl, int kind, int spins, double *****blocks)
{
  const SetHamOTFState *st = &Set_Hamiltonian_OTF_State;
  size_t dyn = 0, nc_at = 0, max_bounce = 0, arena = 0;
  int next_atom = 1;

  check(pl != NULL && pl->valid, "plan built");
  check(pl->kind == kind && pl->spin_count == spins && pl->matomnum == NA, "plan key");
  check(pl->batch_count > 1, "several batches");
  for (int b = 0; b < pl->batch_count; b++) {
    const SetHOTFBatch *bt = &pl->batch[b];
    const unsigned char *d = bt->desc;
    const OTF_Atom *ai = (const OTF_Atom *)(const void *)(d + bt->o_a_items);
    const OTF_Atom *pi = (const OTF_Atom *)(const void *)(d + bt->o_p_items);
    const int *anpts = (const int *)(const void *)(d + bt->o_a_npts), *pnpts = (const int *)(const void *)(d + bt->o_p_npts);
    const size_t *aooff = (const size_t *)(const void *)(d + bt->o_a_ooff), *asoff = (const size_t *)(const void *)(d + bt->o_a_soff);
    const size_t *pioff = (const size_t *)(const void *)(d + bt->o_p_ioff), *pooff = (const size_t *)(const void *)(d + bt->o_p_ooff);
    const size_t *psoff = (const size_t *)(const void *)(d + bt->o_p_soff), *hoff = (const size_t *)(const void *)(d + bt->o_hoff);
    const size_t *o0off = (const size_t *)(const void *)(d + bt->o_o0off);
    const int *no0v = (const int *)(const void *)(d + bt->o_no0), *no1v = (const int *)(const void *)(d + bt->o_no1);
    size_t h_at = 0, n_at = 0, o0_at = 0, o1_at = 0, den_at = 0, sph = 0;
    int ia = 0, ip = 0, max_no = 0, max_nolg = 0, max_sphere = 0;

    check(bt->a0 == next_atom && bt->a0 < bt->a1, "batches cover the atoms in order");
    next_atom = bt->a1;
    check(bt->na == bt->a1 - bt->a0, "batch atom count");
    check(bt->na == 1 || SetH_OTF_KindBatchBytes(kind, bt->a0, bt->a1, spins) <= test_budget[kind], "batch within budget");
    check(bt->dyn_first == dyn, "H/DM offset across batches");
    check(bt->nc_first == SetH_OTF_Static.atom_nc[bt->a0], "point-list offset across batches");
    check(bt->sph_first == st->atom_sph[bt->a0], "sphere offset across batches");
    for (int a = bt->a0; a < bt->a1; a++, ia++) {
      const int g = M2G[a], no0 = Spe_Total_NO[WhatSpecies[g]];
      const size_t atom_o0 = o0_at, atom_den = den_at;

      check(ai[ia].wan == WhatSpecies[g] && ai[ia].no == no0, "atom species");
      check(ai[ia].gx == Gxyz[g][1] && ai[ia].gy == Gxyz[g][2] && ai[ia].gz == Gxyz[g][3], "atom position");
      check(anpts[ia] == GridN_Atom[g] && aooff[ia] == o0_at && asoff[ia] == st->atom_sph[a], "atom points and offsets");
      if (max_no < no0) max_no = no0;
      if (max_sphere < GridN_Atom[g]) max_sphere = GridN_Atom[g];
      o0_at += (size_t)GridN_Atom[g] * (size_t)no0;
      den_at += (size_t)GridN_Atom[g];
      sph += (size_t)GridN_Atom[g];
      for (int h = 0; h <= FNAN[g]; h++, ip++) {
        const int g1 = natn[g][h], no1 = Spe_Total_NO[WhatSpecies[g1]], nolg = NumOLG[a][h];
        const int *nc = SetH_OTF_Static.nc + bt->nc_first + pioff[ip];

        check(pi[ip].wan == WhatSpecies[g1] && pi[ip].no == no1, "pair species");
        check(pi[ip].gx == Gxyz[g1][1] && pi[ip].ax == atv[ncn[g][h]][1] && pi[ip].az == atv[ncn[g][h]][3], "pair image");
        check(pnpts[ip] == nolg && pioff[ip] == n_at && pooff[ip] == o1_at && psoff[ip] == st->atom_sph[a], "pair points and offsets");
        check(no0v[ip] == no0 && no1v[ip] == no1 && hoff[ip] == h_at && o0off[ip] == atom_o0, "pair shape and block offsets");
        if (kind == 1) check(((const size_t *)(const void *)(d + bt->o_denoff))[ip] == atom_den, "density offset");
        for (int k = 0; k < nolg; k++) check(nc[k] == glist[a][h][k], "overlap points");
        if (blocks != NULL) {
          for (int s = 0; s < spins; s++)
            for (int i = 0; i < no0; i++)
              for (int j = 0; j < no1; j++)
                check(pl->dyn[bt->dyn_first + h_at + (size_t)s * no0 * no1 + (size_t)i * no1 + j] == blocks[s][a][h][i][j],
                      "packed H/DM block");
        }
        if (max_no < no1) max_no = no1;
        if (max_nolg < nolg) max_nolg = nolg;
        n_at += (size_t)nolg;
        o1_at += (size_t)nolg * (size_t)no1;
        h_at += (size_t)spins * (size_t)no0 * (size_t)no1;
      }
    }
    check(bt->np == ip && bt->bh == h_at && bt->bn == n_at && bt->bo0 == o0_at && bt->bo1 == o1_at, "batch sizes");
    check(bt->sph == sph && bt->max_no == max_no && bt->max_nolg == max_nolg && bt->max_sphere == max_sphere, "batch maxima");
    check(bt->o_nc <= bt->o_dyn && bt->o_dyn < bt->end, "arena layout");
    if (arena < bt->end) arena = bt->end;
    if (kind == 1 && max_bounce < (size_t)spins * sph) max_bounce = (size_t)spins * sph;
    dyn += h_at;
    nc_at += n_at;
  }
  check(next_atom == NA + 1, "every atom in a batch");
  check(pl->dyn_count == dyn && nc_at == SetH_OTF_Static.nc_count, "rank totals");
  check(pl->arena_bytes == arena, "arena = the largest batch");
  if (kind == 1) check(pl->bounce != NULL && pl->bounce_count == max_bounce, "density bounce buffer");
}

int main(void)
{
  const int expect_pinned = getenv("OPENMX_SETHAM_ONTHEFLY_PINNED_MB") == NULL;
  int checks = 0;

  for (int a = 1; a <= NA; a++) {
    M2G[a] = a;
    WhatSpecies[a] = a % 3;
    GridN_Atom[a] = 30 + 7 * a;
    FNAN[a] = 2 + a % 3;
    for (int j = 1; j <= 3; j++) Gxyz[a][j] = 0.37 * a + 0.11 * j;
    for (int h = 0; h <= FNAN[a]; h++) {
      natn[a][h] = (h == 0) ? a : 1 + (a + 2 * h) % NA;
      ncn[a][h] = (h * 3 + a) % 6;
      NumOLG[a][h] = (h == 0) ? GridN_Atom[a] : (a * 13 + h * 7) % GridN_Atom[a];
      GListTAtoms1[a][h] = glist[a][h];
      for (int k = 0; k < NumOLG[a][h]; k++) glist[a][h][k] = (k * 11 + h) % GridN_Atom[a];
    }
  }
  for (int r = 0; r < 6; r++) for (int j = 1; j <= 3; j++) atv[r][j] = 1.5 * r - 0.25 * j;
  test_budget[0] = 24 * 1024;
  test_budget[1] = 20 * 1024;

  for (int spins = 1; spins <= 4; spins *= 2) {
    double *****blocks = make_blocks(spins);
    long blocks_live = live_host;

    for (int kind = 0; kind < 2; kind++) {
      SetHamOTFState *st = &Set_Hamiltonian_OTF_State;
      SetHOTFPlan *pl, *again;
      long allocs;
      size_t sph = 0;

      /* the device state's host part (Set_Hamiltonian_OTF_Ensure) */
      st->ready = 1;
      st->matomnum = NA;
      st->atom_sph = (size_t *)t_malloc(sizeof(size_t) * (NA + 2));
      for (int a = 1; a <= NA; a++) { st->atom_sph[a] = sph; sph += (size_t)GridN_Atom[a]; }
      st->sph_total = sph;

      pl = SetH_OTF_GetPlan(kind, spins);
      check_plan(pl, kind, spins, NULL);
      if(kind==1) {
        size_t need=0, avail=0;
        const size_t reserve=(size_t)256*1024*1024;
        SpinP_switch=spins-1;
        free_device_bytes=reserve+pl->arena_bytes;
        check(Set_Density_Grid_OTF_Prepare(0,&need,&avail) && need==pl->arena_bytes && avail==need,"preflight exact fit");
        free_device_bytes--;
        check(!Set_Density_Grid_OTF_Prepare(0,&need,&avail) && need>avail,"preflight insufficient memory");
        free_device_bytes=reserve-1;
        check(!Set_Density_Grid_OTF_Prepare(0,&need,&avail) && avail==0,"reserve floor");
        memory_query_ok=0;
        check(!Set_Density_Grid_OTF_Prepare(0,&need,&avail),"failed memory query");
        memory_query_ok=1; otf_possible=0;
        check(!Set_Density_Grid_OTF_Prepare(1,&need,&avail),"ineligible density");
        otf_possible=1; Matomnum=0;
        check(Set_Density_Grid_OTF_Prepare(0,&need,&avail) && need==0 && avail==0,"empty partition");
        Matomnum=NA;
        checks+=6;
      }
      checks++;
      SetH_OTF_PackBlocks(pl, blocks);
      check_plan(pl, kind, spins, blocks);
      checks++;
      check(pl->dyn_pinned == expect_pinned && pl->batch[0].desc_pinned == expect_pinned &&
            SetH_OTF_Static.nc_pinned == expect_pinned, "page-locked buffers as configured");
      if (!expect_pinned) check(pinned_calls == 0, "no page-locked allocation with PINNED_MB=0");

      /* the same key: the cached plan, nothing allocated */
      allocs = host_allocs;
      again = SetH_OTF_GetPlan(kind, spins);
      check(again == pl && host_allocs == allocs, "cached lookup allocates nothing");
      checks++;

      /* the other kind beside it, then a new key for this kind */
      check_plan(SetH_OTF_GetPlan(1 - kind, spins), 1 - kind, spins, NULL);
      check(SetH_OTF_Plan[kind].valid && SetH_OTF_Plan[1 - kind].valid, "both kinds cached");
      test_budget[kind] += 4096;
      pl = SetH_OTF_GetPlan(kind, spins);
      check_plan(pl, kind, spins, NULL);
      test_budget[kind] -= 4096;
      checks += 2;

      /* the device-only release keeps the plans; the full one frees all */
      Set_Hamiltonian_OTF_ReleaseDevice();
      check(SetH_OTF_Plan[0].valid && SetH_OTF_Plan[1].valid && SetH_OTF_Static.ready, "device release keeps the plans");
      Set_Hamiltonian_OTF_Release();
      check(!SetH_OTF_Plan[0].valid && !SetH_OTF_Plan[1].valid && !SetH_OTF_Static.ready, "release drops the plans");
      check(live_pinned == 0 && live_host == blocks_live && SetH_OTF_PinnedBytes == 0, "release frees every buffer");
      checks += 2;
    }
    free_blocks(blocks, spins);
  }
  check(live_host == 0 && live_pinned == 0, "no leak");
  printf("PASS: %d plan checks (%s)\n", checks, expect_pinned ? "page-locked" : "pageable");
  return 0;
}
'''


def build_source():
    parts = [PREFIX,
             typedef("SetHamOTFState"),
             "static SetHamOTFState Set_Hamiltonian_OTF_State;",
             typedef("SetHOTFBatch"),
             typedef("SetHOTFPlan"),
             "static SetHOTFPlan SetH_OTF_Plan[2];",
             static_struct("SetH_OTF_Static"),
             "static size_t SetH_OTF_PinnedBytes = 0;"]
    for name in ["Set_Hamiltonian_OTF_BatchSizes", "SetH_OTF_HostAlloc", "SetH_OTF_HostFree", "SetH_OTF_FreePlan",
                 "SetH_OTF_ReleasePlans", "SetH_OTF_EnsureStatic", "SetH_OTF_KindBatchBytes", "SetH_OTF_GetPlan",
                 "Set_Density_Grid_OTF_Prepare", "SetH_OTF_PackBlocks", "Set_Hamiltonian_OTF_ReleaseDevice", "Set_Hamiltonian_OTF_Release"]:
        parts.append(function(name))
    parts.append(MAIN)
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--sanitize", action="store_true", help="build with -fsanitize=address,undefined")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "otf_plan.c"
        exe = Path(tmp) / "otf_plan"
        src.write_text(build_source())
        flags = ["-std=c11", "-O1", "-g", "-Wall", "-Wno-unused-function", "-Wno-unused-variable"]
        if args.sanitize:
            flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        subprocess.run([args.cc, *flags, str(src), "-o", str(exe)], check=True)
        for pinned in (None, "0"):
            env = dict(os.environ)
            env.pop("OPENMX_SETHAM_ONTHEFLY_PINNED_MB", None)
            if pinned is not None:
                env["OPENMX_SETHAM_ONTHEFLY_PINNED_MB"] = pinned
            subprocess.run([str(exe)], check=True, env=env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
