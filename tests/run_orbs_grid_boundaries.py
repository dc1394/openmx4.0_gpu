#!/usr/bin/env python3
"""Exercise production orbital spline boundaries and host allocation fallbacks.

Only CUDA/OpenACC transfers and the input globals are stubbed. ASan/UBSan check
the extracted production functions, including first/last radial intervals.
"""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'source/Set_Orbitals_Grid.c').read_text()
helpers = source[source.index('typedef struct { float hi, lo; } SOG_df;'):
                 source.index('#pragma acc routine seq\nstatic void SOG_point_eval_df')]
prepare = source[source.index('int SOG_Device_Prepare(void)'):
                 source.index('\nconst char *SOG_Device_EvalPrecisionName')]
code = r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "orbs_grid_gpu.h"
#define SOG_L0MAX 3
static int fail_at, allocation, live, fail_device;
static void *checked_malloc(size_t n) {
  if (++allocation == fail_at) return NULL;
  void *p = malloc(n); assert(p); ++live; return p;
}
static void checked_free(void *p) { if (p) { --live; free(p); } }
static SOG_DeviceTables SOG_dev;
static int SpeciesNum = 1, TCpyCell = 0, Ngrid2 = 3, Ngrid3 = 3;
static int Spe_Num_Mesh_PAO[] = {6}, Spe_MaxL_Basis[] = {0};
static int nb[] = {1}, *Spe_Num_Basis[] = {nb};
static double rv[] = {0.1, 0.2, 0.4, 0.8, 1.6, 3.2};
static double rw[] = {1.1, -0.3, 0.7, 1.5, -0.2, 0.9};
static double *Spe_PAO_RV[] = {rv};
static double *mul[] = {rw}, **angular[] = {mul}, ***Spe_PAO_RWF[] = {angular};
static double atv[1][4], gtv[4][4], Grid_Origin[4];
static int SOG_gpu_eligible(int kind) { return 1; }
static int SOG_eval_double_float(void) { return 1; }
static size_t SOG_arena_off(size_t *pos, size_t bytes) {
  size_t off = *pos; *pos = (off + bytes + 511U) & ~(size_t)511U; return off;
}
static void *SOG_arena_try(size_t n) { return fail_device ? NULL : malloc(n); }
static void acc_memcpy_to_device(void *d, const void *s, size_t n) { memcpy(d,s,n); }
'''
code += helpers
code += '\n#define malloc checked_malloc\n#define free checked_free\n' + prepare
code += r'''
#undef malloc
#undef free
int main(void) {
  for (int failure = 1; failure <= 11; ++failure) {
    fail_at = failure; allocation = 0;
    assert(!SOG_Device_Prepare()); assert(!SOG_dev.ready); assert(!SOG_dev.arena);
    assert(live == 0);
  }
  fail_at = 0; allocation = 0; fail_device = 1;
  assert(!SOG_Device_Prepare()); assert(live == 0); assert(!SOG_dev.ready);
  fail_device = 0; allocation = 0;
  assert(SOG_Device_Prepare()); assert(live == 0); assert(allocation == 11);
  float *radial = malloc(12 * sizeof(float));
  assert(radial); memcpy(radial, SOG_dev.rwd, 12 * sizeof(float));
  /* Use the actual production mesh constants and radial hi/lo table. */
  for (int m = 1; m < 6; ++m) {
    for (int step = 0; step <= 10; ++step) {
      double t = step / 10.0, u = t - 1.0;
      const float *mc = SOG_dev.mcd + SOG_MCD_STRIDE * m;
      double h1 = m == 1 ? rv[0] - rv[2] : rv[m-1] - rv[m-2];
      double h2 = rv[m] - rv[m-1];
      double h3 = m == 5 ? rv[3] - rv[5] : rv[m+1] - rv[m];
      double f1 = rw[m == 1 ? m+1 : m-2], f2 = rw[m-1];
      double f3 = rw[m], f4 = rw[m == 5 ? m-2 : m+1];
      double p = h2 * ((f3-f2)*h1/h2/(h1+h2) + (f2-f1)*h2/h1/(h1+h2));
      double q = h2 * ((f4-f3)*h2/h3/(h2+h3) + (f3-f2)*h3/h2/(h2+h3));
      double reference = u*u*(3*f2+p+(2*f2+p)*u) + t*t*(3*f3-q-(2*f3-q)*t);
      SOG_df result = SOG_radial_df(radial, m, 6,
        SOG_df_mk(mc[2],mc[3]), SOG_df_mk(mc[4],mc[5]),
        SOG_df_mk(mc[6],mc[7]), SOG_df_mk(mc[8],mc[9]),
        SOG_df_mk(mc[10],mc[11]), SOG_df_dbl(t), SOG_df_dbl(u),
        SOG_df_dbl(t*t), SOG_df_dbl(u*u));
      assert(fabs((double)result.hi + result.lo - reference) < 2e-12);
    }
  }
  free(radial); free(SOG_dev.arena);
  puts("PASS: orbital end intervals, 11 allocation failures, device OOM, retry");
}
'''
with tempfile.TemporaryDirectory(prefix='openmx-orbs-test-') as tmp:
    test = Path(tmp) / 'check.c'
    exe = Path(tmp) / 'check'
    test.write_text(code)
    subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-O1', '-g',
                    '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                    '-fno-pie', '-no-pie', '-I' + str(root / 'source'),
                    str(test), '-lm', '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
