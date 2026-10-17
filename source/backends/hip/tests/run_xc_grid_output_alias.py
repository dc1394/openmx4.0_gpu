#!/usr/bin/env python3
"""Check that the GPU XC download keeps the host rotation's store order.

The AMD port of the NVIDIA test of the same name.  The production
write-back block of Set_XC_Grid_HIP (source/set_xc_grid_hip.cu, from the
comment on "the host rotation's store order" up to the optional gradient
outputs) is extracted verbatim, with its SXG_CHECK macro, and built as host
C with hipMemcpy stubbed by memcpy (device-to-host only).  Checked: distinct
outputs; all four outputs aliased (the documented scalar call: equal spin
potentials give zero off-diagonals and Vxc0 must come last); the diagonal
and off-diagonal pairs aliased; the collinear store order with NULL
optional outputs; and a failing copy that ends the block through SXG_CHECK.
Only the store order is checked (the kernels are not: an alias between
Vxc0 and Vxc1 gives the host loops a different result anyway, since they
compute in place, see nv_density_xc_vpot verify #23).

Run: python3 tests/run_xc_grid_output_alias.py [--cc CC]
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = parser.parse_args()

    source = (ROOT / "source/set_xc_grid_hip.cu").read_text()
    start = source.index("    /* the host rotation's store order:")
    end = source.index("    if (dden_out != nullptr)", start)
    macro = re.search(r"^#define SXG_CHECK\(call\).*$", source, re.M)
    if macro is None:
        raise RuntimeError("SXG_CHECK not found")
    code = r'''
#include <assert.h>
#include <stdio.h>
#include <string.h>
typedef int hipError_t;
enum { hipSuccess = 0, hipErrorInvalidValue = 1 };
enum { hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };
static int copies, fail_at = -1;
static hipError_t hipMemcpy(void *out, const void *in, size_t bytes, int kind) {
  assert(out && in && kind == hipMemcpyDeviceToHost);
  if (copies++ == fail_at) return hipErrorInvalidValue;
  memcpy(out, in, bytes);
  return hipSuccess;
}
''' + macro.group(0) + r'''
static int download(int rotate, double *Vxc0, double *Vxc1, double *Vxc2, double *Vxc3,
                    const double *d_vxc0, const double *d_vxc1, const double *d_vxc2, const double *d_vxc3) {
  const size_t nb = 3 * sizeof(double);
  hipError_t status = hipSuccess;
''' + source[start:end] + r'''
  return 1;
done:
  (void)status;
  return 0;
}
int main(void) {
  const double up[3] = {-0.5, -0.25, -0.75}, down[3] = {-0.3, -0.15, -0.65};
  const double re[3] = {0.01, 0.02, 0.03}, im[3] = {0.04, 0.05, 0.06}, zero[3] = {0, 0, 0};
  double a[3], b[3], c[3], d[3];
  assert(download(1, a, b, c, d, up, down, re, im));
  assert(!memcmp(a, up, sizeof(a)) && !memcmp(b, down, sizeof(b)));
  assert(!memcmp(c, re, sizeof(c)) && !memcmp(d, im, sizeof(d)));
  /* the scalar-reference call aliases all four outputs; equal spin
     potentials give zero off-diagonal components */
  assert(download(1, a, a, a, a, up, up, zero, zero));
  assert(!memcmp(a, up, sizeof(a)));
  /* aliased off-diagonal and diagonal pairs keep the diagonal values, as
     the host's non-collinear rotation does */
  assert(download(1, a, b, a, b, up, down, re, im));
  assert(!memcmp(a, up, sizeof(a)) && !memcmp(b, down, sizeof(b)));
  /* the collinear and energy-density paths never touch the optional outputs */
  assert(download(0, a, b, NULL, NULL, up, down, NULL, NULL));
  assert(!memcmp(a, up, sizeof(a)) && !memcmp(b, down, sizeof(b)));
  /* a failing copy leaves the block through SXG_CHECK (the caller then runs
     the host loops, which write every output) */
  copies = 0; fail_at = 2;
  assert(!download(1, a, b, c, d, up, down, re, im));
  assert(copies == 3);
  puts("PASS: XC distinct outputs, scalar alias, pair aliases, optional outputs, failing copy");
  return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="openmx-xc-alias-") as tmp:
        cfile, exe = Path(tmp) / "check.c", Path(tmp) / "check"
        cfile.write_text(code)
        subprocess.run(shlex.split(args.cc) + ["-std=c11", "-O1", "-Wall", "-Wextra", str(cfile), "-o", str(exe)],
                       check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
