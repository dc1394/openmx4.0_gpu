#!/usr/bin/env python3
"""Check the production orbital policy for A100, GeForce, failed probes and overrides."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "source/Set_Orbitals_Grid.c").read_text()
start = source.index("static int SOG_eval_double_float(void)")
end = source.index("\nvoid SOG_Device_Release(void)", start)
code = r'''
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int ratio, get_error, attr_error, queries;
enum { cudaSuccess = 0, cudaDevAttrSingleToDoublePrecisionPerfRatio = 1 };
static int cudaGetDevice(int *device) { ++queries; *device=0; return get_error; }
static int cudaDeviceGetAttribute(int *value, int attribute, int device) {
  (void)attribute; (void)device; ++queries; *value=ratio; return attr_error;
}
'''
code += source[start:end]
code += r'''
int main(void) {
  unsetenv("OPENMX_ORBS_EVAL_PRECISION");
  ratio=2; assert(!SOG_eval_double_float()); /* A100 native FP64 */
  ratio=64; assert(SOG_eval_double_float()); /* GeForce */
  ratio=8; assert(SOG_eval_double_float());
  ratio=4; assert(!SOG_eval_double_float());
  get_error=1; assert(!SOG_eval_double_float()); get_error=0;
  attr_error=1; assert(!SOG_eval_double_float()); attr_error=0;
  const char *native[]={"fp64","FP64","64","double"};
  queries=0; ratio=64;
  for (int i=0;i<4;i++) {
    setenv("OPENMX_ORBS_EVAL_PRECISION",native[i],1); assert(!SOG_eval_double_float());
  }
  ratio=2;
  setenv("OPENMX_ORBS_EVAL_PRECISION","df",1); assert(SOG_eval_double_float());
  setenv("OPENMX_ORBS_EVAL_PRECISION","DF",1); assert(SOG_eval_double_float());
  assert(queries==0); /* Explicit overrides do not probe a device. */
  setenv("OPENMX_ORBS_EVAL_PRECISION","",1); assert(!SOG_eval_double_float());
  puts("PASS: orbital precision defaults, CUDA probe failures and explicit overrides");
}
'''
with tempfile.TemporaryDirectory(prefix="openmx-orbs-precision-") as tmp:
    cfile, binary = Path(tmp)/"check.c", Path(tmp)/"check"
    cfile.write_text(code)
    subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-O2", "-Wall", "-Wextra",
                    "-Werror", str(cfile), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
