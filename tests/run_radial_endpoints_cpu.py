#!/usr/bin/env python3
"""Exercise the real scalar interpolation functions at their final stencil."""
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
names = ("VH_AtomF", "Dr_VH_AtomF", "KumoF", "Dr_KumoF")
main = r'''#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
double VH_AtomF(double,int,double,double,double*,double*,double*);
double Dr_VH_AtomF(double,int,double,double,double*,double*,double*);
double KumoF(int,double,double*,double*,double*);
double Dr_KumoF(int,double,double,double*,double*,double*);
int main(void) {
  enum { N=6 };
  double xv[N],rv[N],*yv=malloc((N+2)*sizeof(double));
  assert(yv);
  for(int i=0;i<N;i++){xv[i]=-8.0+9.0*i/(N-1);rv[i]=exp(xv[i]);}
  for(int i=0;i<N+2;i++)yv[i]=(double)(i*i+1);
  const double r=nextafter(exp(1.0),0.0),x=0.5*log(r*r),tmp=(N-1)/9.0;
  assert(x<1.0 && (N-1)*(x+8.0)/9.0==N-1);
  const double slope=0.5*(yv[N+1]-yv[N-1])*tmp/r;
  assert(VH_AtomF(3.0,N,x,r,xv,rv,yv)==yv[N]);
  assert(Dr_VH_AtomF(3.0,N,x,r,xv,rv,yv)==slope);
  for(int i=0;i<3;i++) {
    const double query=i==0?x:(i==1?1.0:2.0);
    assert(KumoF(N,query,xv,rv,yv)==yv[N]);
    assert(Dr_KumoF(N,query,r,xv,rv,yv)==slope);
  }
  assert(VH_AtomF(3.0,N,1.0,r,xv,rv,yv)==3.0/r);
  assert(Dr_VH_AtomF(3.0,N,1.0,r,xv,rv,yv)==-3.0/r/r);
  free(yv);
  puts("PASS scalar/CUDA/HIP radial endpoints (rounded, exact, clamped, Coulomb)");
  return 0;
}
'''
# Compile the device and capacity-recovery copies as ordinary C as well.
# This executes their real arithmetic under ASan without a GPU runtime.
flat_functions, flat_checks = [], []
for label, relative in (("cuda", "source/Total_Energy_GPU.c"),
                        ("hip", "source/backends/hip/src/Total_Energy_GPU.c")):
    text = (root / relative).read_text()
    pattern = r"static double ((?:AMD_)?TotalEnergy_(?:EH0|Exc0)_(?:Dr_)?(?:VH_AtomF|KumoF)_flat)\("
    found = list(re.finditer(pattern, text))
    assert len(found) == (4 if label == "cuda" else 8)
    for match in found:
        begin = text.index("{", match.start())
        end, depth = begin + 1, 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        original = match[1]
        function = label + "_" + original
        flat_functions.append(text[match.start():end].replace(original, function))
        derivative = "_Dr_" in original
        expected = "slope" if derivative else "yv[N]"
        if "EH0" in original:
            args = "0,N,x,r,xv,rv,yv,&charge,N,N+2"
            flat_checks.append("  assert(" + function + "(" + args + ")==" + expected + ");")
        else:
            for query in ("x", "1.0", "2.0"):
                args = "N," + query + (",r" if derivative else "") + ",xv,rv,yv"
                flat_checks.append("  assert(" + function + "(" + args + ")==" + expected + ");")
main = "#include <math.h>\n" + "\n".join(flat_functions) + "\n" + main.replace(
    "  free(yv);", "  const double charge=3.0;\n" + "\n".join(flat_checks) + "\n  free(yv);")
with tempfile.TemporaryDirectory(prefix="openmx-radial-endpoints-") as folder:
    out = Path(folder)
    (out / "openmx_common.h").write_text("/* These four kernels use no globals. */\n")
    for name in names:
        (out / (name + ".c")).write_bytes((root / "source" / (name + ".c")).read_bytes())
    (out / "main.c").write_text(main)
    flags = [
        "-O1", "-g", "-fno-fast-math", "-ffp-contract=off",
        "-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer",
        *(["-fno-pie", "-no-pie"] if sys.platform.startswith("linux") else [])]
    # KumoF/Dr_KumoF now delegate to the independent Basis library. Compile
    # that actual implementation under the same sanitizers, keeping the C
    # bridge translation units (and their ABI) in this endpoint regression.
    objects = []
    for name in ("main", *names):
        obj = out / (name + ".o")
        subprocess.run([*shlex.split(os.environ.get("CC", "cc")), "-std=c11", *flags,
                        "-I" + str(root / "source"), "-c", str(out / (name + ".c")),
                        "-o", str(obj)], check=True)
        objects.append(str(obj))
    command = [*shlex.split(os.environ.get("CXX", "c++")), "-std=c++20", *flags,
               "-I" + str(root / "modules/basis/include"),
               str(root / "modules/basis/src/density_table.cpp"), *objects,
               "-lm", "-o", str(out / "check")]
    subprocess.run(command, check=True)
    subprocess.run([str(out / "check")], check=True, timeout=30)
