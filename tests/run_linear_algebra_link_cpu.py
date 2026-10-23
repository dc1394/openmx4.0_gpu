#!/usr/bin/env python3
"""Exercise the production CUDA/HIP archive lists without an MPI or GPU SDK.

The small C libraries model symbol providers, not numerical kernels. ELPA
references redistribution symbols before the vendor archive is scanned, as
in the application. Explicit COSMA/COSTA calls must remain available while
ordinary ScaLAPACK calls must resolve to the vendor. Reintroducing COSTA's
unprefixed compatibility archive is an independent negative control.
"""
from pathlib import Path
import os
import re
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def libraries(makefile, prefix):
    lines = makefile.read_text().splitlines()
    starts = [(i, re.match(r"^GPUSOLVER2_LIBS\s*=\s*(.*)$", line))
              for i, line in enumerate(lines)]
    starts = [(i, match) for i, match in starts if match]
    if len(starts) != 1:
        raise RuntimeError("missing production solver archive list: " + str(makefile))
    index, match = starts[0]
    part, pieces = match[1].rstrip(), []
    while part.endswith("\\"):
        pieces.append(part[:-1])
        index += 1
        part = lines[index].rstrip()
    value = " ".join(pieces + [part])
    value = value.replace("$(GPUSOLVER2_PREFIX)", str(prefix))
    if "$" in value:
        raise RuntimeError("unhandled solver archive variable: " + value)
    arguments = shlex.split(value)
    if sys.platform == "darwin":
        # The model archives have no cycles. Preserve their order without
        # passing the native Linux linker's group markers to a Mach-O linker.
        arguments = [item for item in arguments
                     if item not in ("-Wl,--start-group", "-Wl,--end-group")]
    return arguments


def main():
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = shlex.split(os.environ.get("CFLAGS", ""))
    archiver = shlex.split(os.environ.get("AR", "ar"))
    providers = {"pdtran_": 11, "pdgemr2d_": 13, "pztranc_": 19,
                 "pztranu_": 23, "pzgemr2d_": 29}
    calls = "+".join(name + "()" for name in providers)
    declarations = "\n".join("extern int " + name + "(void);" for name in providers)
    with tempfile.TemporaryDirectory(prefix="openmx-linalg-link-") as temp:
        directory = Path(temp)
        prefix = directory / "prefix"
        lib = prefix / "lib"
        lib.mkdir(parents=True)

        def compile_unit(name, code):
            source = directory / (name + ".c")
            target = directory / (name + ".o")
            source.write_text(code)
            subprocess.run(compiler + flags + ["-x", "c", "-std=c11", "-c",
                           str(source), "-o", str(target)], check=True)
            return target

        def archive(name, units):
            path = lib / ("lib" + name + ".a")
            subprocess.run(archiver + ["rcs", str(path), *map(str, units)], check=True)
            return path

        archive("elpa", [compile_unit("elpa", declarations +
                "\nint elpa_boundary_probe(void){return " + calls + ";}\n")])
        archive("cosma_prefixed_pxgemm", [compile_unit("cosma_entry", """
extern int cosma_boundary_transform(void);
int cosma_pdgemm_(void){return 200+cosma_boundary_transform();}
""")])
        archive("cosma", [compile_unit("cosma", """
extern int costa_pdgemr2d_(void),pdgemm_(void);
int cosma_boundary_transform(void){return costa_pdgemr2d_()+pdgemm_();}
""")])
        archive("costa_prefixed_scalapack", [compile_unit("costa_prefix", """
extern int costa_boundary_copy(void);
int costa_pdgemr2d_(void){return costa_boundary_copy();}
""")])
        archive("costa", [compile_unit("costa", "int costa_boundary_copy(void){return 37;}\n")])
        archive("Tiled-MM", [compile_unit("tiled", "int tiled_boundary_unused(void){return 0;}\n")])
        compatibility = archive("costa_scalapack", [compile_unit("costa_compatibility",
            "\n".join("int " + name + "(void){return " + str(value + 100) + ";}"
                      for name, value in providers.items()))])
        # Separate archive members reproduce the vendor library's extraction
        # behavior; one combined object would instead create duplicate symbols.
        vendor_units = [compile_unit("vendor_" + name,
                        "int " + name + "(void){return " + str(value) + ";}\n")
                        for name, value in providers.items()]
        vendor_units.append(compile_unit("vendor_gemm", "int pdgemm_(void){return 17;}\n"))
        vendor_units.append(compile_unit("vendor_svd", declarations + """
extern int pdgemm_(void);
int pdgesvd_(void){return """ + calls + "+pdgemm_();}\n"))
        vendor = archive("vendor_scalapack", vendor_units)
        driver = compile_unit("driver", """
#include <stdio.h>
extern int elpa_boundary_probe(void),pdgesvd_(void),cosma_pdgemm_(void);
int main(void){
  const int elpa=elpa_boundary_probe(),svd=pdgesvd_(),cosma=cosma_pdgemm_();
  printf("providers: ELPA=%d SVD=%d explicit-COSMA=%d\\n",elpa,svd,cosma);
  return elpa!=95 || svd!=112 || cosma!=254;
}
""")
        for backend, makefile in [("CUDA", ROOT / "source/Makefile"),
                                  ("HIP", ROOT / "source/backends/hip/Makefile")]:
            production = libraries(makefile, prefix)
            target = directory / (backend + "-production")
            subprocess.run(compiler + flags + [str(driver), *production, str(vendor),
                           "-o", str(target)], check=True)
            result = subprocess.run([str(target)], text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(backend + " replaces a vendor symbol: " + result.stdout + result.stderr)
            print(backend, result.stdout.strip())
            negative = list(production)
            position = negative.index(str(lib / "libcosta_prefixed_scalapack.a"))
            negative.insert(position, str(compatibility))
            bad = directory / (backend + "-interposed")
            subprocess.run(compiler + flags + [str(driver), *negative, str(vendor),
                           "-o", str(bad)], check=True)
            result = subprocess.run([str(bad)], text=True, capture_output=True)
            if result.returncode != 1 or "ELPA=595 SVD=612 explicit-COSMA=254" not in result.stdout:
                raise RuntimeError(backend + " negative control did not detect provider interposition: " +
                                   result.stdout + result.stderr)
            print(backend, "negative control detects unprefixed COSTA interposition")
    print("PASS: both production archive lists preserve vendor and explicit accelerated providers")


if __name__ == "__main__":
    main()
