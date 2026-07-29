# third_party/dist — bundled release archives for the gpusolver2 stack

`scf.eigen.lib gpusolver2` (distributed multi-GPU cluster diagonalization)
links ELPA (NVIDIA GPU kernels) and COSMA; this directory carries the official
release archives of everything the stack needs so that the build works without
any network access.  `source/third_party/build_gpusolver2_stack.sh` (driven by
`make gpusolver2-stack`, or implicitly by `make openmx`) unpacks and builds
them into `source/third_party/gpusolver2-install/`.

| archive | project | license | role |
|---|---|---|---|
| elpa-new_release_2026_02_002.zip | ELPA 2026.02.002 | LGPL-3 | distributed GPU eigensolver (modern C API; the elpa1/elpa2 keywords keep using the ELPA 2018.05 embedded in the source tree) |
| cosma-v2.8.4.zip | COSMA | BSD-3 | communication-optimal distributed GEMM (GPU backend) |
| COSTA-2484769….zip / Tiled-MM-0eb7517….zip | COSTA / Tiled-MM | BSD-3 | COSMA's grid transforms / GPU GEMM backend, pinned at the commits COSMA v2.8.4 references |
| cmake-3.31.12.tar.gz | CMake | BSD-3 | bootstrapped only when the system cmake is older than 3.24 (needed by COSMA) |

Building ELPA from the GitLab tag archive requires autoconf, automake,
libtool, m4, and python3 on the build machine (standard on Ubuntu).
`SHA256SUMS` lists the checksums of every archive.
