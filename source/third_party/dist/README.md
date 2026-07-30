# third_party/dist — bundled release archives for the gpusolver2 stack

`scf.eigen.lib gpusolver2` (distributed multi-GPU cluster diagonalization)
links DLA-Future (CUDA backend) and COSMA; this directory carries the official
release archives of everything the stack needs so that the build works without
any network access.  `source/third_party/build_gpusolver2_stack.sh` (driven by
`make gpusolver2-stack`, or implicitly by `make openmx`) unpacks and builds
them into `source/third_party/gpusolver2-install/`.

| archive | project | license | role |
|---|---|---|---|
| DLA-Future-0.10.0.zip | DLA-Future | BSD-3 | distributed GPU eigensolver (ScaLAPACK-like C API; the elpa1/elpa2 keywords keep using the ELPA 2018.05 embedded in the source tree) |
| pika-0.31.0.tar.gz | pika | BSL-1.0 | task runtime of DLA-Future (0.31 is the first release whose GPU stream counts DLA-Future can trim, which several ranks sharing one GPU need) |
| umpire-2025.12.0.tar.gz | Umpire | MIT | host/device memory pools of DLA-Future (camp/BLT bundled in the release tarball; 2025.12 is the first release that compiles against CUDA 13) |
| whip-0.3.0.tar.gz | whip | BSD-3 | CUDA/HIP abstraction used by DLA-Future (header-only) |
| blaspp-2025.05.28.tar.gz / lapackpp-2025.05.28.tar.gz | BLAS++ / LAPACK++ | BSD-3 | host BLAS/LAPACK C++ wrappers of DLA-Future |
| fmt-11.1.4.tar.gz | {fmt} | MIT | formatting library required by pika |
| spdlog-1.15.3.tar.gz | spdlog | MIT | logging library required by pika (built against the bundled {fmt}) |
| boost-1.87.0-b2-nodocs.tar.xz | Boost | BSL-1.0 | Boost headers + Boost.Context required by pika (only libboost_context is built) |
| cosma-v2.8.4.zip | COSMA | BSD-3 | communication-optimal distributed GEMM (GPU backend) |
| COSTA-2484769….zip / Tiled-MM-0eb7517….zip | COSTA / Tiled-MM | BSD-3 | COSMA's grid transforms / GPU GEMM backend, pinned at the commits COSMA v2.8.4 references |
| cmake-3.31.12.tar.gz | CMake | BSD-3 | bootstrapped only when the system cmake is older than 3.24 |

hwloc (needed by pika) is taken from the same HPC-X Open MPI installation
openmx links against, so it is not bundled here.
`SHA256SUMS` lists the checksums of every archive.
