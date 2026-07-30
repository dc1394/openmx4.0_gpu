# Bundled source archives for the gpusolver2 stack

These archives are the pinned upstream sources from which
`source/third_party/build_gpusolver2_stack.sh` (invoked by
`make gpusolver2-stack` in `source/Makefile`) builds the libraries
behind `scf.eigen.lib=gpusolver2`, the distributed multi-GPU
diagonalization of the collinear/non-collinear cluster calculations.
The build is fully offline; nothing is downloaded.

| archive | contents |
|---------|----------|
| `slate-2025.05.28.tar.gz` | SLATE 2025.05.28 release tarball (BSD-3-Clause), including its bundled BLAS++, LAPACK++, and TestSweeper sources. <https://github.com/icl-utk-edu/slate/releases/tag/v2025.05.28> |

`SHA256SUMS` holds the checksums of every archive
(`sha256sum -c SHA256SUMS` verifies them).

The eigensolver of the older gpusolver keyword remains the ELPA
2018.05 subset embedded in the OpenMX sources; the SLATE stack is
self-contained and does not interact with it.
