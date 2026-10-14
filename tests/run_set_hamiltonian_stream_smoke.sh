#!/bin/sh
set -eu
# Production CUDA/OpenACC workers versus a long-double quadrature oracle.
# SETHAMILTONIAN_SANITIZE=1 also checks device memory access with memcheck.
test_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec bash "$test_root/tests/run_gpu_quadrature_smoke.sh" set_hamiltonian_stream_smoke
