# openmx4.0_gpu
## What does this code do?
This is a GPU-accelerated version of [OpenMX](https://www.openmx-square.org/), a first-principles calculation code based on numerical atomic orbitals (NAO). The dense eigensolvers of the band calculations (collinear and non-collinear) and of the cluster calculations (collinear and non-collinear), the O(N)-type solvers (DC, DC-LNO, Krylov), and the heavy matrix-construction stages around them (Hamiltonian/overlap assembly, charge-density and orbital grids, forces, charge mixing) are GPU-accelerated. Since v2.0, the cluster diagonalization can also be distributed over multiple GPUs and multiple nodes with ELPA (GPU kernels) + COSMA; see below.

## Code author
Hiroyuki Kawai (Niigata Univ.)</br>
X account: [@dc1394](https://x.com/dc1394)

## How to enable GPU acceleration
GPU acceleration is enabled by default: "scf.eigen.lib" defaults to "gpusolver", so no extra input line is required. Writing it explicitly is of course still fine:

```ini
scf.XcType                  GGA-PBE    # LDA|LSDA-CA|LSDA-PW|GGA-PBE
scf.SpinPolarization        off        # On|Off|NC
scf.ElectronicTemperature  300.0       # default=300 (K)
scf.energycutoff           150.0       # default=150 (Ry)
scf.maxIter                 40         # default=40
scf.EigenvalueSolver       band        # DC|GDC|Cluster|Band
scf.Kgrid                  9 9 9       # means n1 x n2 x n3
scf.Mixing.Type           rmm-diisk    # Simple|Rmm-Diis|Gr-Pulay|Kerker|Rmm-Diisk
scf.Init.Mixing.Weight     0.300       # default=0.30
scf.Min.Mixing.Weight      0.001       # default=0.001 
scf.Max.Mixing.Weight      0.700       # default=0.40 
scf.Mixing.History          7          # default=5
scf.Mixing.StartPulay       5          # default=6
scf.criterion             1.0e-10      # default=1.0e-6 (Hartree) 
scf.eigen.lib             gpusolver     # default=gpusolver
```

To run the conventional CPU paths instead, specify "elpa2" or "elpa1":

```ini
scf.eigen.lib             elpa2         # CPU (ELPA2) paths
```

A run that finds no usable GPU demotes itself to ELPA2 automatically, so the default is also safe on machines without an NVIDIA GPU. The environment variable `OPENMX_GPU=0` forces the same demotion on a machine with a GPU, which is convenient for CPU-vs-GPU comparisons with unmodified input files.

## New in v2.0: distributed multi-GPU cluster diagonalization (ELPA GPU + COSMA)
A cluster calculation has only one k-point, so with the default "gpusolver" its dense eigenvalue problem is solved on a single GPU and, on a multi-GPU machine or a multi-node GPU cluster, the other GPUs idle during the diagonalization (see "Multi-GPU parallelization" below). Since v2.0, selecting

```ini
scf.EigenvalueSolver       cluster
scf.eigen.lib              gpusolver2    # ELPA (GPU kernels) + COSMA
```

replaces the cluster diagonalization (collinear and non-collinear) with [ELPA](https://elpa.mpcdf.mpg.de/) 2026.02 (NVIDIA GPU kernels) and [COSMA](https://github.com/eth-cscs/COSMA) (communication-optimal distributed matrix multiplication, GPU backend), engaging every MPI rank and every GPU. This is the option for machines where the diagonalization must not be confined to one GPU, e.g. one GPU per node × many nodes. Notes:

- "gpusolver2" supports only `scf.EigenvalueSolver cluster`; any other solver stops with an input error. Everything outside the cluster diagonalization behaves exactly like "gpusolver", and a machine without a usable GPU demotes itself to ELPA2 as usual.
- When the GPU memory cannot hold a solve, that solve automatically falls back to the ELPA CPU kernels / ScaLAPACK instead of aborting the run.
- The required libraries (ELPA, COSMA, COSTA, Tiled-MM) are bundled as source archives under `source/third_party/dist/` and are built automatically by the first `make` — no network access is needed (autoconf, automake, libtool, m4 and python3 must be installed). The conventional "elpa1"/"elpa2" paths keep using the ELPA 2018.05 embedded in the OpenMX source; only "gpusolver2" links the bundled ELPA 2026.02.

## GEMMul8: FP64 matrix multiplication on integer tensor cores
The large dense matrix multiplications of the GPU eigensolver path are executed through [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8), which emulates FP64 GEMM on the INT8 tensor cores using the Ozaki scheme II. This is enabled by default and is particularly effective on consumer GPUs (GeForce), whose native FP64 throughput is limited, while the total energy stays at the ~1e-10 Hartree agreement level in our tests. To compare with plain cuBLAS FP64 GEMM, it can be switched off (keyword added in v2.0):

```ini
scf.gemmul8.enable         off           # default=on
```

## Multi-GPU parallelization
How many GPUs a run can actually use is bounded by the number of k-points requested with "scf.Kgrid". The MPI ranks are divided into one group per k-point, and the dense eigenvalue problem of each group is solved on a single GPU, so the eigenvalue solver keeps at most as many GPUs busy as there are k-points; any GPU beyond that number stays idle in this part of the calculation. (The Hamiltonian matrix elements and the grid work are distributed over all MPI ranks, and therefore over all GPUs.)

In particular, a cluster calculation — `scf.EigenvalueSolver cluster`, i.e. `scf.Kgrid 1 1 1` — has only one k-point, so with the default "gpusolver" **only one GPU is used for the diagonalization however many GPUs the node has**. Adding GPUs, or raising the upper bound `scf.Gpu.Num` (default 30, which effectively means "use every GPU found"), does not make such a run faster. (A collinear spin-polarized calculation solves the two spins in separate MPI worlds, so it can occupy two GPUs at most.)

Multiple GPUs therefore pay off for band calculations with a k-mesh; for a cluster calculation, either give the job one GPU and more CPU cores / MPI ranks, or — since v2.0 — select `scf.eigen.lib gpusolver2` (see above), which distributes the cluster diagonalization itself over all ranks and all GPUs.

## MPI vs. hybrid (MPI/OpenMP) parallelization
Use flat MPI. In OpenMX 4.0 GPU, hybrid MPI/OpenMP parallelization is not effective: OpenMP threads can still be requested as usual with the `-nt` option and such runs complete correctly, but they bring no speedup — only MPI parallelization is effective. Assign all the cores you want to use to MPI ranks instead, with one OpenMP thread per rank:

```sh
mpirun -np 16 ./openmx input.dat -nt 1
```

## NVIDIA MPS: recommended whenever several ranks share a GPU
With flat MPI, all the MPI ranks of a node normally share one GPU — so run the
[NVIDIA CUDA Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/).
Without MPS the kernels of the different ranks are serialized by context
switching on the device; under MPS they execute concurrently.
The effect is large: with 48 ranks sharing one H100 PCIe, the `-runtest`
suite below completes in 108.4 s without MPS and 90.2 s with it (17% faster;
mean of two back-to-back A/B pairs on the same node), and the `-runtestL`
suite drops from 3684.7 s without MPS to 1481.6 s with it — 2.5x faster.
Without MPS the 48 time-sliced CUDA contexts make the GPU `-runtestL` run
even slower than the CPU-only run (2203.6 s), so on bigger systems MPS is
not a tweak but a requirement for the GPU to pay off.
Start the control daemon once per node before `mpirun`, and stop it afterwards:

```sh
nvidia-cuda-mps-control -d             # start the MPS control daemon
mpirun -np 48 ./openmx input.dat -nt 1
echo quit | nvidia-cuda-mps-control    # stop it
```

On a multi-node batch job, start one daemon on every node (`/tmp` is usually
node-local, so per-node `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY`
paths work well). MPS requires native Linux; it is not available under WSL2.
The benchmark tables below list the H100 GPU columns both without and with MPS.

## Build and install
Building and installing is more difficult than with standard OpenMX. The build requires the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) and OpenMPI. The Makefile contains build examples for several supercomputer systems (for the Pegasus supercomputer at the University of Tsukuba, a ready-made `Makefile.pegasus` is included); please refer to them. Since v2.0 the first `make` also builds the bundled ELPA/COSMA stack for "gpusolver2" automatically, which adds some time to the first build. A detailed implementation document (English and Japanese, including the list of GPU-related environment variables) is available under [doc/](doc/). If you're unsure about the build and installation process, feel free to ask in English via GitHub issues or [my X account](https://x.com/dc1394) (Japanese is also acceptable on my X account). I'll assist you as much as I can.

## Docker image
I have released the OpenMX 4.0 GPU Docker image.
You can easily try OpenMX 4.0 GPU on computers equipped with NVIDIA GPUs.
The steps are as follows:
1. Install [Docker](https://docs.docker.com/get-started/get-docker/).
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
3. Run: `docker run --gpus all --shm-size=4gb --rm -it -v /path/to/inputs:/work dc1394/openmx4.0-gpu-ubuntu24.04:1.1.1`. Ensure `/path/to/inputs` is created beforehand.
4. Run tests with `cd openmx_work` and `mpirun -np 4 ./openmx -runtest`.

This should yield results like the following:

```ini
   1  input_example/Benzene.dat        Elapsed time(s)=    6.16  diff Utot= 0.000000000001  diff Force= 0.000000000000
   2  input_example/C60.dat            Elapsed time(s)=   10.61  diff Utot= 0.000000000030  diff Force= 0.000000000002
   3  input_example/CO.dat             Elapsed time(s)=    7.95  diff Utot= 0.000000000000  diff Force= 0.000000000004
   4  input_example/Cr2.dat            Elapsed time(s)=    8.37  diff Utot= 0.000000000000  diff Force= 0.000000000003
   5  input_example/Crys-MnO.dat       Elapsed time(s)=   13.44  diff Utot= 0.000000000013  diff Force= 0.000000000002
   6  input_example/GaAs.dat           Elapsed time(s)=   20.83  diff Utot= 0.000000000023  diff Force= 0.000000000001
   7  input_example/Glycine.dat        Elapsed time(s)=    4.93  diff Utot= 0.000000000001  diff Force= 0.000000000000
   8  input_example/Graphite4.dat      Elapsed time(s)=    3.76  diff Utot= 0.000000000002  diff Force= 0.000000000001
   9  input_example/H2O-EF.dat         Elapsed time(s)=    4.69  diff Utot= 0.000000000001  diff Force= 0.000000000001
  10  input_example/H2O.dat            Elapsed time(s)=    4.11  diff Utot= 0.000000000000  diff Force= 0.000000000001
  11  input_example/HMn.dat            Elapsed time(s)=   12.77  diff Utot= 0.000000000000  diff Force= 0.000000000000
  12  input_example/Methane.dat        Elapsed time(s)=    3.43  diff Utot= 0.000000000058  diff Force= 0.000000000001
  13  input_example/Mol_MnO.dat        Elapsed time(s)=    8.31  diff Utot= 0.000000000001  diff Force= 0.000000000000
  14  input_example/Ndia2.dat          Elapsed time(s)=    4.61  diff Utot= 0.000000000004  diff Force= 0.000000000000


Total elapsed time (s)      113.97
```

You can verify that the calculation is correct.

## Benchmarks
For benchmarks of GPU-accelerated OpenMX, please refer to the following literature.
https://journals.jps.jp/doi/10.7566/JPSJ.94.124003

However, the current version offers improved performance compared to the version described in this paper.

### Built-in test suites (-runtest / -runtestL)
The two standard OpenMX test suites were run on two machines, both with the NVIDIA HPC SDK 26.5 (CUDA 13.2) and flat MPI:

- **PC** — Core i9-10980XE (18 cores) + GeForce RTX 5080 (16 GB), 18 ranks sharing the single GPU;
- **Pegasus (CCS, Univ. of Tsukuba) node** — Xeon Platinum 8468 (48 cores) + H100 PCIe (80 GB), 48 ranks sharing the single GPU; the GPU suite was run twice, without CUDA MPS (time-sliced contexts) and with it (see above).

On each machine the same binary was used for both columns: `OPENMX_GPU=0` demotes the whole run to the CPU (ELPA2) paths (the Pegasus CPU jobs additionally had no GPU allocated at all), and the GPU runs use the input-file defaults (`scf.eigen.lib gpusolver`, GEMMul8 on):

```sh
# GPU (defaults; GEMMul8 enabled); N = 18 on the PC, 48 on the Pegasus node.
# On the Pegasus node the GPU suites were run twice: with the MPS daemon up
# ("MPS" columns) and without it ("no MPS" columns).
mpirun -np N ./openmx -runtest  -nt 1
mpirun -np N ./openmx -runtestL -nt 1
# CPU reference (same binary)
OPENMX_GPU=0 mpirun -np N ./openmx -runtest  -nt 1
OPENMX_GPU=0 mpirun -np N ./openmx -runtestL -nt 1
```

`-runtest` (14 small systems, 2–60 atoms; elapsed seconds from runtest.result):

| input | i9 CPU (s) | 5080 GPU (s) | Xeon CPU (s) | H100 GPU, no MPS (s) | H100 GPU, MPS (s) |
|---|---:|---:|---:|---:|---:|
| Benzene | 6.14 | 7.41 | 13.63 | 15.92 | 11.78 |
| C60 | 10.29 | 9.96 | 7.36 | 15.33 | 6.35 |
| CO | 7.97 | 9.13 | 7.25 | 8.34 | 7.47 |
| Cr2 | 8.07 | 7.96 | 8.31 | 8.44 | 7.90 |
| Crys-MnO | 13.07 | 9.73 | 10.84 | 7.08 | 6.44 |
| GaAs | 21.37 | 13.73 | 16.36 | 9.45 | 9.28 |
| Glycine | 4.86 | 5.38 | 4.65 | 5.51 | 4.59 |
| Graphite4 | 3.91 | 2.86 | 4.16 | 3.58 | 3.30 |
| H2O-EF | 4.49 | 4.49 | 4.79 | 4.87 | 4.54 |
| H2O | 3.85 | 3.88 | 4.37 | 5.21 | 4.47 |
| HMn | 12.82 | 11.26 | 10.53 | 10.33 | 10.00 |
| Methane | 3.15 | 3.16 | 3.64 | 3.81 | 3.44 |
| Mol_MnO | 8.17 | 7.42 | 7.50 | 7.27 | 6.99 |
| Ndia2 | 4.64 | 2.60 | 5.01 | 3.31 | 3.04 |
| **Total** | **112.80** | **98.94** | **108.41** | **108.45** | **89.59** |

These systems are far below the GPU/CPU switching thresholds of the dense eigensolvers, so the diagonalization automatically falls back to the CPU and only the GPU-accelerated matrix-construction stages differ — the point of this table is that the whole suite passes on the GPU build with the same accuracy as the CPU paths (max diff Utot ≤ 5.5e-11 Hartree in all five columns). On these tiny systems the GPU pays off only under MPS (108.45 s → 89.59 s); the elevated first one or two cases of each Pegasus column are the one-time warm-up of the batch node (input-file cache, CUDA context creation).

`-runtestL` (16 medium/large systems; each "ratio" column is CPU/GPU on the same machine — for the H100, CPU / MPS-on GPU):

| input | atoms | solver | i9 CPU (s) | 5080 GPU (s) | ratio | Xeon CPU (s) | H100 GPU, no MPS (s) | H100 GPU, MPS (s) | ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5_5_13COb2 | 155 | band | 106.33 | 78.80 | 1.35 | 52.66 | 65.14 | 35.60 | 1.48 |
| B2C62_Band | 64 | band | 704.51 | 540.50 | 1.30 | 320.14 | 878.20 | 157.35 | 2.03 |
| CG15c-DC-LNO | 650 | dc-lno | 161.87 | 136.06 | 1.19 | 65.92 | 95.57 | 49.83 | 1.32 |
| DIA512-1 | 512 | krylov | 184.10 | 185.44 | 0.99 | 68.28 | 353.95 | 56.04 | 1.22 |
| FeBCC | 16 | band (sp) | 183.65 | 190.18 | 0.97 | 78.78 | 83.68 | 66.27 | 1.19 |
| GEL | 40 | band | 56.58 | 52.34 | 1.08 | 31.23 | 52.37 | 22.00 | 1.42 |
| GFRAG | 54 | cluster | 45.10 | 43.25 | 1.04 | 23.22 | 41.56 | 15.50 | 1.50 |
| GGFF | 40 | band (NC) | 1657.70 | 1304.17 | 1.27 | 573.69 | 591.56 | 332.88 | 1.72 |
| MCCN | 564 | krylov | 313.57 | 327.93 | 0.96 | 123.87 | 176.05 | 95.15 | 1.30 |
| Mn12_148_F | 148 | cluster (sp) | 121.04 | 88.59 | 1.37 | 57.85 | 70.06 | 29.04 | 1.99 |
| N1C999 | 1000 | dc-lno (sp) | 1663.46 | 1568.67 | 1.06 | 489.76 | 828.62 | 458.58 | 1.07 |
| Ni63-O64 | 127 | band (sp) | 104.33 | 68.53 | 1.52 | 53.18 | 112.07 | 24.09 | 2.21 |
| Pt63 | 63 | cluster | 83.42 | 60.11 | 1.39 | 31.42 | 74.37 | 23.43 | 1.34 |
| SialicAcid | 40 | cluster | 25.57 | 23.05 | 1.11 | 14.33 | 32.42 | 12.96 | 1.11 |
| ZrB2_2x2 | 76 | band | 307.99 | 222.72 | 1.38 | 137.12 | 142.96 | 68.31 | 2.01 |
| nsV4Bz5 | 64 | cluster | 138.14 | 115.06 | 1.20 | 82.15 | 86.10 | 34.59 | 2.37 |
| **Total** | | | **5857.35** | **5005.41** | **1.17** | **2203.59** | **3684.69** | **1481.62** | **1.49** |

All 16 inputs pass on the GPU (GEMMul8 on) on both machines — max diff Utot = 2.0e-9 Hartree on the RTX 5080 and 2.3e-9 on the H100 (identical with and without MPS), the same order as the official CPU reference results bundled in `work/large_example/runtestL.result_*` (whose largest deviation is also on Pt63, the case that reaches 2.3e-8 in our 48-rank CPU reference column). The no-MPS column is the "NVIDIA MPS" section above in numbers: 48 time-sliced CUDA contexts drag the suite to 3684.69 s — 2.5x the MPS-on time, slower than the CPU-only run, with per-case penalties up to 6.3x (DIA512-1) — while accuracy is unaffected. On the larger inputs the dense band/cluster diagonalizations run on the GPU through GEMMul8; when many ranks share one GPU, some construction stages transiently fall back to the CPU where the device-memory preflight says they do not fit (by design — the run continues and stays correct; this happens on the 16 GB RTX 5080 and, at 48 ranks, even on the 80 GB H100). Keep in mind that these test inputs are correctness tests, not performance showcases: they are small-to-medium systems dominated by stages other than the dense diagonalization, which is where the GPU gains the most. The speedup grows with the system size (see "Important notes" below), and calculations with hundreds of atoms and a dense solver benefit far more than the 1.17x / 1.49x totals above.

## Important notes
At present, GPU-accelerated OpenMX performs faster than standard OpenMX for calculations involving systems containing hundreds of atoms. For calculations involving systems with fewer than a hundred atoms, standard OpenMX should be used (or set `scf.eigen.lib elpa2` to run the CPU paths of this code). Please use with caution as it may contain bugs.

## About bug reports
I would appreciate it if you could actively report any bugs. Please report them via GitHub issues or send them to [my X account](https://x.com/dc1394). Bug reports sent to my X account can be in English.

## License

This project is a derivative work of **OpenMX** and is licensed under the
**GNU General Public License v3.0 or later (GPL-3.0-or-later)**, consistent
with the upstream OpenMX license. See the [LICENSE](LICENSE) file for the
full license text.

```
SPDX-License-Identifier: GPL-3.0-or-later
```

### Upstream Project

This project is based on:

- **OpenMX** (Open source package for Material eXplorer)
- **Version**: 4.0
- **Upstream**: <https://www.openmx-square.org/>
- **Copyright**: Copyright (c) Taisuke Ozaki and OpenMX contributors
- **License**: GNU General Public License v3.0 or later

The original OpenMX source code retains its original copyright notices and
license headers. Modifications made in this project (GPU acceleration for
NVIDIA CUDA and AMD HIP backends) are also licensed under GPL-3.0-or-later.

### Third-Party Components

This project incorporates source code from the following third-party projects.
Each component retains its original license; their original license files are
preserved under `third_party/<component>/`.

#### FFTW3

- **Source**: <https://www.fftw.org/>
- **Copyright**:
  - Copyright (c) 2003, 2007-14 Matteo Frigo
  - Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
- **License**: GNU General Public License v2.0 or later (`GPL-2.0-or-later`)
- **Used here under**: GPL v3 (selected via the "or any later version" clause)
- **Original license file**: [`third_party/fftw3/COPYING`](third_party/fftw3/COPYING)

#### GEMMul8

- **Source**: <https://github.com/RIKEN-RCCS/GEMMul8>
- **Copyright**: Copyright (c) 2025- RIKEN R-CCS
- **Responsible developer**: Yuki Uchino (RIKEN Center for Computational Science)
- **License**: MIT License (`MIT`)
- **Original license file**: [`third_party/gemmul8/LICENSE`](third_party/gemmul8/LICENSE)

GEMMul8 (GEMMulate) is a library for emulating high-precision matrix
multiplication (SGEMM, DGEMM, CGEMM, ZGEMM) using INT8/FP8 matrix engines
based on the Ozaki Scheme II. 

If you use this project in academic work, please also cite the GEMMul8
upstream references (see the GEMMul8 repository).

#### ELPA / COSMA (the "gpusolver2" stack, since v2.0)

The distributed multi-GPU cluster diagonalization (`scf.eigen.lib gpusolver2`)
links four additional libraries, bundled as source archives under
`source/third_party/dist/` and built automatically by the first `make`; each
archive carries its original license file:

- **ELPA** 2026.02 — <https://elpa.mpcdf.mpg.de/> — Copyright (c) the ELPA
  consortium — **GNU Lesser General Public License v3.0 (`LGPL-3.0`)**
  (`COPYING/` inside the archive). The conventional "elpa1"/"elpa2" CPU paths
  use the ELPA 2018.05 source embedded in upstream OpenMX under the same
  license.
- **COSMA** 2.8.4 — <https://github.com/eth-cscs/COSMA> — Copyright (c) ETH
  Zürich (parts: Advanced Micro Devices, Inc.) — **BSD 3-Clause License
  (`BSD-3-Clause`)**
- **COSTA** and **Tiled-MM** (COSMA dependencies) —
  <https://github.com/eth-cscs/COSTA>,
  <https://github.com/eth-cscs/Tiled-MM> — Copyright (c) ETH Zürich —
  **BSD 3-Clause License (`BSD-3-Clause`)**

### License Compatibility Summary

| Component | Original License | Compatible with GPL v3 |
|-----------|------------------|------------------------|
| OpenMX (upstream) | GPL-3.0-or-later | — (same license) |
| GPU modifications (this project) | GPL-3.0-or-later | — (same license) |
| FFTW3 | GPL-2.0-or-later | Yes (via "or later" clause) |
| GEMMul8 | MIT | Yes (permissive → strong copyleft) |
| ELPA (2018.05 embedded; 2026.02 for "gpusolver2") | LGPL-3.0 | Yes (LGPL v3 code may be conveyed under GPL v3) |
| COSMA (with COSTA, Tiled-MM) | BSD-3-Clause | Yes (permissive → strong copyleft) |

The combined work is distributed under the terms of GPL v3 or later.