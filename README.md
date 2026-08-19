# openmx4.0_gpu
## What does this code do?
This is a GPU-accelerated version of [OpenMX](https://www.openmx-square.org/), a first-principles calculation code based on numerical atomic orbitals (NAO). GPU fast paths now cover most of the SCF workflow for band and cluster calculations (collinear and non-collinear):

- the dense eigensolvers (cuSOLVER `Xsyevdx`) of the four global solver paths — band/cluster x collinear/non-collinear — plus per-atom GPU eigensolves in the DC, DC-LNO and Krylov solvers, and an optional distributed multi-GPU cluster path (`scf.eigen.lib gpusolver2`: ELPA GPU kernels + COSMA);
- the Hamiltonian matrix elements (`Set_Hamiltonian`, with SCF-invariant tables kept device-resident), the nonlocal-projector and VNA-projector integrals, the electron-density grid quadrature, density-matrix accumulation, charge mixing, the total-energy terms and most force parts;
- all dense GPU matrix multiplications can optionally run through [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) INT8-based FP64 GEMM emulation (see below).

Every phase keeps a CPU implementation and falls back per rank / per call when a gate condition or a device-memory preflight fails (memory-aware hybrid execution); each run writes a machine-readable `<System.Name>.manifest.json` that records which paths actually executed.

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

A run that finds no usable GPU demotes itself to ELPA2 automatically, so the default is also safe on machines without an NVIDIA GPU.

## FP64 GEMM emulation with GEMMul8
On GPUs whose INT8 tensor-core throughput is much higher than their native FP64 throughput, the dense GPU matrix multiplications can be emulated at FP64 accuracy from INT8 building blocks (Ozaki scheme II). This is controlled by one input keyword:

```ini
scf.gemmul8.enable         on          # default=on; off = plain cuBLAS FP64
```

The default configuration (INT8 backend, `num_moduli=15`, fast mode off) reproduces plain cuBLAS FP64 results at the application level; when the GEMMul8 workspace does not fit beside the other device allocations, the affected GEMM silently and safely falls back to native cuBLAS (the fallback is counted in the run manifest).

## Run manifest
Every run writes `<System.Name>.manifest.json` next to its other output files: build/commit identification, the input checksum, the MPI/GPU layout (including MPS detection), which dense-solver path ran, per-phase GPU/CPU dispatch and fallback counters, GEMMul8 call/fallback statistics, peak device memory and per-rank host memory, per-phase wall times, and a snapshot of all `OPENMX_*`/`GEMMUL8_*`/`CUDA_*`/`OMP_*` environment variables. A "GPU-accelerated" label never has to be taken on faith — check the manifest.

## Multi-GPU parallelization
How many GPUs a run can actually use is bounded by the number of k-points requested with "scf.Kgrid". The MPI ranks are divided into one group per k-point, and the dense eigenvalue problem of each group is solved on a single GPU, so the eigenvalue solver keeps at most as many GPUs busy as there are k-points; any GPU beyond that number stays idle in this part of the calculation. (The Hamiltonian matrix elements and the grid work are distributed over all MPI ranks, and therefore over all GPUs.)

In particular, a cluster calculation — `scf.EigenvalueSolver cluster`, i.e. `scf.Kgrid 1 1 1` — has only one k-point, so **only one GPU is used for the diagonalization however many GPUs the node has**. Adding GPUs, or raising the upper bound `scf.Gpu.Num` (default 30, which effectively means "use every GPU found"), does not make such a run faster. (A collinear spin-polarized calculation solves the two spins in separate MPI worlds, so it can occupy two GPUs at most.)

Multiple GPUs therefore pay off for band calculations with a k-mesh; for a cluster calculation, give the job one GPU and more CPU cores / MPI ranks instead.

## MPI vs. hybrid (MPI/OpenMP) parallelization
Use flat MPI. In OpenMX 4.0 GPU, hybrid MPI/OpenMP parallelization is not effective: OpenMP threads can still be requested as usual with the `-nt` option and such runs complete correctly, but they bring no speedup — only MPI parallelization is effective. Assign all the cores you want to use to MPI ranks instead, with one OpenMP thread per rank:

```sh
mpirun -np 16 ./openmx input.dat -nt 1
```

## Build and install
Building and installing is more difficult than with standard OpenMX. The build requires the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) and OpenMPI. The Makefile contains build examples for several supercomputer systems; please refer to them. If you're unsure about the build and installation process, feel free to ask in English via GitHub issues or [my X account](https://x.com/dc1394) (Japanese is also acceptable on my X account). I'll assist you as much as I can.

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
NVIDIA CUDA backends) are also licensed under GPL-3.0-or-later.

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

### License Compatibility Summary

| Component | Original License | Compatible with GPL v3 |
|-----------|------------------|------------------------|
| OpenMX (upstream) | GPL-3.0-or-later | — (same license) |
| GPU modifications (this project) | GPL-3.0-or-later | — (same license) |
| FFTW3 | GPL-2.0-or-later | Yes (via "or later" clause) |
| GEMMul8 | MIT | Yes (permissive → strong copyleft) |

The combined work is distributed under the terms of GPL v3 or later.