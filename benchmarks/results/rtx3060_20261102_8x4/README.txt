RTX 3060 / Threadripper 3970X benchmark, 8 MPI x 4 OpenMP (2026-11-02)

Source commit: 5afbb9dde927876064a6dcd260d9efc0c2b22533
Completed: 12/12 jobs, 96/96 case executions.
All timings are fresh single measurements. Jobs run sequentially on 32 physical cores.
S/L each use one persistent native -runtest/-runtestL MPI invocation per route.
Suite totals sum native result-file case times (rounded to 0.01 s); named input times use output Elapsed.Time (0.001 s).
run-results.csv also retains launcher wall times and the sum of full-precision output elapsed times.
CPU routes use elpa2; GPU uses gpusolver, private NVIDIA MPS and unchanged default GEMMul8 workspace policy.
CUDA is the apt-installed /usr/local/cuda-13.4. Fork CPU/GPU use the same frozen binary and Core/SIMD libraries.
Upstream is the existing ~/openmx4.0/work/openmx, AOCC 5.2.0 / threaded AOCL 5.3.0 / Open MPI 5.0.10.
Inputs, SCF/MD limits, convergence criteria and the same 44 PAO/VPS datasets are frozen for all routes.
Elapsed-time ratios include differences in SCF iteration counts; case-results.csv records these counts for every case.
In runs.json, scf_history contains [SCF iteration, printed NormRD] pairs; it is not an energy-change series.
sidia333 retains scf.maxIter=25 and scf.criterion=1e-13; normal process completion does not imply SCF convergence.

Workload | Upstream CPU (s) | Fork CPU (s) | GPU + MPS (s) | Upstream/GPU | Fork CPU/GPU
S | 104.190 | 74.010 | 95.310 | 1.093 | 0.777
L | 9239.380 | 6362.270 | 6674.210 | 1.384 | 0.953
sidia333_nc_cluster | 1240.057 | 893.258 | 291.544 | 4.253 | 3.064
cghelgeo | 7771.920 | 10383.035 | 1645.950 | 4.722 | 6.308

Primary three-route comparisons at 1e-7 energy/force: 94/96 PASS.
Historical bundled references: 90/90 PASS.
Maximum absolute energy difference: 1.32241E-7 Ha.
Maximum absolute force-component difference: 3.3776E-8 Ha/bohr.
Every force component, atom identity/printed coordinates and grid was checked; the native suite force-sum check is retained separately.
All numerical mismatches and unconverged calculations are listed in validation.json.
Raw logs, all output forces/energies, process library audits and MPS monitoring: /home/dc1394/openmx4.0_gpu/work/benchmarks_rtx3060_20261102_8x4
Other-environment results were pulled and reviewed before this campaign; their timings were not reused.
Measurements use the pulled source and the configuration recorded in build-manifest.json.

Comparisons exceeding the fixed 1e-7 thresholds (inputs and thresholds were not changed):
cghelgeo / cghelgeo / upstream_cpu vs fork_cpu: delta E=1.03513E-7 Ha; max delta force=9.588E-9 Ha/bohr
cghelgeo / cghelgeo / upstream_cpu vs fork_gpu: delta E=1.32241E-7 Ha; max delta force=2.9922E-8 Ha/bohr

Cases with different total SCF iteration counts across routes:
S / Crys-MnO: upstream_cpu=37, fork_cpu=39, fork_gpu=37
L / 5_5_13COb2: upstream_cpu=67, fork_cpu=66, fork_gpu=66
L / B2C62_Band: upstream_cpu=203, fork_cpu=203, fork_gpu=204
L / GGFF: upstream_cpu=98, fork_cpu=97, fork_gpu=101
L / N1C999: upstream_cpu=85, fork_cpu=59, fork_gpu=59
L / Ni63-O64: upstream_cpu=70, fork_cpu=64, fork_gpu=62
L / Pt63: upstream_cpu=73, fork_cpu=77, fork_gpu=70
cghelgeo / cghelgeo: upstream_cpu=63, fork_cpu=60, fork_gpu=58

SCF limits reached without convergence (elapsed-time ratios include these iteration counts):
sidia333_nc_cluster / upstream_cpu / sidia333_nc_cluster: 25 SCF steps, 1 unconverged MD steps
sidia333_nc_cluster / fork_cpu / sidia333_nc_cluster: 25 SCF steps, 1 unconverged MD steps
sidia333_nc_cluster / fork_gpu / sidia333_nc_cluster: 25 SCF steps, 1 unconverged MD steps
