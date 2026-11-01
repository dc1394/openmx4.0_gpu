Fresh Sirius MI300A GPU benchmark, 2026-11-01

All 59 cases were newly executed under ordinary scf.eigen.lib gpusolver.
S/L/sidia333/cghelgeo: 1 physical node, 4 GPUs, 32 MPI ranks x 3 threads.
L2/L3: 2 physical nodes, 8 GPUs, 64 MPI ranks total, 32 x 3 per node.
S/L/named CPU results in README.md reuse the completed 2026-10-31 measurements.
Both fork CPU and upstream CPU L3 suites were freshly measured. No other
CPU suite, gpusolver2 or gpusolver3 timing was added.

Production numerical source: 5925e71a; task checkout: c6648c29.
The same validated frozen executable and Core/SIMD libraries were reused;
none of the old GPU timing results was used as a fresh measurement.
manifest.json records source, binary, input and dataset hashes.
gpu_*.json retain every energy and force, SCF history, exact command,
BLAS provider probe, environment, node layout and workload wall time.
launch-environments.json retains all 384 rank affinity records.
Native suite per-case rows (2 decimals) are summed for suite totals;
named input times use output Elapsed.Time (3 decimals).

The independent new allowance is 100 points. A conservative pre-submit
GPU estimate was 73.82 points. With the later CPU L3 extension, the peak
cumulative actual spending plus outstanding hard reservations was 98.75.
The actual campaign cost, rounded upward per job, was 61.47 points.
jobs.json records the site's GPU count, coefficient, wall time and exit.
The pre-existing official project-balance floor of 130 remained enabled.
Budget monitor logs and raw numerical logs remain in /work/GPU2026/dc1394/openmx4.0_gpu/work/codex_gpu_rebench_20261101_1318.

Validation: validation.json, cases.csv, comparisons.csv,
current-three-route-comparisons.csv, bundled-reference-comparisons.csv
and previous-prefix-comparisons.csv.
Primary current trio: 96 strict comparisons, all pass (energy/force 1e-7).
Fresh GPU vs three earlier routes and previous optimized GPU: 128 pass.
GPU bundled L2/L3 reference checks: 25 pass, 2 mismatch.
Fresh L3 three-route comparisons: 55 pass, 5 mismatch.
CPU L2 was explicitly deferred by the user; only the GPU suite is in scope.
Original SCF/MD criteria were not changed. L3 inputs retain maxIter=3;
sidia333 retains maxIter=25. Read convergence separately from numerical
agreement, and do not interpret a completed process as a converged SCF.

scripts/ preserves the specific campaign launch and collection helpers.
They reference the frozen runtime_source assets named in manifest.json.
They are measurement provenance, not a standalone installation workflow.
