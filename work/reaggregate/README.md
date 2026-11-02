# Re-aggregated CSV files of the measurement archive

`reaggregate.py` reads the individual outputs of the measurement archive
(the `.out` timing tables and SCF histories, the `.manifest.json` records,
the `.smi` memory samplers, the `.env` node lists, the `.joblog` TIMING
lines, the archive's own aggregation tables and the stand-alone GEMM CSVs)
and writes the CSV files that the paper's appendix on provenance names.
Nothing is recomputed beyond parsing and taking means; every number can be
traced to the file named in the row (`out_file`, `manifest_file`), and
`reaggregate_sha256.txt` lists the SHA-256 of every file the script read.

Regenerate with

```sh
python3 work/reaggregate/reaggregate.py <archive root containing work/> --out work/reaggregate
```

(2026-11-02: revision-5 archive root; 4403 input files; `reaggregate_check.txt`
reports 0 discrepancies between the `.out` tables, the joblog TIMING lines
and the manifest walls.)

## Files

| File | Rows | Content |
| --- | --- | --- |
| `production_runs.csv` | one per run of `sip_*` (H100), `s5p_*` and `s5x_*` (RTX 5080): 395 | `total_s`, `dft_s`, `diag_s`, `set_hamiltonian_s` = the `Max_Time` column of the `.out` table (the slowest rank); SCF count, printed `Utot`; manifest counters (dense path, GEMMul8 calls and fallbacks, workspace policy, memory peaks, binary MD5, tag); the `.smi` maxima; node list and MPS evidence of the `.env`. `retained` = listed in the archive's thesis aggregation table named in `retained_in` (the np18 RTX 5080 series are retained by the policy of `s5prod_thesis_result_v2.txt`); `paper_final` = the rank count the paper's tables use (H100 48; RTX 5080 18 for bcol216/ccol216/cnc216, else 16). |
| `production_summary.csv` | one per (machine, system, config, ranks) over the retained runs: 64 | mean and sample SD of the four times, the speedups quoted in the tables and figures (`speedup_vs_cpu_*` against the CPU reference of the same machine and rank count, `speedup_vs_cublas_*` for GEMMul8, `mps_off_over_on_*` for the MPS-off series), fallback totals, memory high-water maxima. |
| `amdahl_sites.csv` | one per GEMM call site of the Amdahl-type model, both GPUs: 53 | shape and transposition flags, calls per run and GEMMul8 calls (run manifests; bnc216 on the RTX 5080: 7 of 200 back-transforms), stand-alone times from `gemm_micro_5080.csv` / `gemm_micro_h100.csv`, and the mean walls used by `analysis.py` / `analysis_h100.py`. The cnc128 sites were measured on 2026-11-01 (`gemm_micro_5080_cnc128_result.txt`). |
| `accuracy_runs.csv` | one per run of the accuracy series `siacc_*`, `s5a_*` (criterion 1e-13), `siacc8_*`, `s5a8_*` (practical criterion 1e-8) and `siacc64_*`: 88 | SCF count, `converged_by_criterion` (terminated before `scf.maxIter`), last NormRD, printed `Utot`, `|F|max`, GEMMul8 fallbacks. |
| `accuracy_comparison.csv` | one per pair: 77 | decimal differences of the printed `Utot` and the maximum difference of the printed force components for the pairs C-O, G-O, C-G (retained runs, lowest repetition) and the run-to-run pairs of the same configuration. |
| `strong_scaling_runs.csv` | one per run of `sibs_*`: 140 (90 retained) | the H100 band strong-scaling series (1/2/4/8 nodes, cuBLAS and GEMMul8). |
| `distributed_solver_runs.csv` | one per run of `s443nc*`: 99 (81 retained) | the cnc384 single-binary series: 16 ranks per node (`ranks_per_node_label` 16), the 48-ranks-per-node runs, the 36-rank comparison and the aborted `s443nc48_*` namespace (`s443nc48_status.txt`). |
| `supplementary_series_runs.csv` | every other run directory with an output: 288 | exploratory cluster sizes (`s333nc`, `s444nc`, `s444cl`), DC-LNO node scaling (`dia64dc`), geometry-optimization checks (`cghelgeo`), the band/cluster size-search probes (`sib_*`) and the RTX 5080 preflight runs (`s5f_*`). |
| `reaggregate_sha256.txt` | 4403 | SHA-256 and archive-relative path of every input file read. |
| `reaggregate_check.txt` | | counts, the timing cross-check and the paper-final summary rows. |

Times are seconds; `Utot` in Hartree; forces in Hartree/Bohr. Empty
fields mean the source does not record the quantity (for example the
older repetitions without a manifest).
