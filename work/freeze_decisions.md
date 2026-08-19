# v2.0_thesis freeze decisions (plan v2.6 sec. 3.2 / WBS items 2 and 7)

Recorded 2026-08-19 from KAWAI-san's instructions in-session.

## 1. AMD exclusion policy: option (1)

Decision: **(1) release branch with the AMD/HIP files removed carries the
v2.0_thesis tag.** main keeps the AMD files; the Zenodo archive is built
from the release branch, so tag = archive = paper artifact with no
filtering step.

Execution (at freeze time, after WBS items 3-6 land):
- `git checkout -b release/v2.0_thesis`
- remove the AMD/HIP sources (enumerate with `grep -ril 'hip\|rocm' source/`
  plus the AMD entries of the Makefile), commit
- tag `v2.0_thesis` on that branch; production tables are then re-measured
  from the tag per plan sec. 5.

## 2. Tag contents: all measurement inputs AND results

Instruction (2026-08-19): "tag 2.0_thesis には測定に使った全てのファイル
(*.dat と *.sh) と測定結果をすべて含めてください。"

The release branch therefore additionally commits, under `work/`:
- every measurement deck `*.dat` and job script `*.sh` (generators
  `gen_*.sh`, summarizers `summarize_*.sh`, `report_*.sh`, `mps_node.sh`,
  `nsys_wrap.sh` included),
- all result aggregates: `*_result.txt`, `*.plt`, `*.png`,
  `siband_jobids.txt`, `siband_candidates.txt`, provenance notes,
- per-run primary evidence: `*.joblog`, `*.out`, `*.env`, `*.smi`,
  `*.std`, `*.err`, `*.progress`, `*.ene`, `*.md`/`*.md2` of every
  campaign run directory (sib_/sibs_/sic_/sicpu_/sip_/siacc_/simani_/
  sinsys_),
- documents: `gpu_phase_audit.md`, `run_manifest_design.md`, this file.

Excluded (size, reproducible from the above): bulky volumetric outputs
(`*.cube`, `*.xsf`), restart directories (`*_rst/`), `*.dat#` input echoes
if space demands, and binaries (`openmx`, `*.bak`).  Size audit before the
freeze decides whether `.cube` files really must stay out; if they fit the
repository they go in too ("すべて" is the default, exclusion only under
a hard size constraint, documented here).

## 3. Related state

- P0 run manifest implemented 2026-08-19 (openmx_common.c registry +
  <System.Name>.manifest.json; validation campaign simani_*).
- Binary before P0: md5 962f8d2519c2e6aa5a6295513f76fee9 (backed up as
  work/openmx.962f8d25.bak); after P0: 13031e1e7b075b447fe9cf345fafca92.
- All Pegasus measurement campaigns of plan v2.6 done pre-P0; production
  tables will be re-measured from the tag with the instrumented binary.
