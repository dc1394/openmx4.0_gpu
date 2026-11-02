#!/usr/bin/env python3
"""M3 assembly, H100 leg: combine gemm_micro_h100.csv with the v2.0_thesis
production-matrix manifests/walls (work/siprod_thesis_result.txt, np48,
n=5 means) into the plan-sec.-14.3 lightweight Amdahl model -- the exact
H100 counterpart of analysis.py (RTX 5080).

    S_pred = 1 / ((1-f) + f/s_eff)

Differences from the 5080 leg, both physical not methodological:
  - every H100 g run has ZERO GEMMul8 fallbacks (80 GB), so the g-leg mix
    is pure G8 -- bnc216 on the H100 realizes the 5080 doc's "bnc216*"
    zero-fallback hypothetical;
  - the walls are per-system np48 (whole node), not per-system-best np.
Writes amdahl_h100.csv and prints a human-readable table.
"""
import csv
import sys

CSVFILE = sys.argv[1] if len(sys.argv) > 1 else "gemm_micro_h100.csv"

# ---- measured campaign walls (np48; seconds): the means of the five retained
# runs of each configuration at full precision, from the individual outputs
# (work/reaggregate/production_runs.csv; a mean of five millisecond times is
# exact at four decimals).  Until 2026-11-02 these were the two-decimal means
# of siprod_thesis_result.txt; the paper's Fig. 4 uses the full-precision means.
# system: (np, total_o, diag_o, total_g, diag_g)
WALL = {
    "bcol216": (48, 81.4546, 37.0774, 81.3278, 36.7692),
    "bcol300": (48, 131.367, 83.0376, 151.0346, 81.6544),
    "bnc216":  (48, 252.141, 153.8086, 255.8168, 157.2112),
    "ccol216": (48, 51.3468, 8.2918, 51.9024, 8.1402),
    "ccol360": (48, 69.5634, 16.0054, 69.532, 16.221),
    "cnc216":  (48, 100.518, 31.1026, 99.912, 30.853),
    "cnc300":  (48, 135.5192, 57.0272, 135.1304, 56.8828),
}

# ---- GEMM sites per production run: (csv label, calls, g8_calls_in_g_leg) ----
# calls from the sip_*_g11 manifests (gemmul8.d_calls/z_calls; d_fallbacks =
# z_fallbacks = 0 on every H100 run, so g8_calls == calls throughout).
SITES = {
    "bcol216": [("bcol216_fwd1", 200, 200), ("bcol216_fwd2", 200, 200), ("bcol216_back", 200, 200)],
    "bcol300": [("bcol300_fwd1", 200, 200), ("bcol300_fwd2", 200, 200), ("bcol300_back", 200, 200)],
    "bnc216":  [("bnc216_fwd1", 600, 600), ("bnc216_fwd2", 600, 600), ("bnc216_back", 200, 200)],
    "ccol216": [("ccol216_fwd1", 25, 25), ("ccol216_fwd2", 25, 25), ("ccol216_back", 24, 24),
                ("ccol216_back1", 1, 1)],
    "ccol360": [("ccol360_fwd1", 25, 25), ("ccol360_fwd2", 25, 25), ("ccol360_back", 24, 24),
                ("ccol360_back1", 1, 1)],
    "cnc216":  [("cnc216_dfwd1", 150, 150), ("cnc216_dfwd2", 150, 150), ("cnc216_zback", 24, 24),
                ("cnc216_zback1", 1, 1)],
    "cnc300":  [("cnc300_dfwd1", 150, 150), ("cnc300_dfwd2", 150, 150), ("cnc300_zback", 24, 24),
                ("cnc300_zback1", 1, 1)],
}

rows = {}
with open(CSVFILE) as f:
    for r in csv.DictReader(f):
        rows[r["label"]] = r

def t_ms(label, col):
    return float(rows[label][col])

out = []
print(f"{'system':<9} {'T_gemm_cu':>9} {'T_gemm_g8':>9} {'s_eff':>6} "
      f"{'f_diag':>6} {'Sp_diag':>7} {'Sm_diag':>7} {'f_tot':>6} {'Sp_tot':>6} {'Sm_tot':>6}")
for sys_name, sites in SITES.items():
    np_, total_o, diag_o, total_g, diag_g = WALL[sys_name]
    t_cu = sum(c * t_ms(lbl, "cublas_ms_med") for lbl, c, _ in sites) / 1000.0
    t_g8 = sum((g8c * t_ms(lbl, "g8_ms_med") + (c - g8c) * t_ms(lbl, "cublas_ms_med")) for lbl, c, g8c in sites) / 1000.0
    s_eff = t_cu / t_g8
    f_d = t_cu / diag_o
    f_t = t_cu / total_o
    sp_d = 1.0 / ((1.0 - f_d) + f_d / s_eff)
    sp_t = 1.0 / ((1.0 - f_t) + f_t / s_eff)
    sm_d = diag_o / diag_g
    sm_t = total_o / total_g
    print(f"{sys_name:<9} {t_cu:>9.2f} {t_g8:>9.2f} {s_eff:>6.2f} "
          f"{f_d:>6.3f} {sp_d:>7.3f} {sm_d:>7.3f} {f_t:>6.3f} {sp_t:>6.3f} {sm_t:>6.3f}")
    out.append(dict(system=sys_name, np=np_, T_gemm_cublas_s=round(t_cu, 3), T_gemm_g8mix_s=round(t_g8, 3),
                    s_eff=round(s_eff, 3), f_diag=round(f_d, 4), S_pred_diag=round(sp_d, 3),
                    S_meas_diag=round(sm_d, 3), f_total=round(f_t, 4), S_pred_total=round(sp_t, 3),
                    S_meas_total=round(sm_t, 3)))

with open("amdahl_h100.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print("\nwrote amdahl_h100.csv")
